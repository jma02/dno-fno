"""Audit the predeclared two-stream JONSWAP revision-4 fresh gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_SAMPLING_REVISION_V4,
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)


JsonRecord: TypeAlias = dict[str, object]
STREAM_IDS = (941, 942)
ACCEPTED_PER_CELL_PER_STREAM = 10
EXPECTED_ACCEPTED_PER_STREAM = 270
EXPECTED_BATCH_SIZE = 32
EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE = 4
MAXIMUM_AGGREGATE_REJECTION_RATE = 0.02
SUMMARY_NAME = "paper_dataset_jonswap_tma_validation.summary.json"
AGGREGATE_NAME = "jonswap_relative_band_fresh_gate.summary.json"


def _strict_json_object(path: Path) -> JsonRecord:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_stream(
    summary: Mapping[str, object],
    *,
    stream_id: int,
) -> JsonRecord:
    if summary.get("status") != "complete":
        raise ValueError(f"stream {stream_id} is not complete")
    run_spec = _mapping(summary.get("run_spec"), name="run_spec")
    expected_identity = {
        "family_name": "jonswap_tma",
        "family_id": 4,
        "revision_id": JONSWAP_TMA_SAMPLING_REVISION_V4,
        "split_id": "validation",
        "stream_id": stream_id,
        "batch_size": EXPECTED_BATCH_SIZE,
        "maximum_attempts_per_accepted_case": (
            EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
        ),
    }
    observed_identity = {key: run_spec.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(
            f"stream {stream_id} run identity differs from the frozen gate"
        )

    expected_cells = JONSWAP_TMA_SAMPLE_CELL_IDS
    expected_quotas = [
        {
            "cell_id": cell_id,
            "target_accepted": ACCEPTED_PER_CELL_PER_STREAM,
        }
        for cell_id in expected_cells
    ]
    if run_spec.get("quotas") != expected_quotas:
        raise ValueError(f"stream {stream_id} quotas differ from the frozen gate")

    counts = _mapping(summary.get("counts"), name="counts")
    accepted = _integer(counts.get("accepted"), name="counts.accepted")
    attempted = _integer(counts.get("attempted"), name="counts.attempted")
    rejected = _integer(counts.get("rejected"), name="counts.rejected")
    if accepted != EXPECTED_ACCEPTED_PER_STREAM:
        raise ValueError(f"stream {stream_id} accepted count is not 270")
    if attempted != accepted + rejected:
        raise ValueError(f"stream {stream_id} counts do not add up")

    by_cell = _mapping(counts.get("by_cell"), name="counts.by_cell")
    if set(by_cell) != set(expected_cells):
        raise ValueError(f"stream {stream_id} cells differ from the gate")
    for cell_id in expected_cells:
        cell_counts = _mapping(by_cell[cell_id], name=f"by_cell[{cell_id}]")
        cell_accepted = _integer(
            cell_counts.get("accepted"),
            name=f"by_cell[{cell_id}].accepted",
        )
        cell_attempted = _integer(
            cell_counts.get("attempted"),
            name=f"by_cell[{cell_id}].attempted",
        )
        target = _integer(
            cell_counts.get("target_accepted"),
            name=f"by_cell[{cell_id}].target_accepted",
        )
        if cell_accepted != ACCEPTED_PER_CELL_PER_STREAM or target != cell_accepted:
            raise ValueError(f"stream {stream_id} did not fill cell {cell_id}")
        if cell_attempted > (EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE * target):
            raise ValueError(f"stream {stream_id} exhausted cell {cell_id}")

    raw_reasons = _mapping(
        counts.get("rejection_reasons"),
        name="counts.rejection_reasons",
    )
    reasons = {
        str(reason): _integer(count, name=f"rejection_reasons[{reason}]")
        for reason, count in raw_reasons.items()
    }
    return {
        "stream_id": stream_id,
        "configuration_fingerprint": summary.get("configuration_fingerprint"),
        "accepted": accepted,
        "attempted": attempted,
        "rejected": rejected,
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def audit(output_root: Path, *, write: bool) -> JsonRecord:
    """Return the aggregate gate decision, or an in-progress record."""

    selected_root = output_root.expanduser().resolve()
    summary_paths = {
        stream_id: selected_root / f"stream_{stream_id}" / SUMMARY_NAME
        for stream_id in STREAM_IDS
    }
    missing = [str(path) for path in summary_paths.values() if not path.is_file()]
    if missing:
        return {
            "schema": "jonswap_relative_band_fresh_gate_summary_v1",
            "status": "in_progress",
            "missing_summaries": missing,
        }

    streams = tuple(
        _validate_stream(
            _strict_json_object(summary_paths[stream_id]),
            stream_id=stream_id,
        )
        for stream_id in STREAM_IDS
    )
    accepted = sum(int(stream["accepted"]) for stream in streams)
    attempted = sum(int(stream["attempted"]) for stream in streams)
    rejected = sum(int(stream["rejected"]) for stream in streams)
    rejection_reasons: Counter[str] = Counter()
    for stream in streams:
        raw_reasons = _mapping(
            stream["rejection_reasons"],
            name="stream.rejection_reasons",
        )
        rejection_reasons.update(
            {str(reason): int(count) for reason, count in raw_reasons.items()}
        )
    rejection_rate = rejected / attempted
    passed = rejection_rate <= MAXIMUM_AGGREGATE_REJECTION_RATE
    result: JsonRecord = {
        "schema": "jonswap_relative_band_fresh_gate_summary_v1",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "predeclared_gate": {
            "accepted_per_cell": 2 * ACCEPTED_PER_CELL_PER_STREAM,
            "accepted_total": 2 * EXPECTED_ACCEPTED_PER_STREAM,
            "maximum_rejection_rate": MAXIMUM_AGGREGATE_REJECTION_RATE,
            "maximum_attempts_per_accepted_case": (
                EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
            ),
        },
        "observed": {
            "accepted": accepted,
            "attempted": attempted,
            "rejected": rejected,
            "rejection_rate": rejection_rate,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "streams": list(streams),
        "source": {
            "auditor_sha256": file_sha256(Path(__file__)),
            "stream_summary_sha256": {
                str(stream_id): file_sha256(summary_paths[stream_id])
                for stream_id in STREAM_IDS
            },
        },
    }
    if write:
        write_json_atomic(selected_root / AGGREGATE_NAME, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit(args.output_root, write=bool(args.write))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
