"""Split generated simulations into train/validation/test and export their arrays."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from solver.gen_data.pipeline.build_dataset import build_dataset


def build_paper_dataset(
    summary_paths: Sequence[Path],
    *,
    output_root: Path,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Path:
    """Pool completed runs without imposing any family proportions."""
    if not (
        0 <= validation_fraction < 1
        and 0 <= test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError(
            "validation/test fractions must be nonnegative and sum to less than one"
        )
    batch_paths: list[Path] = []
    runs: set[tuple[str, int]] = set()
    for summary_path in summary_paths:
        path = summary_path.expanduser().resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_spec = summary["run_spec"]
        identity = (run_spec["family_name"], run_spec["seed"])
        if identity in runs:
            raise ValueError("generation runs must use distinct family/seed pairs")
        runs.add(identity)
        batch_paths.extend(
            (path.parent / value).resolve() for value in summary["batch_paths"]
        )
    return build_dataset(
        output_root,
        batch_paths,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-summary",
        type=Path,
        action="append",
        required=True,
        help="Completed generation summary; repeat for the runs to combine.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=42, help="Fixed simulation-split seed."
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    dataset = build_paper_dataset(
        args.run_summary,
        output_root=args.output_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    print(json.dumps({"dataset": str(dataset)}))
