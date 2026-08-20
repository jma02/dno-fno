"""Cross-check the frozen paper-corpus layout at every release boundary."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import stat
from typing import Final, Sequence
import unittest


# Importing the family auditors initializes JAX.  Contract tests must not
# compete with the live numerical generators for a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"

from scripts import audit_completed_benjamin_feir_revision4 as bf_audit  # noqa: E402
from scripts import audit_completed_jonswap_revision4 as jonswap_audit  # noqa: E402
from scripts import audit_completed_tanaka_revision3 as tanaka_audit  # noqa: E402
from scripts import audit_paper_corpus_training_handoff as handoff  # noqa: E402
from scripts import bind_completed_stokes_revision2_audit as stokes_audit  # noqa: E402
from scripts import run_paper_corpus_postcompletion as postcompletion  # noqa: E402
from scripts import watch_paper_corpus_workers as watchdog  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    PhysicalFamilyId,
    paper_corpus_revision_id,
)


ROOT: Final = Path(__file__).resolve().parents[1]
JONSWAP_LAUNCHER: Final = ROOT / "scripts/launch_revision4_jonswap_bulk.sh"
TANAKA_LAUNCHER: Final = ROOT / "scripts/launch_revision3_tanaka_bulk.sh"
VIEW_BUILDER: Final = ROOT / "scripts/build_literature_aligned_paper_corpus_views.sh"
EXECUTABLE_ENTRYPOINTS: Final = (
    JONSWAP_LAUNCHER,
    TANAKA_LAUNCHER,
    ROOT / "scripts/launch_tanaka_after_revision4_jonswap.sh",
    VIEW_BUILDER,
    ROOT / "scripts/launch_paper_corpus_postcompletion.sh",
)


@dataclass(frozen=True)
class ChunkContract:
    """One source summary and its additive population coordinates."""

    split: str
    accepted_before: int
    accepted_count: int
    stream_id: int
    relative_summary: Path

    @property
    def layout(self) -> tuple[str, int, int, int]:
        return (
            self.split,
            self.accepted_before,
            self.accepted_count,
            self.stream_id,
        )

    @property
    def relative_root(self) -> Path:
        return self.relative_summary.parent


@dataclass(frozen=True)
class LauncherChunk:
    """One row from a current two-lane bulk launcher."""

    lane: int
    split: str
    accepted_count: int
    accepted_before: int
    stream_id: int
    relative_root: Path
    label: str

    @property
    def layout(self) -> tuple[str, int, int, int]:
        return (
            self.split,
            self.accepted_before,
            self.accepted_count,
            self.stream_id,
        )


def _shell_array_words(source: str, name: str) -> tuple[str, ...]:
    """Let ``shlex`` decode one simple Bash array without executing it."""

    match = re.search(
        rf"(?ms)^[ \t]*{re.escape(name)}=\(\n(?P<body>.*?)^[ \t]*\)\n",
        source,
    )
    if match is None:
        raise AssertionError(f"missing shell array {name}")
    return tuple(shlex.split(match.group("body"), comments=True, posix=True))


def _shell_root_default(source: str, name: str) -> Path:
    """Resolve a ``$ROOT/...`` scalar or overridable scalar default."""

    match = re.search(rf"(?m)^{re.escape(name)}=(?P<value>[^\n]+)$", source)
    if match is None:
        raise AssertionError(f"missing shell scalar {name}")
    value = match.group("value")
    overridable_prefix = f"${{{name}:-$ROOT/"
    if value.startswith(overridable_prefix) and value.endswith("}"):
        relative = value[len(overridable_prefix) : -1]
    elif value.startswith("$ROOT/"):
        relative = value[len("$ROOT/") :]
    else:
        raise AssertionError(f"{name} is not a repository-root default: {value}")
    return (ROOT / relative).resolve()


def _launcher_chunks(source: str) -> tuple[LauncherChunk, ...]:
    """Decode the seven-field semantic rows in a bulk launcher."""

    def parse(row: str) -> LauncherChunk:
        fields = shlex.split(row, comments=True, posix=True)
        if len(fields) != 7:
            raise AssertionError(f"invalid launcher chunk row: {row}")
        lane, split, count, before, stream, relative_root, label = fields
        return LauncherChunk(
            lane=int(lane),
            split=split,
            accepted_count=int(count),
            accepted_before=int(before),
            stream_id=int(stream),
            relative_root=Path(relative_root),
            label=label,
        )

    return tuple(map(parse, _shell_array_words(source, "CHUNKS")))


def _launcher_argv(
    template: Sequence[str],
    chunk: LauncherChunk,
    *,
    python: Path,
    output_root: Path,
) -> tuple[str, ...]:
    """Evaluate the launcher's command template for one chunk row."""

    substitutions = {
        "$PYTHON": str(python),
        "$SPLIT": chunk.split,
        "$COUNT": str(chunk.accepted_count),
        "$BEFORE": str(chunk.accepted_before),
        "$STREAM": str(chunk.stream_id),
        "$CHUNK_ROOT": str(output_root / chunk.relative_root),
    }
    return tuple(substitutions.get(word, word) for word in template) + ("--execute",)


def _standard_audit_chunks(
    chunks: Sequence[object],
) -> tuple[ChunkContract, ...]:
    """Normalize auditors whose expected chunks expose ``summary_path``."""

    return tuple(
        ChunkContract(
            split=getattr(chunk, "split").value,
            accepted_before=getattr(chunk, "accepted_before"),
            accepted_count=getattr(chunk, "accepted_count"),
            stream_id=getattr(chunk, "stream_id"),
            relative_summary=getattr(chunk, "summary_path"),
        )
        for chunk in chunks
    )


def _stokes_audit_chunks() -> tuple[ChunkContract, ...]:
    return tuple(
        ChunkContract(
            split=chunk.split.value,
            accepted_before=chunk.accepted_before,
            accepted_count=chunk.accepted_count,
            stream_id=chunk.stream_id,
            relative_summary=chunk.relative_summary,
        )
        for chunk in stokes_audit.EXPECTED_CHUNKS
    )


AUDIT_CHUNKS: Final = {
    "stokes": _stokes_audit_chunks(),
    "tanaka": _standard_audit_chunks(tanaka_audit.EXPECTED_CHUNKS),
    "benjamin_feir": _standard_audit_chunks(bf_audit.EXPECTED_CHUNKS),
    "jonswap_tma": _standard_audit_chunks(jonswap_audit.EXPECTED_CHUNKS),
}
AUDIT_ROOTS: Final = {
    "stokes": stokes_audit.DEFAULT_CORPUS_ROOT.resolve(),
    "tanaka": tanaka_audit.DEFAULT_CORPUS_ROOT.resolve(),
    "benjamin_feir": bf_audit.DEFAULT_CORPUS_ROOT.resolve(),
    "jonswap_tma": jonswap_audit.DEFAULT_CORPUS_ROOT.resolve(),
}
AUDIT_REVISIONS: Final = {
    "stokes": stokes_audit.EXPECTED_REVISION_ID,
    "tanaka": tanaka_audit.EXPECTED_REVISION_ID,
    "benjamin_feir": bf_audit.EXPECTED_REVISION_ID,
    "jonswap_tma": jonswap_audit.EXPECTED_REVISION_ID,
}


class PaperCorpusReleaseLayoutContractTest(unittest.TestCase):
    def test_release_shell_entrypoints_keep_executable_mode(self) -> None:
        modes = {
            path.relative_to(ROOT).as_posix(): stat.S_IMODE(path.stat().st_mode)
            for path in EXECUTABLE_ENTRYPOINTS
        }
        self.assertEqual(modes, {path: 0o755 for path in modes})

    def test_launchers_watchdog_and_auditors_share_current_layouts(self) -> None:
        python = ROOT / ".venv/bin/python"
        jonswap_source = JONSWAP_LAUNCHER.read_text(encoding="utf-8")
        tanaka_source = TANAKA_LAUNCHER.read_text(encoding="utf-8")
        launcher_sources = {
            "jonswap_tma": jonswap_source,
            "tanaka": tanaka_source,
        }
        expected_runners = {
            "jonswap_tma": "scripts/run_paper_corpus_jonswap_bucketed.py",
            "tanaka": "scripts/run_paper_corpus_quota.py",
        }
        launcher_chunks = {
            family: _launcher_chunks(source)
            for family, source in launcher_sources.items()
        }

        self.assertEqual(
            _shell_root_default(jonswap_source, "OUTPUT_BASE"),
            AUDIT_ROOTS["jonswap_tma"],
        )
        self.assertEqual(
            _shell_root_default(tanaka_source, "OUTPUT_BASE"),
            AUDIT_ROOTS["tanaka"],
        )
        self.assertEqual(
            watchdog.DEFAULT_JONSWAP_ROOT.resolve(), AUDIT_ROOTS["jonswap_tma"]
        )
        self.assertEqual(watchdog.DEFAULT_TANAKA_ROOT.resolve(), AUDIT_ROOTS["tanaka"])

        # The first JONSWAP chunk was already immutable before the two-lane
        # launcher was created.  Seven rows resume it; eight sources release it.
        jonswap_audited = {chunk.layout for chunk in AUDIT_CHUNKS["jonswap_tma"]}
        jonswap_launched = {chunk.layout for chunk in launcher_chunks["jonswap_tma"]}
        self.assertEqual(
            jonswap_audited - jonswap_launched,
            {("train", 0, 2_048, 0)},
        )
        self.assertEqual(len(jonswap_audited), 8)
        self.assertEqual(len(jonswap_launched), 7)
        self.assertEqual(
            {chunk.layout for chunk in launcher_chunks["tanaka"]},
            {chunk.layout for chunk in AUDIT_CHUNKS["tanaka"]},
        )
        self.assertEqual(len(launcher_chunks["tanaka"]), 6)

        plans = watchdog.build_worker_plans(
            checkout_root=ROOT,
            python=python,
            jonswap_root=AUDIT_ROOTS["jonswap_tma"],
            tanaka_root=AUDIT_ROOTS["tanaka"],
        )
        plans_by_family = {
            family: tuple(plan for plan in plans if plan.family == family)
            for family in launcher_sources
        }
        for family, source in launcher_sources.items():
            chunks = launcher_chunks[family]
            plans_by_layout = {
                (
                    plan.split,
                    int(plan.argv[plan.argv.index("--accepted-cases-before") + 1]),
                    int(plan.argv[plan.argv.index("--accepted-cases") + 1]),
                    int(plan.argv[plan.argv.index("--stream-id") + 1]),
                ): plan
                for plan in plans_by_family[family]
            }
            self.assertEqual(set(plans_by_layout), {chunk.layout for chunk in chunks})

            command_template = _shell_array_words(source, "COMMAND")
            self.assertEqual(command_template[1], expected_runners[family])
            self.assertEqual(
                command_template[command_template.index("--family") + 1], family
            )
            for chunk in chunks:
                plan = plans_by_layout[chunk.layout]
                self.assertEqual(plan.lane, chunk.lane)
                self.assertEqual(plan.gpu_index, str(chunk.lane))
                self.assertEqual(
                    plan.output_root,
                    AUDIT_ROOTS[family] / chunk.relative_root,
                )
                self.assertEqual(
                    plan.argv,
                    _launcher_argv(
                        command_template,
                        chunk,
                        python=python,
                        output_root=AUDIT_ROOTS[family],
                    ),
                )

    def test_builder_auditors_and_postcompletion_share_release_contract(self) -> None:
        source = VIEW_BUILDER.read_text(encoding="utf-8")
        root_variables = {
            "stokes": "STOKES_ROOT",
            "tanaka": "TANAKA_ROOT",
            "benjamin_feir": "BF_ROOT",
            "jonswap_tma": "JONSWAP_ROOT",
        }
        train_variables = {
            "stokes": "STOKES_TRAIN",
            "tanaka": "TANAKA_TRAIN",
            "benjamin_feir": "BF_TRAIN",
            "jonswap_tma": "JONSWAP_TRAIN",
        }
        fixed = _shell_array_words(source, "FIXED_SPLITS")

        builder_source_counts: dict[str, int] = {}
        for family in postcompletion.FAMILY_ORDER:
            root_variable = root_variables[family]
            self.assertEqual(
                _shell_root_default(source, root_variable),
                AUDIT_ROOTS[family],
            )
            prefix = f"${root_variable}/"
            paths = _shell_array_words(source, train_variables[family]) + tuple(
                path for path in fixed if path.startswith(prefix)
            )
            expected = {
                prefix + chunk.relative_summary.as_posix()
                for chunk in AUDIT_CHUNKS[family]
            }
            self.assertEqual(set(paths), expected)
            self.assertEqual(len(paths), len(expected))
            builder_source_counts[family] = len(paths)

        expected_source_counts = {
            "stokes": 6,
            "tanaka": 6,
            "benjamin_feir": 6,
            "jonswap_tma": 8,
        }
        self.assertEqual(builder_source_counts, expected_source_counts)
        self.assertEqual(postcompletion.SOURCE_COUNT_BY_FAMILY, expected_source_counts)
        self.assertEqual(
            dict(handoff.FINAL_CONTRACT.source_count_by_family),
            expected_source_counts,
        )
        self.assertEqual(postcompletion.FINAL_SOURCE_COUNT, 26)
        self.assertEqual(handoff.FINAL_CONTRACT.source_count, 26)

        checkpoint_calls = tuple(
            tuple(map(int, shlex.split(match.group("arguments"))))
            for match in re.finditer(
                r"(?m)^build_checkpoint (?P<arguments>[0-9 ]+)$",
                source,
            )
        )
        self.assertEqual(
            checkpoint_calls,
            ((2_048, 1, 1), (4_096, 2, 2), (8_192, 3, 3), (16_384, 4, 6)),
        )

        expected_layouts = {
            family: tuple(chunk.layout for chunk in AUDIT_CHUNKS[family])
            for family in postcompletion.FAMILY_ORDER
        }
        for family, expected in expected_layouts.items():
            self.assertEqual(postcompletion.expected_layout(family), expected)
            handoff_layout = tuple(
                (split, *chunk)
                for split in postcompletion.SPLIT_ORDER
                for chunk in handoff.FINAL_CONTRACT.chunk_layout_by_family_and_split[
                    split
                ][family]
            )
            self.assertEqual(handoff_layout, expected)

        production_revisions = {
            family: paper_corpus_revision_id(
                PhysicalFamilyId(handoff.FAMILY_IDS[family])
            )
            for family in postcompletion.FAMILY_ORDER
        }
        self.assertEqual(AUDIT_REVISIONS, production_revisions)
        self.assertEqual(postcompletion.REVISION_BY_FAMILY, production_revisions)
        self.assertEqual(
            dict(handoff.FINAL_CONTRACT.revision_by_family),
            production_revisions,
        )

        args = postcompletion.parse_args([])
        audit_defaults = {
            "stokes": args.stokes_binding.parent.resolve(),
            "tanaka": args.tanaka_audit.parent.resolve(),
            "benjamin_feir": args.bf_audit.parent.resolve(),
            "jonswap_tma": args.jonswap_audit.parent.resolve(),
        }
        self.assertEqual(audit_defaults, AUDIT_ROOTS)
        output_base = _shell_root_default(source, "OUTPUT_BASE")
        expected_final_summary = (
            output_base
            / "combined/c16384_v01024_t01024"
            / "paper_corpus_all_splits_c16384.summary.json"
        )
        self.assertEqual(args.combined_summary.resolve(), expected_final_summary)


if __name__ == "__main__":
    unittest.main()
