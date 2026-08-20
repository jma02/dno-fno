from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import run_paper_corpus_postcompletion as runner
from scripts import audit_completed_benjamin_feir_revision4 as bf_audit
from solver.gen_data.pipeline.archive import BatchPaths
from solver.gen_data.pipeline.production import SplitId


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _training_implementation_record(
    repository_root: Path = runner.ROOT,
) -> dict[str, object]:
    files = [
        {
            "path": relative,
            "role": role,
            "bytes": (repository_root / relative).stat().st_size,
            "sha256": runner._sha256(repository_root / relative),
        }
        for relative, role in runner.TRAINING_IMPLEMENTATION_FILES
    ]
    payload: dict[str, object] = {
        "schema": runner.TRAINING_IMPLEMENTATION_SCHEMA,
        "files": files,
        "semantic_relationship": dict(runner.TRAINING_IMPLEMENTATION_RELATIONSHIP),
    }
    return {**payload, "fingerprint": runner._canonical_sha256(payload)}


def _synthetic_generation_bindings(
    artifacts: dict[str, runner.Artifact],
) -> tuple[runner.DependencyBinding, dict[str, runner.GenerationSourceBinding]]:
    dependency_files = {
        "pyproject.toml": artifacts["stokes"],
        "uv.lock": artifacts["stokes_legacy"],
    }
    environment = {
        "python": {"implementation": "synthetic", "version": "0"},
        "packages": {"jax": "0", "jaxlib": "0", "numpy": "0"},
        "files_sha256": {
            name: artifact.sha256 for name, artifact in dependency_files.items()
        },
    }
    dependency = runner.DependencyBinding(
        environment=environment,
        fingerprint=runner._canonical_sha256(environment),
        files=dependency_files,
        chunk_count=0,
    )
    generation_sources = {
        family: runner.GenerationSourceBinding(
            source_sha256={f"synthetic/{family}.json": artifacts[family].sha256},
            fingerprint=runner._canonical_sha256(
                {f"synthetic/{family}.json": artifacts[family].sha256}
            ),
            current_sources={f"synthetic/{family}.json": artifacts[family]},
            historical_source_paths=(),
            chunk_count=0,
        )
        for family in runner.FAMILY_ORDER
    }
    return dependency, generation_sources


def _bf_source_evidence(
    *, snapshot_root: Path = runner.BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    historical_digests = {
        path: str(spec["sha256"])
        for path, spec in runner.BF_HISTORICAL_SOURCE_SNAPSHOTS.items()
    }
    source_mapping = {
        path: historical_digests.get(path, runner._sha256(runner.ROOT / path))
        for path in runner.BF_EXPECTED_GENERATION_SOURCE_PATHS
    }
    resolved_snapshot_root = snapshot_root.resolve()
    historical_sources = {
        repository_path: {
            "snapshot_path": str(resolved_snapshot_root / str(spec["name"])),
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
        }
        for repository_path, spec in runner.BF_HISTORICAL_SOURCE_SNAPSHOTS.items()
    }
    current_sources = {
        repository_path: {
            "path": str((runner.ROOT / repository_path).resolve()),
            "bytes": (runner.ROOT / repository_path).stat().st_size,
            "sha256": digest,
        }
        for repository_path, digest in source_mapping.items()
        if repository_path not in historical_digests
    }
    identity = {
        "source_sha256_fingerprint": runner._canonical_sha256(source_mapping),
        "historical_shared_source_snapshot_binding": {
            "schema": "paper_corpus_bf_revision4_historical_source_binding_v1",
            "role": "inert_historical_byte_recovery_only",
            "snapshot_root": str(resolved_snapshot_root),
            "sha256sums": {
                "path": str(resolved_snapshot_root / "SHA256SUMS"),
                "bytes": runner.BF_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["bytes"],
                "sha256": runner.BF_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["sha256"],
            },
            "sources": historical_sources,
            "chunk_source_maps_checked": 6,
        },
        "nonhistorical_generation_source_binding": {
            "schema": "paper_corpus_bf_revision4_current_source_binding_v1",
            "role": (
                "current_repository_bytes_for_all_nonhistorical_generation_sources"
            ),
            "repository_root": str(runner.ROOT.resolve()),
            "source_map_fingerprint": runner._canonical_sha256(source_mapping),
            "source_count": 19,
            "current_source_count": 17,
            "historical_snapshot_source_paths": sorted(historical_digests),
            "sources": current_sources,
            "chunk_source_maps_checked": 6,
        },
    }
    source_records = tuple(
        {"run_spec": {"configuration": {"source_sha256": dict(source_mapping)}}}
        for _ in range(6)
    )
    return {"identity": identity}, source_records


def _stokes_source_evidence(
    *,
    snapshot_root: Path = runner.STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
    repository_root: Path = runner.ROOT,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    historical_digests = {
        path: str(spec["sha256"])
        for path, spec in runner.STOKES_HISTORICAL_SOURCE_SNAPSHOTS.items()
    }
    source_mapping = {
        path: historical_digests[path]
        if path in historical_digests
        else runner._sha256(repository_root / path)
        for path in runner.STOKES_EXPECTED_GENERATION_SOURCE_PATHS
    }
    fingerprints = [f"{index + 1:064x}" for index in range(6)]
    resolved_snapshot_root = snapshot_root.resolve()
    historical_sources = {
        repository_path: {
            "snapshot_path": str(resolved_snapshot_root / str(spec["name"])),
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
        }
        for repository_path, spec in (runner.STOKES_HISTORICAL_SOURCE_SNAPSHOTS.items())
    }
    current_sources = {
        repository_path: {
            "path": str((repository_root / repository_path).resolve()),
            "bytes": (repository_root / repository_path).stat().st_size,
            "sha256": digest,
        }
        for repository_path, digest in source_mapping.items()
        if repository_path not in historical_digests
    }
    identity = {
        "dependency_fingerprint": "4" * 64,
        "execution_fingerprint": "5" * 64,
        "source_fingerprint": runner._canonical_sha256(source_mapping),
        "execution_platform": "cpu",
        "configuration_fingerprints": fingerprints,
        "source_sha256": source_mapping,
        "historical_generation_source_snapshot_binding": {
            "schema": "paper_corpus_stokes_revision2_historical_source_binding_v1",
            "role": "inert_historical_byte_recovery_only",
            "source_commit": runner.STOKES_HISTORICAL_SOURCE_COMMIT,
            "snapshot_root": str(resolved_snapshot_root),
            "source_map_fingerprint": runner._canonical_sha256(source_mapping),
            "source_count": 14,
            "historical_source_count": 3,
            "sha256sums": {
                "path": str(resolved_snapshot_root / "SHA256SUMS"),
                "bytes": runner.STOKES_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["bytes"],
                "sha256": runner.STOKES_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["sha256"],
            },
            "sources": historical_sources,
            "chunk_source_maps_checked": 6,
        },
        "nonhistorical_generation_source_binding": {
            "schema": "paper_corpus_stokes_revision2_current_source_binding_v1",
            "role": (
                "current_repository_bytes_for_all_nonhistorical_generation_sources"
            ),
            "repository_root": str(repository_root.resolve()),
            "source_map_fingerprint": runner._canonical_sha256(source_mapping),
            "source_count": 14,
            "current_source_count": 11,
            "historical_snapshot_source_paths": sorted(historical_digests),
            "sources": current_sources,
            "chunk_source_maps_checked": 6,
        },
    }
    source_records = tuple(
        {
            "configuration_fingerprint": fingerprint,
            "run_spec": {"configuration": {"source_sha256": dict(source_mapping)}},
        }
        for fingerprint in fingerprints
    )
    return {"identity": identity}, source_records


class PostcompletionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.combined = self.inputs / "paper_corpus_all_splits_c16384.summary.json"
        _write_json(
            self.combined,
            {"schema": runner.COMBINED_SUMMARY_SCHEMA, "status": "complete"},
        )
        self.manifest = self.inputs / "combined.dataset.json"
        self.trajectory_map = self.inputs / "combined.trajectory_map.npz"
        _write_json(self.manifest, {"schema_version": 2})
        self.trajectory_map.write_bytes(b"synthetic-map")
        audit_names = (
            "stokes_binding",
            "stokes_legacy",
            "benjamin_feir",
            "jonswap_tma",
            "tanaka",
        )
        self.audit_files = {name: self.inputs / f"{name}.json" for name in audit_names}
        for name, path in self.audit_files.items():
            _write_json(path, {"schema": f"synthetic_{name}", "status": "pass"})
        self.audit_paths = runner.AuditPaths(
            stokes_binding=self.audit_files["stokes_binding"],
            stokes_legacy=self.audit_files["stokes_legacy"],
            benjamin_feir=self.audit_files["benjamin_feir"],
            jonswap_tma=self.audit_files["jonswap_tma"],
            tanaka=self.audit_files["tanaka"],
        )
        self.state_root = self.root / "state"
        self.release_snapshot_root = self.root / "bf_release_snapshots"
        self.release_snapshot_root.mkdir()
        for name in ("SHA256SUMS", "production.py", "trajectory_family_adapters.py"):
            shutil.copy2(
                runner.BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT / name,
                self.release_snapshot_root / name,
            )
        self.release_stokes_snapshot_root = self.root / "stokes_release_snapshots"
        self.release_stokes_snapshot_root.mkdir()
        for name in (
            "SHA256SUMS",
            "run_paper_corpus_quota.py",
            "manifest.py",
            "production.py",
        ):
            shutil.copy2(
                runner.STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT / name,
                self.release_stokes_snapshot_root / name,
            )
        self.worker = self.root / "stage_worker.py"
        self.worker.write_text(
            """\
import json
from pathlib import Path
import sys
import time

output = Path(sys.argv[1])
stage = sys.argv[2]
counter = Path(sys.argv[3])
behavior = sys.argv[4]
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if behavior == "fail_first" and count == 1:
    raise SystemExit(7)
if behavior == "timeout":
    time.sleep(5.0)
order = Path(sys.argv[5])
with order.open("a", encoding="utf-8") as handle:
    handle.write(stage + "\\n")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"status": "complete", "stage": stage}) + "\\n")
""",
            encoding="utf-8",
        )

        self.multi_output_worker = self.root / "multi_output_stage_worker.py"
        self.multi_output_worker.write_text(
            """\
import json
from pathlib import Path
import sys

output_root = Path(sys.argv[1])
stage = sys.argv[2]
counter = Path(sys.argv[3])
behavior = sys.argv[4]
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
pdf = output_root / f"{stage}.pdf"
png = output_root / f"{stage}.png"
sidecar = output_root / f"{stage}.json"
pdf.write_bytes(b"complete-pdf")
if behavior == "partial_first" and count == 1:
    raise SystemExit(7)
png.write_bytes(b"complete-png")
sidecar.write_text(
    json.dumps({"status": "complete", "stage": stage}) + "\\n",
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        self.order = self.root / "order.txt"

    def _strict_stage_fixture(self):
        from scripts.test_build_paper_corpus_document_handoff import (
            DocumentHandoffTests,
        )

        fixture = DocumentHandoffTests(
            methodName="test_success_and_idempotent_identical_output"
        )
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _release(self, *, alternate_bf: bool = False) -> runner.ReleaseBinding:
        audits = {
            "stokes": runner._artifact(self.audit_files["stokes_binding"]),
            "stokes_legacy": runner._artifact(self.audit_files["stokes_legacy"]),
            "benjamin_feir": runner._artifact(self.audit_files["benjamin_feir"]),
            "jonswap_tma": runner._artifact(self.audit_files["jonswap_tma"]),
            "tanaka": runner._artifact(self.audit_files["tanaka"]),
        }
        if alternate_bf:
            alternate = self.inputs / "alternate_bf.json"
            _write_json(
                alternate,
                {"schema": "synthetic_benjamin_feir", "status": "pass"},
            )
            audits["benjamin_feir"] = runner._artifact(alternate)
        dependency, generation_sources = _synthetic_generation_bindings(audits)
        return runner.ReleaseBinding(
            combined=runner.CombinedBinding(
                summary=runner._artifact(self.combined),
                manifest=runner._artifact(self.manifest),
                trajectory_map=runner._artifact(self.trajectory_map),
                chunks=(),
            ),
            audits=audits,
            dependency=dependency,
            generation_sources=generation_sources,
            historical_source_snapshots={
                "stokes": {
                    "SHA256SUMS": runner._artifact(
                        self.release_stokes_snapshot_root / "SHA256SUMS"
                    ),
                    "scripts/run_paper_corpus_quota.py": runner._artifact(
                        self.release_stokes_snapshot_root / "run_paper_corpus_quota.py"
                    ),
                    "solver/gen_data/pipeline/manifest.py": runner._artifact(
                        self.release_stokes_snapshot_root / "manifest.py"
                    ),
                    "solver/gen_data/pipeline/production.py": runner._artifact(
                        self.release_stokes_snapshot_root / "production.py"
                    ),
                },
                "benjamin_feir": {
                    "SHA256SUMS": runner._artifact(
                        self.release_snapshot_root / "SHA256SUMS"
                    ),
                    "solver/gen_data/pipeline/production.py": runner._artifact(
                        self.release_snapshot_root / "production.py"
                    ),
                    "solver/gen_data/trajectory_family_adapters.py": runner._artifact(
                        self.release_snapshot_root / "trajectory_family_adapters.py"
                    ),
                },
            },
        )

    def _config(self, **overrides: object) -> runner.RunnerConfig:
        values: dict[str, object] = {
            "combined_summary": self.combined,
            "audits": self.audit_paths,
            "state_root": self.state_root,
            "python": Path(sys.executable),
            "poll_seconds": 0.0,
            "wait_timeout_seconds": 1.0,
            "stage_attempts": 1,
            "retry_delay_seconds": 0.0,
            "kill_after_seconds": 0.05,
        }
        values.update(overrides)
        return runner.RunnerConfig(**values)  # type: ignore[arg-type]

    def _stage(
        self,
        name: str,
        output_root: Path,
        *,
        behavior: str = "success",
        requires: tuple[str, ...] = (),
        timeout: float = 2.0,
    ) -> runner.StageSpec:
        output = output_root / f"{name}.json"
        counter = self.root / f"{name}.count"

        def validate() -> dict[str, object]:
            record = runner._strict_json(output)
            if record != {"status": "complete", "stage": name}:
                raise RuntimeError("wrong synthetic stage output")
            return {"output": runner._artifact(output).record()}

        return runner.StageSpec(
            name=name,
            command=(
                sys.executable,
                str(self.worker),
                str(output),
                name,
                str(counter),
                behavior,
                str(self.order),
            ),
            timeout_seconds=timeout,
            outputs=(output,),
            validate=validate,
            requires=requires,
        )

    def _four_stages(
        self,
        _config: runner.RunnerConfig,
        _release: runner.ReleaseBinding,
        output_root: Path,
    ) -> tuple[runner.StageSpec, ...]:
        return tuple(
            self._stage(name, output_root)
            for name in ("renderer", "order", "training", "figure")
        )

    def _producer_owned_multi_output_stage(
        self,
        name: str,
        output_root: Path,
        *,
        behavior: str = "success",
    ) -> runner.StageSpec:
        pdf = output_root / f"{name}.pdf"
        png = output_root / f"{name}.png"
        sidecar = output_root / f"{name}.json"
        counter = self.root / f"{name}.count"

        def validate() -> dict[str, object]:
            record = runner._strict_json(sidecar)
            if record != {"status": "complete", "stage": name}:
                raise RuntimeError("wrong synthetic multi-output stage output")
            return {
                "outputs": [
                    runner._artifact(path).record() for path in (pdf, png, sidecar)
                ]
            }

        return runner.StageSpec(
            name=name,
            command=(
                sys.executable,
                str(self.multi_output_worker),
                str(output_root),
                name,
                str(counter),
                behavior,
            ),
            timeout_seconds=2.0,
            outputs=(pdf, png, sidecar),
            validate=validate,
            completion_marker=sidecar,
            producer_owns_uncommitted_outputs=True,
        )

    def test_wait_polls_at_most_sixty_seconds(self) -> None:
        missing = self.root / "arrives_later"
        sleeps: list[float] = []

        def sleeper(delay: float) -> None:
            sleeps.append(delay)
            missing.write_text("ready", encoding="utf-8")

        runner.wait_for_inputs(
            (missing,),
            poll_seconds=60.0,
            timeout_seconds=None,
            sleeper=sleeper,
        )
        self.assertEqual(sleeps, [60.0])
        with self.assertRaisesRegex(ValueError, "between 0 and 60"):
            runner.wait_for_inputs(
                (self.root / "never",),
                poll_seconds=61.0,
                timeout_seconds=1.0,
            )

    def test_fast_path_lock_contention_creates_no_journal(self) -> None:
        release = self._release()
        authentication_calls = 0

        def authenticate(
            _path: Path,
            _audits: runner.AuditPaths,
        ) -> runner.ReleaseBinding:
            nonlocal authentication_calls
            authentication_calls += 1
            return release

        with (
            patch.object(runner, "_lock_release_root", return_value=None),
            patch.object(
                runner,
                "RunJournal",
                side_effect=AssertionError("fast lock failure created a journal"),
            ),
        ):
            self.assertEqual(
                runner.run_postcompletion(
                    self._config(),
                    authenticate=authenticate,
                    stage_factory=lambda _config, _release, _output: (),
                ),
                1,
            )

        self.assertEqual(authentication_calls, 1)
        self.assertFalse((self.state_root / "status.json").exists())

    def test_post_wait_lock_contention_remains_durably_journaled(self) -> None:
        release = self._release()
        delayed = self.audit_files["tanaka"]
        delayed_bytes = delayed.read_bytes()
        delayed.unlink()
        sleeps: list[float] = []

        def sleeper(delay: float) -> None:
            sleeps.append(delay)
            delayed.write_bytes(delayed_bytes)

        with patch.object(runner, "_lock_release_root", return_value=None):
            self.assertEqual(
                runner.run_postcompletion(
                    self._config(),
                    authenticate=lambda _path, _audits: release,
                    stage_factory=lambda _config, _release, _output: (),
                    sleeper=sleeper,
                ),
                1,
            )

        self.assertEqual(sleeps, [0.0])
        status = runner._strict_json(self.state_root / "status.json")
        self.assertEqual(status["status"], "lock_failed")
        self.assertEqual(status["error"], "another runner owns this release SHA")
        self.assertNotIn("missing_inputs", status)

    def test_order_and_idempotent_restart_skip_valid_outputs(self) -> None:
        release = self._release()

        def authenticate(
            _path: Path, _audits: runner.AuditPaths
        ) -> runner.ReleaseBinding:
            return release

        config = self._config()
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=authenticate,
                stage_factory=self._four_stages,
            ),
            0,
        )
        output_root = self.state_root / release.combined.summary.sha256
        output_hashes = {
            name: runner._sha256(output_root / f"{name}.json")
            for name in ("renderer", "order", "training", "figure")
        }
        self.assertEqual(
            self.order.read_text(encoding="utf-8").splitlines(),
            ["renderer", "order", "training", "figure"],
        )
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=authenticate,
                stage_factory=self._four_stages,
            ),
            0,
        )
        self.assertEqual(
            self.order.read_text(encoding="utf-8").splitlines(),
            ["renderer", "order", "training", "figure"],
        )
        self.assertEqual(
            output_hashes,
            {
                name: runner._sha256(output_root / f"{name}.json")
                for name in output_hashes
            },
        )
        child_status = runner._strict_json(output_root / "status.json")
        top_status = runner._strict_json(self.state_root / "status.json")
        self.assertEqual(child_status["status"], "complete")
        self.assertEqual(top_status["status"], "complete")
        self.assertEqual(
            Path(top_status["child_status_path"]), output_root / "status.json"
        )
        self.assertTrue(child_status["release_evidence_unchanged_after_stages"])
        self.assertTrue(
            child_status["release_evidence"]["source_cross_binding"]["verified"]
        )

    def test_terminal_document_handoff_prevents_any_rerun_mutation(self) -> None:
        from scripts import build_paper_corpus_document_handoff as handoff

        fixture = self._strict_stage_fixture()
        top_status = fixture.base / "status.json"
        _write_json(
            top_status,
            {
                "schema": runner.RUNNER_SCHEMA,
                "status": "complete",
                "child_status_path": str(fixture.status_path),
            },
        )
        terminal_handoff = handoff.build_document_handoff(
            fixture.status_path,
            release_authenticator=fixture._authenticate,
        )
        self.assertEqual(
            terminal_handoff,
            fixture.root / runner.DOCUMENT_HANDOFF_NAME,
        )
        audits = runner.AuditPaths(
            stokes_binding=fixture.release.audits["stokes"].path,
            stokes_legacy=fixture.release.audits["stokes_legacy"].path,
            benjamin_feir=fixture.release.audits["benjamin_feir"].path,
            jonswap_tma=fixture.release.audits["jonswap_tma"].path,
            tanaka=fixture.release.audits["tanaka"].path,
        )
        config = runner.RunnerConfig(
            combined_summary=fixture.summary,
            audits=audits,
            state_root=fixture.base,
            python=Path(sys.executable),
            poll_seconds=0.0,
            wait_timeout_seconds=1.0,
            stage_attempts=1,
            retry_delay_seconds=0.0,
            kill_after_seconds=0.05,
        )
        before_files = {
            path.relative_to(fixture.base): path.read_bytes()
            for path in fixture.base.rglob("*")
            if path.is_file()
        }
        before_directories = {
            path.relative_to(fixture.base)
            for path in fixture.base.rglob("*")
            if path.is_dir()
        }
        authentication_calls = 0

        def authenticate(
            combined: Path,
            audit_paths: runner.AuditPaths,
        ) -> runner.ReleaseBinding:
            nonlocal authentication_calls
            authentication_calls += 1
            return fixture._authenticate(combined, audit_paths)

        original_terminal_check = runner._terminal_document_handoff_exists

        def terminal_check(output_root: Path) -> bool:
            with (output_root / "runner.lock").open("a+b") as contender:
                with self.assertRaises(BlockingIOError):
                    runner.fcntl.flock(
                        contender.fileno(),
                        runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB,
                    )
            return original_terminal_check(output_root)

        with (
            patch.object(
                runner,
                "RunJournal",
                side_effect=AssertionError("terminal rerun created a RunJournal"),
            ),
            patch.object(
                runner,
                "_terminal_document_handoff_exists",
                side_effect=terminal_check,
            ),
        ):
            self.assertEqual(
                runner.run_postcompletion(
                    config,
                    authenticate=authenticate,
                    stage_factory=lambda _config, _release, _output: (),
                ),
                1,
            )

        self.assertEqual(authentication_calls, 1)
        self.assertEqual(
            {
                path.relative_to(fixture.base): path.read_bytes()
                for path in fixture.base.rglob("*")
                if path.is_file()
            },
            before_files,
        )
        self.assertEqual(
            {
                path.relative_to(fixture.base)
                for path in fixture.base.rglob("*")
                if path.is_dir()
            },
            before_directories,
        )

    def test_training_stage_is_revalidated_after_later_stages(self) -> None:
        release = self._release()
        code = self.root / "bound_training_code.py"
        code.write_text("# original\n", encoding="utf-8")
        expected_sha256 = runner._sha256(code)

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            base_training = self._stage("training_handoff_audit", output_root)

            def validate_training() -> dict[str, object]:
                validated = dict(base_training.validate())
                if runner._sha256(code) != expected_sha256:
                    raise RuntimeError("bound training code changed")
                return validated

            training = runner.StageSpec(
                name=base_training.name,
                command=base_training.command,
                timeout_seconds=base_training.timeout_seconds,
                outputs=base_training.outputs,
                validate=validate_training,
            )
            figure_output = output_root / "family_case_figure.json"

            def validate_figure() -> dict[str, object]:
                return {"output": runner._artifact(figure_output).record()}

            figure = runner.StageSpec(
                name="family_case_figure",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path\n"
                        "import sys\n"
                        "Path(sys.argv[1]).write_text('complete\\n')\n"
                        "Path(sys.argv[2]).write_text('# changed\\n')\n"
                    ),
                    str(figure_output),
                    str(code),
                ),
                timeout_seconds=2.0,
                outputs=(figure_output,),
                validate=validate_figure,
            )
            return training, figure

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        training = status["stages"]["training_handoff_audit"]
        self.assertEqual(training["status"], "failed_final_validation")
        self.assertIn("bound training code changed", training["error"])

    def test_renderer_artifact_mutated_after_validation_fails_final_gate(self) -> None:
        release = self._release()

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            base_renderer = self._stage("final_renderer", output_root)
            renderer_output = base_renderer.outputs[0]

            def validate_renderer() -> dict[str, object]:
                record = runner._strict_json(renderer_output)
                if (
                    record.get("status") != "complete"
                    or record.get("stage") != "final_renderer"
                ):
                    raise RuntimeError("wrong synthetic renderer output")
                return {"output": runner._artifact(renderer_output).record()}

            renderer = runner.StageSpec(
                name=base_renderer.name,
                command=base_renderer.command,
                timeout_seconds=base_renderer.timeout_seconds,
                outputs=base_renderer.outputs,
                validate=validate_renderer,
            )
            later_output = output_root / "later_stage.json"
            later = runner.StageSpec(
                name="later_stage",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path\n"
                        "import sys\n"
                        "Path(sys.argv[1]).write_text('later complete\\n')\n"
                        "Path(sys.argv[2]).write_text("
                        '\'{"status":"complete","stage":"final_renderer",\''
                        "'\"mutated\":true}\\n')\n"
                    ),
                    str(later_output),
                    str(renderer_output),
                ),
                timeout_seconds=2.0,
                outputs=(later_output,),
                validate=lambda: {"output": runner._artifact(later_output).record()},
                requires=("final_renderer",),
            )
            return renderer, later

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        renderer = status["stages"]["final_renderer"]
        self.assertEqual(renderer["status"], "failed_final_validation")
        self.assertIn("artifact map changed", renderer["error"])

    def test_duplicate_stage_names_fail_before_execution(self) -> None:
        release = self._release()
        counter = self.root / "duplicate.count"

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            first = self._stage("duplicate", output_root)
            second = self._stage("duplicate", output_root)
            return first, second

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        self.assertFalse(counter.exists())
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertEqual(status["status"], "diagnostics_failed")
        self.assertEqual(status["diagnostic_failures"], ["stage_configuration"])
        self.assertIn(
            "must be unique", status["stages"]["stage_configuration"]["error"]
        )

    def test_training_revalidation_follows_deep_second_authentication(self) -> None:
        release = self._release()
        code = self.root / "deep_auth_bound_training_code.py"
        code.write_text("# original\n", encoding="utf-8")
        expected_sha256 = runner._sha256(code)

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            base = self._stage("training_handoff_audit", output_root)

            def validate_training() -> dict[str, object]:
                result = dict(base.validate())
                if runner._sha256(code) != expected_sha256:
                    raise RuntimeError("training code changed during deep auth")
                return result

            return (
                runner.StageSpec(
                    name=base.name,
                    command=base.command,
                    timeout_seconds=base.timeout_seconds,
                    outputs=base.outputs,
                    validate=validate_training,
                ),
            )

        calls = 0

        def authenticate(
            _path: Path,
            _audits: runner.AuditPaths,
        ) -> runner.ReleaseBinding:
            nonlocal calls
            calls += 1
            if calls == 2:
                code.write_text("# changed by deep second auth\n", encoding="utf-8")
            return release

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=authenticate,
                stage_factory=stages,
            ),
            1,
        )
        self.assertEqual(calls, 2)
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        training = status["stages"]["training_handoff_audit"]
        self.assertEqual(training["status"], "failed_final_validation")
        self.assertIn("changed during deep auth", training["error"])
        self.assertEqual(status["release_reauthentication"]["status"], "pass")

    def test_failed_attempt_retries_only_when_no_output_was_committed(self) -> None:
        release = self._release()
        config = self._config(stage_attempts=2)

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            return (self._stage("retry", output_root, behavior="fail_first"),)

        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            0,
        )
        self.assertEqual((self.root / "retry.count").read_text(), "2")
        status = runner._strict_json(
            self.state_root / release.combined.summary.sha256 / "status.json"
        )
        self.assertEqual(len(status["stages"]["retry"]["attempts"]), 2)

    def test_producer_owned_uncommitted_set_retries_after_partial_publish(
        self,
    ) -> None:
        release = self._release()

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            return (
                self._producer_owned_multi_output_stage(
                    "resumable", output_root, behavior="partial_first"
                ),
            )

        self.assertEqual(
            runner.run_postcompletion(
                self._config(stage_attempts=2),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            0,
        )
        self.assertEqual((self.root / "resumable.count").read_text(), "2")
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertEqual(status["stages"]["resumable"]["status"], "complete")
        self.assertEqual(len(status["stages"]["resumable"]["attempts"]), 2)

    def test_restart_resumes_producer_owned_set_without_completion_marker(
        self,
    ) -> None:
        release = self._release()

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            return (
                self._producer_owned_multi_output_stage(
                    "restartable", output_root, behavior="partial_first"
                ),
            )

        config = self._config(stage_attempts=1)
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        output_root = self.state_root / release.combined.summary.sha256
        self.assertTrue((output_root / "restartable.pdf").is_file())
        self.assertFalse((output_root / "restartable.json").exists())
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            0,
        )
        self.assertEqual((self.root / "restartable.count").read_text(), "2")
        self.assertTrue((output_root / "restartable.json").is_file())

    def test_marker_present_partial_set_and_symlink_still_fail_closed(self) -> None:
        output_root = self.root / "protected_multi"
        output_root.mkdir()
        spec = self._producer_owned_multi_output_stage("protected", output_root)
        marker = output_root / "protected.json"
        marker.write_text('{"status":"complete","stage":"protected"}\n')
        with self.assertRaisesRegex(runner.ExistingOutputError, "partial pre-existing"):
            runner._validate_existing(spec)

        (output_root / "protected.pdf").write_bytes(b"pdf")
        (output_root / "protected.png").write_bytes(b"png")
        marker.write_text('{"status":"wrong","stage":"protected"}\n')
        with self.assertRaisesRegex(runner.ExistingOutputError, "unauthenticated"):
            runner._validate_existing(spec)

        marker.unlink()
        (output_root / "protected.pdf").unlink()
        target = self.root / "target.pdf"
        target.write_bytes(b"target")
        (output_root / "protected.pdf").symlink_to(target)
        with self.assertRaisesRegex(runner.ExistingOutputError, "symbolic-link"):
            runner._validate_existing(spec)

    def test_unknown_partial_set_and_non_file_output_still_fail_closed(self) -> None:
        output_root = self.root / "unknown_partial"
        output_root.mkdir()
        unknown = runner.StageSpec(
            name="unknown",
            command=(sys.executable, "-c", "raise SystemExit(99)"),
            timeout_seconds=2.0,
            outputs=(output_root / "unknown.pdf", output_root / "unknown.json"),
            validate=lambda: {},
        )
        unknown.outputs[0].write_bytes(b"unknown")
        with self.assertRaisesRegex(runner.ExistingOutputError, "partial pre-existing"):
            runner._validate_existing(unknown)

        owned = self._producer_owned_multi_output_stage("owned", output_root)
        owned.outputs[0].mkdir()
        with self.assertRaisesRegex(runner.ExistingOutputError, "non-file"):
            runner._validate_existing(owned)

    def test_only_default_figure_stage_owns_markerless_payloads(self) -> None:
        release = self._release()
        output_root = self.root / "default_stage_outputs"
        stages = runner.default_stages(self._config(), release, output_root)

        self.assertEqual(
            [stage.name for stage in stages if stage.producer_owns_uncommitted_outputs],
            ["family_case_figure"],
        )
        figure = stages[-1]
        self.assertEqual(
            figure.completion_marker, output_root / "family_case_examples.json"
        )
        self.assertIn(figure.completion_marker, figure.outputs)

    def test_timeout_is_explicit_dependency_is_not_run_and_later_stage_continues(
        self,
    ) -> None:
        release = self._release()
        input_hashes = {
            path: runner._sha256(path)
            for path in (self.combined, *self.audit_paths.all())
        }

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            return (
                self._stage("renderer", output_root, behavior="timeout", timeout=0.05),
                self._stage("order", output_root, requires=("renderer",)),
                self._stage("training", output_root),
            )

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        output_root = self.state_root / release.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertEqual(status["status"], "diagnostics_failed")
        self.assertEqual(status["stages"]["renderer"]["status"], "failed")
        self.assertEqual(
            status["stages"]["order"]["status"], "not_run_dependency_failed"
        )
        self.assertEqual(status["stages"]["training"]["status"], "complete")
        self.assertFalse((self.root / "order.count").exists())
        self.assertEqual(
            input_hashes,
            {path: runner._sha256(path) for path in input_hashes},
        )

    def test_invalid_preexisting_output_is_never_overwritten(self) -> None:
        release = self._release()

        def stages(
            _config: runner.RunnerConfig,
            _release: runner.ReleaseBinding,
            output_root: Path,
        ) -> tuple[runner.StageSpec, ...]:
            output = output_root / "protected.json"
            output.write_bytes(b"do-not-overwrite")
            return (self._stage("protected", output_root),)

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=lambda _path, _audits: release,
                stage_factory=stages,
            ),
            1,
        )
        output = self.state_root / release.combined.summary.sha256 / "protected.json"
        self.assertEqual(output.read_bytes(), b"do-not-overwrite")
        self.assertFalse((self.root / "protected.count").exists())

    def test_existing_sha_root_rejects_different_audit_identity(self) -> None:
        release = self._release()
        config = self._config()
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=lambda _path, _audits: release,
                stage_factory=lambda _config, _release, _output: (),
            ),
            0,
        )
        identity_path = (
            self.state_root / release.combined.summary.sha256 / "release_identity.json"
        )
        original_hash = runner._sha256(identity_path)
        mismatched = self._release(alternate_bf=True)
        self.assertEqual(
            runner.run_postcompletion(
                config,
                authenticate=lambda _path, _audits: mismatched,
                stage_factory=lambda _config, _release, _output: (),
            ),
            1,
        )
        self.assertEqual(runner._sha256(identity_path), original_hash)
        top_status = runner._strict_json(self.state_root / "status.json")
        self.assertEqual(top_status["status"], "release_identity_failed")

    def test_cross_binding_rejects_summary_permutation_between_intervals(self) -> None:
        def binding(
            family: str,
            path: str,
            before: int,
            count: int,
            stream: int,
        ) -> dict[str, object]:
            return {
                "summary_path": path,
                "summary_sha256": ("1" if path.endswith("a") else "2") * 64,
                "family": family,
                "split": "train",
                "accepted_before": before,
                "accepted_count": count,
                "accepted_after": before + count,
                "attempted_count": count,
                "stream_id": stream,
                "configuration_fingerprint": ("3" if stream == 0 else "4") * 64,
            }

        stokes = (
            binding("stokes", "/source/a", 0, 2_048, 0),
            binding("stokes", "/source/b", 2_048, 2_048, 1),
        )
        other = {
            family: (binding(family, f"/source/{family}-a", 0, 1, 0),)
            for family in runner.FAMILY_ORDER
            if family != "stokes"
        }
        combined = (*stokes, *(records[0] for records in other.values()))
        audits: dict[str, tuple[dict[str, object], ...]] = {
            "stokes": stokes,
            **other,
        }
        runner._cross_bind_audit_sources(combined, audits)
        permuted = (dict(stokes[0]), dict(stokes[1]))
        for field in ("summary_path", "summary_sha256"):
            permuted[0][field], permuted[1][field] = (
                permuted[1][field],
                permuted[0][field],
            )
        with self.assertRaisesRegex(RuntimeError, "source order/identity differs"):
            runner._cross_bind_audit_sources(
                combined,
                {"stokes": permuted, **other},
            )

    def test_final_reauthentication_detects_release_mutation(self) -> None:
        initial = self._release()
        calls = 0

        def authenticate(
            _path: Path, _audits: runner.AuditPaths
        ) -> runner.ReleaseBinding:
            nonlocal calls
            calls += 1
            return initial if calls == 1 else self._release(alternate_bf=True)

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=authenticate,
                stage_factory=lambda _config, _release, _output: (),
            ),
            1,
        )
        output_root = self.state_root / initial.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertEqual(status["status"], "diagnostics_failed")
        self.assertFalse(status["release_evidence_unchanged_after_stages"])
        self.assertIn("release_reauthentication", status["diagnostic_failures"])

    def test_symlink_release_artifact_is_rejected_before_resolve(self) -> None:
        target = self.root / "target.json"
        _write_json(target, {"status": "pass"})
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            runner._artifact(link)

    def test_tanaka_audit_must_explicitly_exclude_historical_compatibility(
        self,
    ) -> None:
        checks = dict(runner.EXPECTED_AUDIT_CHECKS["tanaka"])
        del checks["historical_generation_compatibility_used"]
        with self.assertRaisesRegex(RuntimeError, "missing=.*historical"):
            runner._validate_audit_checks({"checks": checks}, family="tanaka")
        checks["historical_generation_compatibility_used"] = False
        runner._validate_audit_checks({"checks": checks}, family="tanaka")

    def test_tanaka_audit_requires_behavioral_amplitude_checks(self) -> None:
        for check in (
            "below_old_floor_amplitude_behavior_verified",
            "solver_amplitude_validator_called_exactly_once",
            "amplitude_validator_deliberate_mismatch_rejected",
        ):
            with self.subTest(check=check):
                checks = dict(runner.EXPECTED_AUDIT_CHECKS["tanaka"])
                runner._validate_audit_checks({"checks": checks}, family="tanaka")
                del checks[check]
                with self.assertRaisesRegex(RuntimeError, f"missing=.*{check}"):
                    runner._validate_audit_checks(
                        {"checks": checks},
                        family="tanaka",
                    )

    def test_jonswap_audit_requires_population_conditioning_contract(self) -> None:
        checks = dict(runner.EXPECTED_AUDIT_CHECKS["jonswap_tma"])
        runner._validate_audit_checks({"checks": checks}, family="jonswap_tma")
        del checks["population_conditioning_semantics_and_27_cell_counts_verified"]
        with self.assertRaisesRegex(RuntimeError, "missing=.*population_conditioning"):
            runner._validate_audit_checks({"checks": checks}, family="jonswap_tma")

    def test_bf_and_tanaka_audits_require_reconstructed_summary_cell_counts(
        self,
    ) -> None:
        check = "summary_cell_counts_match_reconstructed_transactions"
        for family in ("benjamin_feir", "tanaka"):
            with self.subTest(family=family):
                checks = dict(runner.EXPECTED_AUDIT_CHECKS[family])
                runner._validate_audit_checks({"checks": checks}, family=family)
                del checks[check]
                with self.assertRaisesRegex(RuntimeError, f"missing=.*{check}"):
                    runner._validate_audit_checks(
                        {"checks": checks},
                        family=family,
                    )

    def test_bf_audit_requires_historical_shared_source_snapshot_binding(
        self,
    ) -> None:
        self.assertEqual(
            bf_audit.EXPECTED_GENERATION_SOURCE_PATHS,
            runner.BF_EXPECTED_GENERATION_SOURCE_PATHS,
        )
        for check in (
            "historical_shared_source_snapshots_bound_to_all_chunks",
            "all_nonhistorical_generation_sources_match_current_bytes",
        ):
            with self.subTest(check=check):
                checks = dict(runner.EXPECTED_AUDIT_CHECKS["benjamin_feir"])
                runner._validate_audit_checks(
                    {"checks": checks}, family="benjamin_feir"
                )
                del checks[check]
                with self.assertRaisesRegex(RuntimeError, f"missing=.*{check}"):
                    runner._validate_audit_checks(
                        {"checks": checks},
                        family="benjamin_feir",
                    )

    def test_stokes_audit_requires_exact_source_closure_checks(self) -> None:
        for check in (
            "summary_cell_counts_match_completed_quota_state",
            "exact_equal_14_path_source_maps",
            "source_map_fingerprint_verified",
            "historical_generation_source_snapshots_bound_to_all_chunks",
            "all_nonhistorical_generation_sources_match_current_bytes",
        ):
            with self.subTest(check=check):
                checks = dict(runner.EXPECTED_AUDIT_CHECKS["stokes"])
                runner._validate_audit_checks({"checks": checks}, family="stokes")
                del checks[check]
                with self.assertRaisesRegex(RuntimeError, f"missing=.*{check}"):
                    runner._validate_audit_checks(
                        {"checks": checks},
                        family="stokes",
                    )

    def test_stokes_source_identity_is_strict_and_physically_rehashed(self) -> None:
        audit_record, source_records = _stokes_source_evidence()
        artifacts = runner._validate_stokes_source_identity(
            audit_record,
            source_records,
        )
        self.assertEqual(
            set(artifacts),
            {
                "SHA256SUMS",
                "scripts/run_paper_corpus_quota.py",
                "solver/gen_data/pipeline/manifest.py",
                "solver/gen_data/pipeline/production.py",
            },
        )

        missing_identity = deepcopy(audit_record)
        del missing_identity["identity"][
            "historical_generation_source_snapshot_binding"
        ]
        with self.assertRaisesRegex(RuntimeError, "Stokes identity fields differ"):
            runner._validate_stokes_source_identity(missing_identity, source_records)

        forged_fingerprint = deepcopy(audit_record)
        forged_fingerprint["identity"]["source_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source_fingerprint differs"):
            runner._validate_stokes_source_identity(
                forged_fingerprint,
                source_records,
            )

        mismatched_sixth = deepcopy(source_records)
        mismatched_sixth[-1]["run_spec"]["configuration"]["source_sha256"][
            "solver/gen_data/pipeline/production.py"
        ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exactly equal maps"):
            runner._validate_stokes_source_identity(
                audit_record,
                tuple(mismatched_sixth),
            )

        copied_snapshot_root = self.root / "stokes_checksum_mutation"
        shutil.copytree(
            runner.STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
            copied_snapshot_root,
        )
        copied_audit, copied_sources = _stokes_source_evidence(
            snapshot_root=copied_snapshot_root
        )
        checksum = copied_snapshot_root / "SHA256SUMS"
        checksum.write_bytes(checksum.read_bytes() + b"\n")
        with self.assertRaisesRegex(RuntimeError, "SHA256SUMS"):
            runner._validate_stokes_source_identity(
                copied_audit,
                copied_sources,
                snapshot_root=copied_snapshot_root,
            )

    def test_stokes_current_source_ancestor_symlink_is_rejected(self) -> None:
        repository = self.root / "stokes_repository"
        historical_paths = set(runner.STOKES_HISTORICAL_SOURCE_SNAPSHOTS)
        for relative in (
            runner.STOKES_EXPECTED_GENERATION_SOURCE_PATHS - historical_paths
        ):
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runner.ROOT / relative, destination)
        real_pipeline = self.root / "real_pipeline"
        (repository / "solver/gen_data/pipeline").rename(real_pipeline)
        (repository / "solver/gen_data/pipeline").symlink_to(
            real_pipeline,
            target_is_directory=True,
        )
        audit_record, source_records = _stokes_source_evidence(
            repository_root=repository
        )
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            runner._validate_stokes_source_identity(
                audit_record,
                source_records,
                repository_root=repository,
            )

    def test_bf_source_identity_is_strict_and_physically_rehashed(self) -> None:
        audit_record, source_records = _bf_source_evidence()
        artifacts = runner._validate_bf_source_identity(audit_record, source_records)
        self.assertEqual(
            set(artifacts),
            {
                "SHA256SUMS",
                "solver/gen_data/pipeline/production.py",
                "solver/gen_data/trajectory_family_adapters.py",
            },
        )

        missing_identity = deepcopy(audit_record)
        del missing_identity["identity"]["historical_shared_source_snapshot_binding"]
        with self.assertRaisesRegex(TypeError, "historical source snapshot binding"):
            runner._validate_bf_source_identity(missing_identity, source_records)

        replaced_identity = deepcopy(audit_record)
        replaced_identity["identity"]["historical_shared_source_snapshot_binding"][
            "role"
        ] = "generation_source"
        with self.assertRaisesRegex(RuntimeError, "snapshot identity differs"):
            runner._validate_bf_source_identity(replaced_identity, source_records)

        missing_path_records = deepcopy(source_records)
        for source_record in missing_path_records:
            del source_record["run_spec"]["configuration"]["source_sha256"][
                "solver/gen_data/pipeline/archive.py"
            ]
        with self.assertRaisesRegex(RuntimeError, "frozen 19-path contract"):
            runner._validate_bf_source_identity(
                audit_record, tuple(missing_path_records)
            )

        extra_path_records = deepcopy(source_records)
        for source_record in extra_path_records:
            source_record["run_spec"]["configuration"]["source_sha256"][
                "solver/gen_data/pipeline/extra.py"
            ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "frozen 19-path contract"):
            runner._validate_bf_source_identity(audit_record, tuple(extra_path_records))

        copied_snapshot_root = self.root / "physical_snapshot_mutation"
        copied_snapshot_root.mkdir()
        for name in ("SHA256SUMS", "production.py", "trajectory_family_adapters.py"):
            shutil.copy2(
                runner.BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT / name,
                copied_snapshot_root / name,
            )
        copied_audit, copied_sources = _bf_source_evidence(
            snapshot_root=copied_snapshot_root
        )
        runner._validate_bf_source_identity(
            copied_audit,
            copied_sources,
            snapshot_root=copied_snapshot_root,
        )
        production = copied_snapshot_root / "production.py"
        production.write_bytes(production.read_bytes() + b"\n")
        with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
            runner._validate_bf_source_identity(
                copied_audit,
                copied_sources,
                snapshot_root=copied_snapshot_root,
            )

    def test_final_reauthentication_binds_snapshot_artifact_mutation(self) -> None:
        initial = self._release()
        calls = 0

        def authenticate(
            _path: Path, _audits: runner.AuditPaths
        ) -> runner.ReleaseBinding:
            nonlocal calls
            calls += 1
            if calls == 1:
                return initial
            snapshot = self.release_snapshot_root / "production.py"
            snapshot.write_bytes(snapshot.read_bytes() + b"\n")
            return self._release()

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=authenticate,
                stage_factory=lambda _config, _release, _output: (),
            ),
            1,
        )
        output_root = self.state_root / initial.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertFalse(status["release_evidence_unchanged_after_stages"])
        release_identity = runner._strict_json(output_root / "release_identity.json")
        self.assertEqual(
            release_identity["historical_source_snapshots"]["artifact_count"],
            7,
        )
        self.assertEqual(
            release_identity["historical_source_snapshots"]["family_count"],
            2,
        )
        self.assertEqual(
            {
                family: record["artifact_count"]
                for family, record in release_identity["historical_source_snapshots"][
                    "families"
                ].items()
            },
            {"benjamin_feir": 3, "stokes": 4},
        )
        families = release_identity["historical_source_snapshots"]["families"]
        self.assertNotEqual(
            families["stokes"]["artifacts"]["solver/gen_data/pipeline/production.py"][
                "path"
            ],
            families["benjamin_feir"]["artifacts"][
                "solver/gen_data/pipeline/production.py"
            ]["path"],
        )
        self.assertNotEqual(
            families["stokes"]["artifacts"]["SHA256SUMS"]["sha256"],
            families["benjamin_feir"]["artifacts"]["SHA256SUMS"]["sha256"],
        )

    def test_final_reauthentication_binds_stokes_snapshot_mutation(self) -> None:
        initial = self._release()
        calls = 0

        def authenticate(
            _path: Path, _audits: runner.AuditPaths
        ) -> runner.ReleaseBinding:
            nonlocal calls
            calls += 1
            if calls == 1:
                return initial
            snapshot = self.release_stokes_snapshot_root / "production.py"
            snapshot.write_bytes(snapshot.read_bytes() + b"\n")
            return self._release()

        self.assertEqual(
            runner.run_postcompletion(
                self._config(),
                authenticate=authenticate,
                stage_factory=lambda _config, _release, _output: (),
            ),
            1,
        )
        output_root = self.state_root / initial.combined.summary.sha256
        status = runner._strict_json(output_root / "status.json")
        self.assertFalse(status["release_evidence_unchanged_after_stages"])

    def test_renderer_validation_requires_every_ranking_and_figure(self) -> None:
        fixture = self._strict_stage_fixture()
        output_dir = fixture.root / "worst_cases"
        summary_path = output_dir / "summary.json"
        baseline = runner._strict_json(summary_path)
        runner.validate_renderer_output(output_dir, fixture.release)

        mutations = []

        missing_source = deepcopy(baseline)
        missing_source["sources"].pop()
        mutations.append(missing_source)

        duplicate_case = deepcopy(baseline)
        ranking = duplicate_case["families"]["stokes"]["rankings"]["eta_slope"]
        ranking[1] = deepcopy(ranking[0])
        mutations.append(duplicate_case)

        forged_ownership = deepcopy(baseline)
        case = forged_ownership["families"]["stokes"]["rankings"]["eta_slope"][0]
        case["accepted_index"] += 1
        mutations.append(forged_ownership)

        forged_interpretation = deepcopy(baseline)
        forged_interpretation["interpretation"] = "diagnostic claims release acceptance"
        mutations.append(forged_interpretation)

        forged_water = deepcopy(baseline)
        case = forged_water["families"]["stokes"]["rankings"]["eta_slope"][0]
        case["minimum_water_column"] = 123.0
        mutations.append(forged_water)

        forged_quantile = deepcopy(baseline)
        forged_quantile["families"]["stokes"]["quantiles"]["eta_slope"]["q100"] += 1.0
        mutations.append(forged_quantile)

        forged_sign_changes = deepcopy(baseline)
        stokes = forged_sign_changes["families"]["stokes"]
        for ranking_cases in stokes["rankings"].values():
            for case in ranking_cases:
                case["maximum_thresholded_gxi_sign_changes"] = 9_999
        for quantile in stokes["quantiles"]["gxi_sign_changes"]:
            stokes["quantiles"]["gxi_sign_changes"][quantile] = 9_999.0
        mutations.append(forged_sign_changes)

        missing_artifact = deepcopy(baseline)
        missing_artifact["artifacts"].pop(next(iter(missing_artifact["artifacts"])))
        mutations.append(missing_artifact)

        forged_animation_identity = deepcopy(baseline)
        animation = forged_animation_identity["animations"][
            "stokes_worst_eta_slope.gif"
        ]
        animation["case_id"] += 1
        mutations.append(forged_animation_identity)

        forged_animation_limits = deepcopy(baseline)
        animation = forged_animation_limits["animations"][
            "benjamin_feir_worst_combined.gif"
        ]
        animation["field_y_limits"]["eta"][1] += 1.0
        mutations.append(forged_animation_limits)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                _write_json(summary_path, mutation)
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    runner.validate_renderer_output(output_dir, fixture.release)
        _write_json(summary_path, baseline)

        noncanonical_path = deepcopy(baseline)
        name = next(iter(noncanonical_path["artifacts"]))
        artifact_path = Path(noncanonical_path["artifacts"][name]["path"])
        noncanonical_path["artifacts"][name]["path"] = str(
            artifact_path.parent / ".." / artifact_path.parent.name / artifact_path.name
        )
        _write_json(summary_path, noncanonical_path)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_renderer_output(output_dir, fixture.release)
        _write_json(summary_path, baseline)

        original_artifact = artifact_path.read_bytes()
        backup = artifact_path.with_suffix(artifact_path.suffix + ".original")
        artifact_path.rename(backup)
        artifact_path.symlink_to(backup.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "path differs"):
                runner.validate_renderer_output(output_dir, fixture.release)
        finally:
            artifact_path.unlink()
            backup.rename(artifact_path)
        self.assertEqual(artifact_path.read_bytes(), original_artifact)

        gif_name = "jonswap_tma_worst_combined.gif"
        gif_path = output_dir / gif_name
        original_gif = gif_path.read_bytes()
        mixed_duration = deepcopy(baseline)
        try:
            with Image.open(gif_path) as image:
                frames = [
                    image.seek(index) or image.copy() for index in range(image.n_frames)
                ]
            durations = [250] * len(frames)
            durations[len(durations) // 2] = 100
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
            )
            mixed_duration["artifacts"][gif_name] = runner._artifact(gif_path).record()
            _write_json(summary_path, mixed_duration)
            with self.assertRaisesRegex(RuntimeError, "GIF contract differs"):
                runner.validate_renderer_output(output_dir, fixture.release)
        finally:
            gif_path.write_bytes(original_gif)
            _write_json(summary_path, baseline)

        selected_case = baseline["families"]["stokes"]["rankings"]["eta_slope"][0]
        source = fixture.chunks[selected_case["source_index"]]
        manifest = runner._strict_json(Path(source["manifest_path"]))
        selected_shard = (
            Path(source["summary_path"]).parent
            / manifest["dataset_shards"][selected_case["shard_index"]]["path"]
        )
        original_shard = selected_shard.read_bytes()
        selected_shard.write_bytes(original_shard + b"mutated after release binding")
        try:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                runner.validate_renderer_output(output_dir, fixture.release)
        finally:
            selected_shard.write_bytes(original_shard)

    def test_renderer_validation_allows_water_fraction_above_one(self) -> None:
        fixture = self._strict_stage_fixture()
        output_dir = fixture.root / "worst_cases"
        summary_path = output_dir / "summary.json"
        summary = runner._strict_json(summary_path)
        summary["families"]["tanaka"]["quantiles"]["minimum_water_fraction"][
            "q100"
        ] = 1.1
        for ranking in summary["families"]["tanaka"]["rankings"].values():
            for case in ranking:
                case["minimum_water_column"] = 1.1 * case["depth"]
                case["minimum_water_fraction"] = 1.1
        _write_json(summary_path, summary)

        runner.validate_renderer_output(output_dir, fixture.release)

        summary["families"]["tanaka"]["quantiles"]["minimum_water_fraction"][
            "q000"
        ] = 0.0
        _write_json(summary_path, summary)
        with self.assertRaisesRegex(
            RuntimeError, "tanaka minimum_water_fraction quantile range differs"
        ):
            runner.validate_renderer_output(output_dir, fixture.release)

    def test_order_validation_requires_rescan_and_three_complete_cases(self) -> None:
        fixture = self._strict_stage_fixture()
        output = fixture.root / "jonswap_order_convergence.json"
        renderer_summary = fixture.root / "worst_cases/summary.json"
        baseline = runner._strict_json(output)
        runner.validate_order_output(
            output,
            release=fixture.release,
            renderer_summary=renderer_summary,
        )

        mutations = []
        no_cases = deepcopy(baseline)
        no_cases["cases"] = []
        mutations.append(no_cases)

        duplicate_case = deepcopy(baseline)
        duplicate_case["cases"][1]["identity"] = deepcopy(
            duplicate_case["cases"][0]["identity"]
        )
        mutations.append(duplicate_case)

        forged_audit_maximum = deepcopy(baseline)
        forged_audit_maximum["selection"]["completion_audit_global_maximum_value"] += (
            1.0e-5
        )
        mutations.append(forged_audit_maximum)

        forged_series_maximum = deepcopy(baseline)
        forged_series_maximum["cases"][0]["stored_float32_vs_recomputed_order6"][
            "maximum_relative_l2_difference"
        ] = 1.0
        mutations.append(forged_series_maximum)

        missing_chunk = deepcopy(baseline)
        missing_chunk["chunk_bindings"].pop()
        mutations.append(missing_chunk)

        reversed_timestamps = deepcopy(baseline)
        reversed_timestamps["invocation_started_at"] = "2027-01-01T00:00:00+00:00"
        reversed_timestamps["invocation_finished_at"] = "2026-01-01T00:00:00+00:00"
        mutations.append(reversed_timestamps)

        naive_timestamp = deepcopy(baseline)
        naive_timestamp["invocation_started_at"] = "2026-01-01T00:00:00"
        mutations.append(naive_timestamp)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                _write_json(output, mutation)
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    runner.validate_order_output(
                        output,
                        release=fixture.release,
                        renderer_summary=renderer_summary,
                    )
        _write_json(output, baseline)

        for name in ("proposal", "result", "shard"):
            target = Path(baseline["cases"][0]["sources"][f"{name}_path"])
            original = target.read_bytes()
            target.write_bytes(original + b"mutated after release binding")
            try:
                with self.subTest(physical_source=name):
                    with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                        runner.validate_order_output(
                            output,
                            release=fixture.release,
                            renderer_summary=renderer_summary,
                        )
            finally:
                target.write_bytes(original)

        renderer = runner._strict_json(renderer_summary)
        renderer_case = renderer["families"]["jonswap_tma"]["rankings"][
            "gxi_high_band"
        ][0]
        alternate_source = next(
            (index, source)
            for index, source in enumerate(renderer["sources"])
            if source["family"] == "jonswap_tma"
            and source["root"] != renderer_case["source_root"]
        )
        renderer_case["source_index"] = alternate_source[0]
        renderer_case["source_root"] = alternate_source[1]["root"]
        _write_json(renderer_summary, renderer)
        cross = baseline["selection"]["final_renderer_cross_check"]
        cross["sha256"] = runner._sha256(renderer_summary)
        _write_json(output, baseline)
        with (
            patch.object(runner, "validate_renderer_output", return_value={}),
            self.assertRaisesRegex(RuntimeError, "source ownership differs"),
        ):
            runner.validate_order_output(
                output,
                release=fixture.release,
                renderer_summary=renderer_summary,
            )

    def test_training_validation_requires_stats_and_sampler_contract(self) -> None:
        fixture = self._strict_stage_fixture()
        output = fixture.root / "training_handoff_audit.json"
        baseline = runner._strict_json(output)
        runner.validate_training_output(output, fixture.release.combined.summary)

        mutations = []
        missing_view = deepcopy(baseline)
        del missing_view["dataset_view"]
        mutations.append(missing_view)

        extra_view_field = deepcopy(baseline)
        extra_view_field["dataset_view"]["sentinel"] = True
        mutations.append(extra_view_field)

        reordered_shards = deepcopy(baseline)
        reordered_shards["dataset_view"]["shards"][0:2] = reversed(
            reordered_shards["dataset_view"]["shards"][0:2]
        )
        mutations.append(reordered_shards)

        redistributed_selection = deepcopy(baseline)
        redistributed_selection["dataset_view"]["shards"][0][
            "selected_training_rows"
        ] -= 1
        redistributed_selection["dataset_view"]["shards"][1][
            "selected_training_rows"
        ] += 1
        mutations.append(redistributed_selection)

        wrong_selected_total = deepcopy(baseline)
        wrong_selected_total["dataset_view"]["shards"][0]["selected_training_rows"] -= 1
        mutations.append(wrong_selected_total)

        missing_stats = deepcopy(baseline)
        del missing_stats["training_normalization"]["stats"]
        mutations.append(missing_stats)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                _write_json(output, mutation)
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    runner.validate_training_output(
                        output, fixture.release.combined.summary
                    )
        _write_json(output, baseline)

        shard = Path(baseline["dataset_view"]["shards"][0]["path"])
        original = shard.read_bytes()
        shard.write_bytes(original + b"corrupted after training audit")
        try:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                runner.validate_training_output(
                    output, fixture.release.combined.summary
                )
        finally:
            shard.write_bytes(original)

    def test_training_validation_authenticates_exact_implementation_set(self) -> None:
        repository = self.root / "training_repository"
        for relative, _ in runner.TRAINING_IMPLEMENTATION_FILES:
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
        valid = _training_implementation_record(repository)

        with patch.object(runner, "ROOT", repository):
            self.assertEqual(
                runner.validate_training_implementation(valid),
                valid,
            )

            for name, mutate, message in (
                (
                    "extra",
                    lambda record: record["files"].append(
                        {
                            "path": "extra.py",
                            "role": "extra",
                            "bytes": 0,
                            "sha256": "0" * 64,
                        }
                    ),
                    "file set differs",
                ),
                (
                    "missing",
                    lambda record: record["files"].pop(),
                    "file set differs",
                ),
                (
                    "rebound_path",
                    lambda record: record["files"][0].update(
                        {"path": record["files"][1]["path"]}
                    ),
                    "path or role differs",
                ),
            ):
                with self.subTest(name=name):
                    changed = deepcopy(valid)
                    mutate(changed)
                    payload = {
                        key: changed[key]
                        for key in ("schema", "files", "semantic_relationship")
                    }
                    changed["fingerprint"] = runner._canonical_sha256(payload)
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner.validate_training_implementation(changed)

            trainer_util = repository / "train-jax-10m/util.py"
            trainer_util.write_text("# physically changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                runner.validate_training_implementation(valid)

    def test_training_validation_detects_code_mutation_during_rehash(self) -> None:
        repository = self.root / "training_repository_toctou"
        for relative, _ in runner.TRAINING_IMPLEMENTATION_FILES:
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
        valid = _training_implementation_record(repository)
        target = repository / "train-jax-10m/util.py"
        original_artifact = runner._artifact
        mutated = False

        def mutate_after_hash(path: Path, **kwargs: object) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, **kwargs)
            if path.resolve() == target.resolve() and not mutated:
                target.write_text("# changed during hash\n", encoding="utf-8")
                mutated = True
            return observed

        with (
            patch.object(runner, "ROOT", repository),
            patch.object(runner, "_artifact", side_effect=mutate_after_hash),
            self.assertRaisesRegex(RuntimeError, "changed while being hashed"),
        ):
            runner.validate_training_implementation(valid)

    def test_training_validation_sweeps_earlier_files_after_all_hashes(self) -> None:
        repository = self.root / "training_repository_set_sweep"
        for relative, _ in runner.TRAINING_IMPLEMENTATION_FILES:
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
        valid = _training_implementation_record(repository)
        first = repository / runner.TRAINING_IMPLEMENTATION_FILES[0][0]
        original_artifact = runner._artifact
        mutated = False

        def mutate_earlier_file(path: Path, **kwargs: object) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, **kwargs)
            if path.name == "util.py" and not mutated:
                first.write_text("# changed after earlier hash\n", encoding="utf-8")
                mutated = True
            return observed

        with (
            patch.object(runner, "ROOT", repository),
            patch.object(runner, "_artifact", side_effect=mutate_earlier_file),
            self.assertRaisesRegex(RuntimeError, "changed after authentication"),
        ):
            runner.validate_training_implementation(valid)

    def test_training_validation_rejects_symlink_substitution(self) -> None:
        for substitution in ("direct", "ancestor"):
            with self.subTest(substitution=substitution):
                repository = self.root / f"training_repository_{substitution}"
                for relative, _ in runner.TRAINING_IMPLEMENTATION_FILES:
                    path = repository / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"# {relative}\n", encoding="utf-8")
                valid = _training_implementation_record(repository)
                original_artifact = runner._artifact
                substituted = False

                def substitute_after_hash(
                    path: Path, **kwargs: object
                ) -> runner.Artifact:
                    nonlocal substituted
                    observed = original_artifact(path, **kwargs)
                    trigger = (
                        path.name == "util.py"
                        if substitution == "direct"
                        else path.name == "1d_dno_fno_jax.py"
                    )
                    if trigger and not substituted:
                        if substitution == "direct":
                            first = (
                                repository / runner.TRAINING_IMPLEMENTATION_FILES[0][0]
                            )
                            backup = first.with_suffix(".original.py")
                            first.rename(backup)
                            first.symlink_to(backup.name)
                        else:
                            trainer_dir = repository / "train-jax-10m"
                            backup = repository / "trainer-original"
                            trainer_dir.rename(backup)
                            trainer_dir.symlink_to(
                                backup.name, target_is_directory=True
                            )
                        substituted = True
                    return observed

                with (
                    patch.object(runner, "ROOT", repository),
                    patch.object(
                        runner,
                        "_artifact",
                        side_effect=substitute_after_hash,
                    ),
                    self.assertRaisesRegex(ValueError, "symbolic link"),
                ):
                    runner.validate_training_implementation(valid)

    def test_figure_validation_requires_exact_selected_cases(self) -> None:
        fixture = self._strict_stage_fixture()
        stem = fixture.root / "family_case_examples"
        output = stem.with_suffix(".json")
        baseline = runner._strict_json(output)
        runner.validate_figure_output(stem, fixture.release)

        mutations = []
        no_cases = deepcopy(baseline)
        no_cases["cases"] = []
        mutations.append(no_cases)

        forged_selection = deepcopy(baseline)
        forged_selection["cases"][0]["selection"]["lower_median_index_zero_based"] += 1
        mutations.append(forged_selection)

        forged_ownership = deepcopy(baseline)
        forged_ownership["cases"][0]["owned_row"]["shard_row"] += 1
        mutations.append(forged_ownership)

        missing_source_proof = deepcopy(baseline)
        del missing_source_proof["cases"][0]["source_trajectory_map"]
        mutations.append(missing_source_proof)

        forged_release = deepcopy(baseline)
        forged_release["release_contract"]["source_count"] -= 1
        mutations.append(forged_release)

        missing_implementation = deepcopy(baseline)
        del missing_implementation["figure_implementation"]
        mutations.append(missing_implementation)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                _write_json(output, mutation)
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    runner.validate_figure_output(stem, fixture.release)
        _write_json(output, baseline)

        selected_shard = Path(baseline["cases"][0]["selected_shard"]["path"])
        original_shard = selected_shard.read_bytes()
        selected_shard.write_bytes(original_shard + b"mutated after release binding")
        try:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                runner.validate_figure_output(stem, fixture.release)
        finally:
            selected_shard.write_bytes(original_shard)

    def test_stage_validators_rehash_cached_release_parents(self) -> None:
        fixture = self._strict_stage_fixture()
        combined = fixture.release.combined.summary.path
        original_combined = combined.read_bytes()
        combined.write_bytes(original_combined + b"\n")
        validators = (
            lambda: runner.validate_renderer_output(
                fixture.root / "worst_cases", fixture.release
            ),
            lambda: runner.validate_order_output(
                fixture.root / "jonswap_order_convergence.json",
                release=fixture.release,
                renderer_summary=fixture.root / "worst_cases/summary.json",
            ),
            lambda: runner.validate_training_output(
                fixture.root / "training_handoff_audit.json",
                fixture.release.combined.summary,
            ),
            lambda: runner.validate_figure_output(
                fixture.root / "family_case_examples", fixture.release
            ),
        )
        try:
            for index, validate in enumerate(validators):
                with self.subTest(combined_validator=index):
                    with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                        validate()
        finally:
            combined.write_bytes(original_combined)

        audit = fixture.release.audits["stokes"].path
        original_audit = audit.read_bytes()
        audit.write_bytes(original_audit + b"\n")
        audit_validators = (validators[0], validators[1], validators[3])
        try:
            for index, validate in enumerate(audit_validators):
                with self.subTest(audit_validator=index):
                    with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                        validate()
        finally:
            audit.write_bytes(original_audit)

    def test_stage_validators_reject_noncanonical_parent_path_spellings(self) -> None:
        fixture = self._strict_stage_fixture()

        def alternate(path_value: str) -> str:
            path = Path(path_value)
            return str(path.parent / ".." / path.parent.name / path.name)

        renderer_path = fixture.root / "worst_cases/summary.json"
        renderer = runner._strict_json(renderer_path)
        original_renderer = deepcopy(renderer)
        renderer["source_binding"]["combined_summary_path"] = alternate(
            renderer["source_binding"]["combined_summary_path"]
        )
        _write_json(renderer_path, renderer)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_renderer_output(renderer_path.parent, fixture.release)
        _write_json(renderer_path, original_renderer)

        order_path = fixture.root / "jonswap_order_convergence.json"
        order = runner._strict_json(order_path)
        original_order = deepcopy(order)
        order["completion_audit"]["path"] = alternate(order["completion_audit"]["path"])
        _write_json(order_path, order)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_order_output(
                order_path,
                release=fixture.release,
                renderer_summary=renderer_path,
            )
        _write_json(order_path, original_order)

        training_path = fixture.root / "training_handoff_audit.json"
        training = runner._strict_json(training_path)
        original_training = deepcopy(training)
        training["combined_summary"]["path"] = alternate(
            training["combined_summary"]["path"]
        )
        _write_json(training_path, training)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_training_output(
                training_path, fixture.release.combined.summary
            )
        _write_json(training_path, original_training)

        figure_path = fixture.root / "family_case_examples.json"
        figure = runner._strict_json(figure_path)
        figure["combined_summary"]["path"] = alternate(
            figure["combined_summary"]["path"]
        )
        _write_json(figure_path, figure)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_figure_output(
                fixture.root / "family_case_examples", fixture.release
            )

    @unittest.skipUnless(
        (
            runner.ROOT / "outputs/paper_corpus_cap4_revision2_20260728/"
            "stokes_completion_binding.json"
        ).is_file(),
        "local immutable Stokes/BF release artifacts are unavailable",
    )
    def test_current_real_stokes_and_bf_artifacts_authenticate_read_only(self) -> None:
        stokes_root = runner.ROOT / "outputs/paper_corpus_cap4_revision2_20260728"
        stokes, stokes_sources, stokes_snapshots = runner._validate_family_audit(
            runner.FamilyAuditSpec(
                "stokes",
                "paper_corpus_stokes_revision2_completion_binding_v1",
                stokes_root / "stokes_completion_binding.json",
                runner.EXPECTED_ROWS_BY_FAMILY["stokes"],
                nested_counts=True,
            )
        )
        legacy = runner._validate_stokes_legacy(
            stokes.path, stokes_root / "stokes_completion_audit.json"
        )
        bf_path = (
            runner.ROOT
            / "outputs/paper_corpus_bf_revision4_jonswap_revision3_literature_aligned_v1/"
            "benjamin_feir_completion_audit.json"
        )
        bf, bf_sources, bf_snapshots = runner._validate_family_audit(
            runner.FamilyAuditSpec(
                "benjamin_feir",
                "paper_corpus_benjamin_feir_revision4_completion_audit_v1",
                bf_path,
                runner.EXPECTED_ROWS_BY_FAMILY["benjamin_feir"],
            )
        )
        self.assertEqual(stokes.sha256, runner.EXPECTED_STOKES_BINDING_SHA256)
        self.assertEqual(legacy.sha256, runner.EXPECTED_STOKES_LEGACY_SHA256)
        self.assertEqual(len(stokes_sources), 6)
        self.assertEqual(len(stokes_snapshots), 4)
        self.assertEqual(len(bf_sources), 6)
        self.assertEqual(len(bf_snapshots), 3)
        self.assertEqual(
            bf.sha256,
            "c155b35276d0cfef6844f6b0db3e91da3ce15579ad8b0f0ca482f44cc912753c",
        )

    def test_main_preserves_virtual_environment_python_symlink(self) -> None:
        python_target = self.root / "python-target"
        python_target.write_text("#!/bin/sh\n", encoding="utf-8")
        python_target.chmod(0o755)
        python_link = self.root / "venv-python"
        python_link.symlink_to(python_target.name)

        with patch.object(runner, "run_postcompletion", return_value=0) as run:
            status = runner.main(
                [
                    "--python",
                    str(python_link),
                    "--state-root",
                    str(self.root / "state"),
                ]
            )

        self.assertEqual(status, 0)
        config = run.call_args.args[0]
        self.assertEqual(config.python, python_link.absolute())
        self.assertTrue(config.python.is_symlink())


class CombinedSummaryAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.name = "paper_corpus_all_splits_c16384"
        self.summary = self.root / f"{self.name}.summary.json"
        self.manifest = self.root / f"{self.name}.dataset.json"
        self.trajectory_map = self.root / f"{self.name}.trajectory_map.npz"
        self.contract = {"canonical": "proposal-derived"}
        _write_json(
            self.manifest,
            {
                "dataset_contract": self.contract,
                "dataset_contract_fingerprint": runner._canonical_sha256(self.contract),
            },
        )
        self.trajectory_map.write_bytes(b"canonical-map")
        self.source_paths = tuple(
            self.root / "sources" / f"source-{index:02d}.summary.json"
            for index in range(runner.FINAL_SOURCE_COUNT)
        )
        self.preflight = {
            "schema": runner.COMBINED_PREFLIGHT_SCHEMA,
            "expected_rows_by_split_and_family": (
                runner._expected_rows_by_split_and_family()
            ),
            "chunks": [
                {"summary_path": str(path.resolve())} for path in self.source_paths
            ],
        }
        self.plan = SimpleNamespace(
            splits=tuple(SimpleNamespace(value=split) for split in runner.SPLIT_ORDER),
            accepted_cases_per_family_by_split=(runner.CASES_PER_FAMILY_BY_SPLIT),
            accepted_cases=runner.FINAL_ACCEPTED_CASES,
            expected_rows=runner.FINAL_RETAINED_ROWS,
        )
        self.view = {
            "dataset_contract_fingerprint": runner._canonical_sha256(self.contract),
            "manifest": runner._artifact(self.manifest).record(),
            "trajectory_map": runner._artifact(self.trajectory_map).record(),
        }
        self._write_summary(self.preflight, self.view)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_summary(self, preflight: object, view: object) -> None:
        _write_json(
            self.summary,
            {
                "schema": runner.COMBINED_SUMMARY_SCHEMA,
                "status": "complete",
                "preflight": preflight,
                "dataset_view": view,
            },
        )

    def _patches(self, *, validated_view: object | None = None) -> tuple[object, ...]:
        return (
            patch.object(
                runner.combined_view_builder,
                "preflight",
                return_value=(
                    self.plan,
                    self.root.resolve(),
                    self.name,
                    self.preflight,
                ),
            ),
            patch.object(
                runner.combined_view_builder,
                "_validate_view",
                return_value=validated_view or self.view,
            ),
            patch.object(
                runner,
                "_reconstruct_dataset_contract",
                return_value=self.contract,
            ),
        )

    def test_canonical_preflight_view_and_contract_authenticate(self) -> None:
        preflight_patch, validate_patch, contract_patch = self._patches()
        with preflight_patch as preflight, validate_patch, contract_patch:
            observed = runner.authenticate_combined_summary(self.summary)
        self.assertEqual(observed.summary.path, self.summary.resolve())
        preflight.assert_called_once_with(
            self.source_paths,
            output_root=self.root.resolve(),
            name=self.name,
        )

    def test_coherently_rebound_wrong_dataset_contract_fails_closed(self) -> None:
        wrong_contract = {**self.contract, "coherent_but_wrong": True}
        _write_json(
            self.manifest,
            {
                "dataset_contract": wrong_contract,
                "dataset_contract_fingerprint": runner._canonical_sha256(
                    wrong_contract
                ),
            },
        )
        wrong_view = {
            **self.view,
            "dataset_contract_fingerprint": runner._canonical_sha256(wrong_contract),
            "manifest": runner._artifact(self.manifest).record(),
        }
        self._write_summary(self.preflight, wrong_view)
        preflight_patch, validate_patch, contract_patch = self._patches(
            validated_view=wrong_view
        )
        with (
            preflight_patch,
            validate_patch,
            contract_patch,
            self.assertRaisesRegex(RuntimeError, "proposal reconstruction"),
        ):
            runner.authenticate_combined_summary(self.summary)

    def test_stored_preflight_must_equal_canonical_reconstruction(self) -> None:
        mutated = deepcopy(self.preflight)
        mutated["coherent_but_wrong"] = True
        self._write_summary(mutated, self.view)
        preflight_patch, validate_patch, contract_patch = self._patches()
        with (
            preflight_patch,
            validate_patch,
            contract_patch,
            self.assertRaisesRegex(RuntimeError, "preflight differs"),
        ):
            runner.authenticate_combined_summary(self.summary)

    def test_stored_view_must_equal_canonical_validation(self) -> None:
        mutated = {**self.view, "coherent_but_wrong": True}
        self._write_summary(self.preflight, mutated)
        preflight_patch, validate_patch, contract_patch = self._patches()
        with (
            preflight_patch,
            validate_patch,
            contract_patch,
            self.assertRaisesRegex(RuntimeError, "dataset_view differs"),
        ):
            runner.authenticate_combined_summary(self.summary)

    def test_parent_mutation_after_view_validation_fails_closed(self) -> None:
        def validate_and_mutate(*_args: object, **_kwargs: object) -> object:
            self.trajectory_map.write_bytes(b"changed after validation")
            return self.view

        preflight_patch, _, contract_patch = self._patches()
        with (
            preflight_patch,
            patch.object(
                runner.combined_view_builder,
                "_validate_view",
                side_effect=validate_and_mutate,
            ),
            contract_patch,
            self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"),
        ):
            runner.authenticate_combined_summary(self.summary)


class ReleaseGenerationIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repository"
        self.repository.mkdir()
        current_paths = set().union(
            *(
                runner.EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                - runner.HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                for family in runner.FAMILY_ORDER
            )
        )
        for relative in sorted(current_paths):
            path = self.repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source:{relative}\n", encoding="utf-8")
        for relative in runner.DEPENDENCY_FILE_PATHS:
            (self.repository / relative).write_text(
                f"dependency:{relative}\n",
                encoding="utf-8",
            )
        self.environment = {
            "python": {
                "implementation": runner.python_platform.python_implementation(),
                "version": runner.python_platform.python_version(),
            },
            "packages": {
                "jax": runner.jax.__version__,
                "jaxlib": runner.jaxlib.__version__,
                "numpy": runner.np.__version__,
            },
            "files_sha256": {
                relative: runner._sha256(self.repository / relative)
                for relative in runner.DEPENDENCY_FILE_PATHS
            },
        }
        self.source_maps = {
            family: {
                relative: (
                    runner._canonical_sha256({"historical": family, "path": relative})
                    if relative
                    in runner.HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                    else runner._sha256(self.repository / relative)
                )
                for relative in sorted(
                    runner.EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                )
            }
            for family in runner.FAMILY_ORDER
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _family_material(
        self,
        family: str,
        *,
        source_map: dict[str, str] | None = None,
        environment: dict[str, object] | None = None,
    ) -> tuple[
        dict[str, object],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        selected_sources = source_map or self.source_maps[family]
        selected_environment = environment or self.environment
        source_fingerprint = runner._canonical_sha256(selected_sources)
        dependency_fingerprint = runner._canonical_sha256(selected_environment)
        source_records = [
            {
                "run_spec": {
                    "configuration": {
                        "source_sha256": deepcopy(selected_sources),
                        "dependency_environment": deepcopy(selected_environment),
                    }
                }
            }
            for _ in range(runner.SOURCE_COUNT_BY_FAMILY[family])
        ]
        source_key = (
            "source_fingerprint" if family == "stokes" else "source_sha256_fingerprint"
        )
        dependency_key = (
            "dependency_fingerprint"
            if family == "stokes"
            else "dependency_environment_fingerprint"
        )
        audit = {
            "identity": {
                source_key: source_fingerprint,
                dependency_key: dependency_fingerprint,
            }
        }
        combined_chunks = [
            {
                "family": family,
                "source_fingerprint": source_fingerprint,
                "dependency_fingerprint": dependency_fingerprint,
            }
            for _ in range(runner.SOURCE_COUNT_BY_FAMILY[family])
        ]
        return audit, source_records, combined_chunks

    def _validate_family(
        self,
        family: str,
        *,
        audit: dict[str, object] | None = None,
        source_records: list[dict[str, object]] | None = None,
        combined_chunks: list[dict[str, object]] | None = None,
        artifact_cache: dict[Path, runner.Artifact] | None = None,
    ) -> runner.GenerationSourceBinding:
        default_audit, default_sources, default_chunks = self._family_material(family)
        binding, _ = runner._validate_family_generation_identity(
            family=family,
            audit_record=audit or default_audit,
            source_records=source_records or default_sources,
            combined_chunks=combined_chunks or default_chunks,
            repository_root=self.repository,
            artifact_cache=artifact_cache if artifact_cache is not None else {},
        )
        return binding

    def _global_material(
        self,
        *,
        environment: dict[str, object] | None = None,
    ) -> tuple[
        runner.CombinedBinding,
        dict[str, runner.Artifact],
        dict[str, tuple[dict[str, object], ...]],
    ]:
        selected_environment = environment or self.environment
        evidence_root = self.base / "evidence"
        audits: dict[str, runner.Artifact] = {}
        source_bindings: dict[str, tuple[dict[str, object], ...]] = {}
        combined_chunks: list[dict[str, object]] = []
        for family in runner.FAMILY_ORDER:
            audit, source_records, chunks = self._family_material(
                family,
                environment=selected_environment,
            )
            audit_path = evidence_root / f"{family}.audit.json"
            _write_json(audit_path, audit)
            audits[family] = runner._artifact(audit_path)
            family_bindings: list[dict[str, object]] = []
            for index, source_record in enumerate(source_records):
                summary_path = evidence_root / family / f"source-{index}.json"
                _write_json(summary_path, source_record)
                manifest_path = summary_path.with_suffix(".dataset.json")
                map_path = summary_path.with_suffix(".trajectory_map.npz")
                _write_json(manifest_path, {"source": index})
                map_path.write_bytes(f"map:{family}:{index}".encode())
                summary = runner._artifact(summary_path)
                manifest = runner._artifact(manifest_path)
                trajectory_map = runner._artifact(map_path)
                family_bindings.append(
                    {
                        "summary_path": str(summary.path),
                        "summary_bytes": summary.bytes,
                        "summary_sha256": summary.sha256,
                        "manifest_path": str(manifest.path),
                        "manifest_bytes": manifest.bytes,
                        "manifest_sha256": manifest.sha256,
                        "trajectory_map_path": str(trajectory_map.path),
                        "trajectory_map_bytes": trajectory_map.bytes,
                        "trajectory_map_sha256": trajectory_map.sha256,
                    }
                )
            source_bindings[family] = tuple(family_bindings)
            combined_chunks.extend(chunks)
        placeholder = runner._artifact(self.repository / "pyproject.toml")
        combined = runner.CombinedBinding(
            summary=placeholder,
            manifest=placeholder,
            trajectory_map=placeholder,
            chunks=tuple(combined_chunks),
        )
        return combined, audits, source_bindings

    def test_current_tanaka_without_historical_compatibility_reconstructs(self) -> None:
        family = "tanaka"
        root = self.base / "current_tanaka_contract"
        paths = BatchPaths.under(
            root,
            family=family,
            split=SplitId.TRAIN.value,
            batch_id=0,
        )
        paths.proposal.parent.mkdir(parents=True, exist_ok=True)
        numerical = {
            "nx": 1_024,
            "length": float(2.0 * runner.np.pi),
            "gravity": 9.81,
            "dno_order": 6,
            "pad_factor": 4,
            "maximum_wavenumber": 128.0,
            "dtype": "float64",
            "fine_dt": 0.01,
        }
        execution = {"role": "shared_target", "numerical": numerical}
        new_source_map = {"fresh_corrected_tanaka.py": "e" * 64}
        metadata = {
            "case_kind": "trajectory",
            "trajectory_execution": execution,
            "additional_metadata": {
                "run_spec": {
                    "configuration": {
                        "schema": "paper_corpus_quota_configuration_v1",
                        "dependency_environment": self.environment,
                        "source_sha256": new_source_map,
                        "execution_platform": "cpu",
                    }
                }
            },
        }
        fingerprint = "c" * 64
        runner.np.savez(
            paths.proposal,
            metadata_json=runner.np.asarray(json.dumps(metadata)),
            family_id=runner.np.asarray(
                int(runner.combined_view_builder.FAMILY_IDS[family])
            ),
            revision_id=runner.np.asarray(runner.REVISION_BY_FAMILY[family]),
            split_id=runner.np.asarray(0),
            batch_id=runner.np.asarray(0),
            case_id=runner.np.asarray([1]),
            cell_id=runner.np.asarray([1]),
            config_fingerprint=runner.np.asarray(fingerprint),
        )
        summary = root / "source.summary.json"
        _write_json(summary, {})
        chunk = runner.combined_view_builder.CompletedChunk(
            summary_path=summary,
            summary_sha256=runner._sha256(summary),
            root=root,
            family=family,
            revision_id=runner.REVISION_BY_FAMILY[family],
            split=SplitId.TRAIN,
            stream_id=0,
            accepted_before=0,
            accepted_count=1,
            accepted_after=1,
            attempted_count=1,
            fingerprint=fingerprint,
            dependency_fingerprint=runner._canonical_sha256(self.environment),
            execution_fingerprint=runner._canonical_sha256(execution),
            generation_compatibility_id=None,
            source_fingerprint=runner._canonical_sha256(new_source_map),
            source_sha256=new_source_map,
            execution_platform="cpu",
            batches=(paths,),
        )
        plan = runner.combined_view_builder.CombinedCorpusPlan(
            chunks=(chunk,),
            splits=(SplitId.TRAIN,),
            accepted_cases_per_family_by_split={"train": 1},
            attempted_cases_by_split={"train": 1},
            attempted_cases=1,
            accepted_cases=1,
            expected_rows=runner.ROWS_PER_CASE[family],
            fingerprints=(fingerprint,),
        )
        contract = runner._reconstruct_dataset_contract(plan)
        generation = contract["generation_identity"]
        family_record = generation["family_revisions"][0]
        self.assertNotIn("generation_compatibility_id", family_record)
        self.assertEqual(
            family_record["generation_variants"][0]["source_sha256"],
            new_source_map,
        )

    def test_historical_terminal_group_rejects_mutation_and_symlink(self) -> None:
        root = self.base / "historical_snapshot"
        root.mkdir()
        source = root / "historical.py"
        source.write_text("# historical\n", encoding="utf-8")
        expected = runner._artifact(source)
        groups = {"test historical snapshot": (root, {source.name: expected})}

        source.write_text("# changed late\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
            runner._terminal_reauthenticate_artifact_groups(groups)

        source.write_text("# historical\n", encoding="utf-8")
        expected = runner._artifact(source)
        groups = {"test historical snapshot": (root, {source.name: expected})}
        backup = source.with_suffix(".original.py")
        source.rename(backup)
        source.symlink_to(backup.name)
        try:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                runner._terminal_reauthenticate_artifact_groups(groups)
        finally:
            source.unlink()
            backup.rename(source)

    def test_all_four_exact_maps_bind_32_unique_current_files(self) -> None:
        cache: dict[Path, runner.Artifact] = {}
        bindings = {
            family: self._validate_family(family, artifact_cache=cache)
            for family in runner.FAMILY_ORDER
        }
        self.assertEqual(len(cache), 32)
        self.assertEqual(
            {
                family: len(binding.source_sha256)
                for family, binding in bindings.items()
            },
            {"stokes": 14, "tanaka": 21, "benjamin_feir": 19, "jonswap_tma": 20},
        )
        self.assertEqual(
            {
                family: len(binding.current_sources)
                for family, binding in bindings.items()
            },
            {"stokes": 11, "tanaka": 21, "benjamin_feir": 17, "jonswap_tma": 20},
        )
        self.assertEqual(
            {
                family: len(binding.historical_source_paths)
                for family, binding in bindings.items()
            },
            {"stokes": 3, "tanaka": 0, "benjamin_feir": 2, "jonswap_tma": 0},
        )

    def test_global_binding_covers_every_chunk_and_family(self) -> None:
        combined, audits, sources = self._global_material()
        dependency, generation = runner._authenticate_release_generation_identity(
            combined=combined,
            audits=audits,
            audit_sources_by_family=sources,
            repository_root=self.repository,
        )
        self.assertEqual(runner.RELEASE_IDENTITY_SCHEMA.rsplit("_", 1)[-1], "v2")
        self.assertEqual(dependency.chunk_count, runner.FINAL_SOURCE_COUNT)
        self.assertEqual(set(generation), set(runner.FAMILY_ORDER))
        self.assertEqual(
            dependency.record()["schema"],
            runner.DEPENDENCY_BINDING_SCHEMA,
        )

    def test_missing_extra_malformed_and_late_unequal_maps_fail_closed(self) -> None:
        family = "jonswap_tma"
        audit, records, chunks = self._family_material(family)
        first_path = next(iter(self.source_maps[family]))
        mutations = {
            "missing": lambda value: value.pop(first_path),
            "extra": lambda value: value.__setitem__("extra.py", "a" * 64),
            "parent path": lambda value: value.__setitem__("../escape.py", "a" * 64),
            "digest": lambda value: value.__setitem__(first_path, "A" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(records)
                source_map = changed[0]["run_spec"]["configuration"]["source_sha256"]
                mutate(source_map)
                with self.assertRaises((RuntimeError, ValueError)):
                    self._validate_family(
                        family,
                        audit=audit,
                        source_records=changed,
                        combined_chunks=chunks,
                    )
        changed = deepcopy(records)
        changed[-1]["run_spec"]["configuration"]["source_sha256"][first_path] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "unequal exact source maps"):
            self._validate_family(
                family,
                audit=audit,
                source_records=changed,
                combined_chunks=chunks,
            )

    def test_cross_family_shared_digest_conflict_fails_closed(self) -> None:
        cache: dict[Path, runner.Artifact] = {}
        self._validate_family("stokes", artifact_cache=cache)
        family = "tanaka"
        changed_map = dict(self.source_maps[family])
        changed_map["solver/solvers/dno_series_jax.py"] = "f" * 64
        audit, records, chunks = self._family_material(
            family,
            source_map=changed_map,
        )
        with self.assertRaisesRegex(RuntimeError, "conflicts with another family"):
            self._validate_family(
                family,
                audit=audit,
                source_records=records,
                combined_chunks=chunks,
                artifact_cache=cache,
            )

    def test_jonswap_and_tanaka_current_byte_mutations_fail_closed(self) -> None:
        targets = {
            "jonswap_tma": "solver/gen_data/jonswap_tma.py",
            "tanaka": "solver/tanaka_ICs/modified_tanaka.py",
        }
        for family, relative in targets.items():
            with self.subTest(family=family):
                path = self.repository / relative
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "artifact SHA-256 differs"
                    ):
                        self._validate_family(family)
                finally:
                    path.write_bytes(original)

    def test_current_source_direct_and_ancestor_symlinks_fail_closed(self) -> None:
        direct = self.repository / "solver/gen_data/generate_tanaka_dataset_v2.py"
        direct_backup = direct.with_suffix(".original.py")
        direct.rename(direct_backup)
        direct.symlink_to(direct_backup.name)
        try:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                self._validate_family("tanaka")
        finally:
            direct.unlink()
            direct_backup.rename(direct)

        ancestor = self.repository / "solver/tanaka_ICs"
        ancestor_backup = self.repository / "solver/tanaka_ICs-original"
        ancestor.rename(ancestor_backup)
        ancestor.symlink_to(ancestor_backup.name, target_is_directory=True)
        try:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                self._validate_family("tanaka")
        finally:
            ancestor.unlink()
            ancestor_backup.rename(ancestor)

    def test_late_unequal_dependency_record_fails_closed(self) -> None:
        family = "jonswap_tma"
        audit, records, chunks = self._family_material(family)
        records[-1]["run_spec"]["configuration"]["dependency_environment"]["packages"][
            "numpy"
        ] = "different"
        with self.assertRaisesRegex(RuntimeError, "unequal dependencies"):
            self._validate_family(
                family,
                audit=audit,
                source_records=records,
                combined_chunks=chunks,
            )

    def test_global_runtime_and_dependency_file_drift_fail_closed(self) -> None:
        changed_environment = deepcopy(self.environment)
        changed_environment["packages"]["jax"] = "not-current"
        combined, audits, sources = self._global_material(
            environment=changed_environment
        )
        with self.assertRaisesRegex(RuntimeError, "environment is not current"):
            runner._authenticate_release_generation_identity(
                combined=combined,
                audits=audits,
                audit_sources_by_family=sources,
                repository_root=self.repository,
            )

        combined, audits, sources = self._global_material()
        dependency = self.repository / "uv.lock"
        dependency.write_bytes(dependency.read_bytes() + b"changed")
        with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
            runner._authenticate_release_generation_identity(
                combined=combined,
                audits=audits,
                audit_sources_by_family=sources,
                repository_root=self.repository,
            )

    def test_dependency_direct_and_root_ancestor_symlinks_fail_closed(self) -> None:
        combined, audits, sources = self._global_material()
        dependency = self.repository / "pyproject.toml"
        backup = dependency.with_suffix(".original.toml")
        dependency.rename(backup)
        dependency.symlink_to(backup.name)
        try:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                runner._authenticate_release_generation_identity(
                    combined=combined,
                    audits=audits,
                    audit_sources_by_family=sources,
                    repository_root=self.repository,
                )
        finally:
            dependency.unlink()
            backup.rename(dependency)

        repository_backup = self.base / "repository-original"
        self.repository.rename(repository_backup)
        self.repository.symlink_to(repository_backup.name, target_is_directory=True)
        try:
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                runner._authenticate_release_generation_identity(
                    combined=combined,
                    audits=audits,
                    audit_sources_by_family=sources,
                    repository_root=self.repository,
                )
        finally:
            self.repository.unlink()
            repository_backup.rename(self.repository)

    def test_terminal_sweep_catches_early_source_mutated_during_later_hash(
        self,
    ) -> None:
        combined, audits, sources = self._global_material()
        early = self.repository / "scripts/run_paper_corpus_quota.py"
        trigger = (self.repository / "uv.lock").resolve()
        original_artifact = runner._artifact
        mutated = False

        def artifact_and_mutate(
            path: Path,
            *,
            expected_sha256: object | None = None,
        ) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, expected_sha256=expected_sha256)
            if observed.path == trigger and not mutated:
                early.write_bytes(early.read_bytes() + b"changed during later hash")
                mutated = True
            return observed

        with (
            patch.object(runner, "_artifact", side_effect=artifact_and_mutate),
            self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"),
        ):
            runner._authenticate_release_generation_identity(
                combined=combined,
                audits=audits,
                audit_sources_by_family=sources,
                repository_root=self.repository,
            )
        self.assertTrue(mutated)

    def test_outer_parent_sweep_catches_late_source_map_mutation(self) -> None:
        combined, audits, sources = self._global_material()
        target = Path(str(sources["tanaka"][-1]["trajectory_map_path"]))
        target.write_bytes(target.read_bytes() + b"changed after family validation")
        with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
            runner._terminal_reauthenticate_release_parents(
                combined=combined,
                audits=audits,
                audit_sources_by_family=sources,
            )


if __name__ == "__main__":
    unittest.main()
