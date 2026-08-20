from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import build_paper_corpus_view as view_builder
from scripts import build_paper_corpus_document_handoff as handoff
from scripts import diagnose_final_jonswap_order_convergence as order_diagnostic
from scripts import build_parameterized_corpus_case_figure as figure_producer
from scripts import render_paper_corpus_worst_cases as renderer_producer
from PIL import Image
from scripts import run_paper_corpus_postcompletion as runner
from solver.gen_data.pipeline.archive import BatchPaths
from solver.gen_data.pipeline.production import SplitId


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _implementation_record(
    repository_root: Path,
    *,
    schema: str,
    files_and_roles: Sequence[tuple[str, str]],
    semantic_relationship: dict[str, object],
) -> dict[str, object]:
    files = [
        {
            "path": relative,
            "role": role,
            "bytes": (repository_root / relative).stat().st_size,
            "sha256": runner._sha256(repository_root / relative),
        }
        for relative, role in files_and_roles
    ]
    payload: dict[str, object] = {
        "schema": schema,
        "files": files,
        "semantic_relationship": dict(semantic_relationship),
    }
    return {**payload, "fingerprint": runner._canonical_sha256(payload)}


def _training_implementation_record(
    repository_root: Path | None = None,
) -> dict[str, object]:
    return _implementation_record(
        repository_root or runner.ROOT,
        schema=runner.TRAINING_IMPLEMENTATION_SCHEMA,
        files_and_roles=runner.TRAINING_IMPLEMENTATION_FILES,
        semantic_relationship=dict(runner.TRAINING_IMPLEMENTATION_RELATIONSHIP),
    )


def _synthetic_generation_bindings(
    _artifacts: dict[str, runner.Artifact],
    *,
    repository_root: Path = runner.ROOT,
) -> tuple[runner.DependencyBinding, dict[str, runner.GenerationSourceBinding]]:
    if repository_root != runner.ROOT:
        current_paths = set().union(
            *(
                runner.EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                - runner.HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                for family in runner.FAMILY_ORDER
            )
        )
        for relative in sorted(current_paths):
            path = repository_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source:{relative}\n", encoding="utf-8")
        for relative in runner.DEPENDENCY_FILE_PATHS:
            path = repository_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"dependency:{relative}\n", encoding="utf-8")
    dependency_files = {
        relative: runner._artifact(repository_root / relative)
        for relative in runner.DEPENDENCY_FILE_PATHS
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
    historical_digests = {
        "stokes": {
            relative: str(spec["sha256"])
            for relative, spec in runner.STOKES_HISTORICAL_SOURCE_SNAPSHOTS.items()
        },
        "benjamin_feir": {
            relative: str(spec["sha256"])
            for relative, spec in runner.BF_HISTORICAL_SOURCE_SNAPSHOTS.items()
        },
        "tanaka": {},
        "jonswap_tma": {},
    }
    source_maps = {
        family: {
            relative: (
                historical_digests[family][relative]
                if relative in historical_digests[family]
                else runner._sha256(repository_root / relative)
            )
            for relative in sorted(
                runner.EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
            )
        }
        for family in runner.FAMILY_ORDER
    }
    generation_sources = {
        family: runner.GenerationSourceBinding(
            source_sha256=source_maps[family],
            fingerprint=runner._canonical_sha256(source_maps[family]),
            current_sources={
                relative: runner._artifact(repository_root / relative)
                for relative in sorted(
                    runner.EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                    - runner.HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
                )
            },
            historical_source_paths=tuple(
                sorted(runner.HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family])
            ),
            chunk_count=runner.SOURCE_COUNT_BY_FAMILY[family],
        )
        for family in runner.FAMILY_ORDER
    }
    return dependency, generation_sources


def _extrema(count: int, minimum: float, maximum: float) -> dict[str, object]:
    return {"count": count, "minimum": minimum, "maximum": maximum}


class ModuleEntryPointTests(unittest.TestCase):
    def test_adoption_gate_identity_is_fixed_to_the_repository_artifact(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            handoff.JONSWAP_ADOPTION_GATE_PATH,
            repository_root / "outputs/jonswap_relative_band_fresh_gate_20260804/"
            "jonswap_relative_band_fresh_gate.summary.json",
        )
        self.assertEqual(
            handoff.JONSWAP_ADOPTION_GATE_SHA256,
            "4228660e0db7fbf83caa05596d5409a93e183d59454453ded4b3aeee2670a9d8",
        )

    def test_jonswap_population_conditioning_semantics_are_exact(self) -> None:
        self.assertEqual(
            handoff.JONSWAP_POPULATION_CONDITIONING,
            {
                "proposal_law_applies_to": "attempted_specifications",
                "released_case_law": (
                    "proposal_conditioned_on_complete_case_acceptance_within_"
                    "preassigned_cell"
                ),
                "cell_marginals": "accepted_quota",
                "posthoc_parameter_gate": False,
            },
        )

    def test_release_population_conditioning_is_exact_for_all_families(self) -> None:
        self.assertEqual(
            handoff.RELEASE_POPULATION_CONDITIONING,
            {
                "proposal_law_applies_to": "attempted_specifications",
                "released_case_law": (
                    "proposal_conditioned_on_complete_case_acceptance_within_"
                    "preassigned_cell"
                ),
                "cell_marginals": "accepted_quota",
                "posthoc_parameter_gate": False,
                "family_acceptance_events": {
                    "stokes": (
                        "declared_static_support_finite_state_positive_water_column_"
                        "and_finite_target"
                    ),
                    "tanaka": (
                        "declared_construction_support_then_complete_autonomous_"
                        "trajectory_acceptance"
                    ),
                    "benjamin_feir": (
                        "declared_initial_state_validity_then_complete_autonomous_"
                        "trajectory_acceptance"
                    ),
                    "jonswap_tma": (
                        "declared_initial_construction_and_graph_validity_then_"
                        "nonlinear_adjustment_and_complete_autonomous_trajectory_"
                        "acceptance"
                    ),
                },
            },
        )

    def test_module_help_runs_from_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_paper_corpus_document_handoff",
                "--help",
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Completed SHA-keyed postcompletion directory", completed.stdout)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)


class DocumentHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        self.adoption_gate = self.inputs / "jonswap_adoption_gate.json"
        _write_json(
            self.adoption_gate,
            {
                "schema": handoff.JONSWAP_ADOPTION_GATE_SCHEMA,
                "status": "pass",
                "passed": True,
                "predeclared_gate": handoff.JONSWAP_ADOPTION_GATE_CRITERION,
                "observed": {
                    **handoff.JONSWAP_ADOPTION_GATE_COUNTS,
                    "rejection_rate": 8 / 548,
                    "rejection_reasons": {"INCOMPLETE_TRAJECTORY": 8},
                },
                "streams": [{"stream_id": 941}, {"stream_id": 942}],
                "source": {"auditor_sha256": "a" * 64},
            },
        )
        self.gate_path_patch = patch.object(
            handoff, "JONSWAP_ADOPTION_GATE_PATH", self.adoption_gate
        )
        self.gate_sha_patch = patch.object(
            handoff,
            "JONSWAP_ADOPTION_GATE_SHA256",
            runner._sha256(self.adoption_gate),
        )
        self.gate_path_patch.start()
        self.gate_sha_patch.start()
        self.chunks = self._chunks()
        self.combined_root = self.inputs / "combined"
        final_view_root = self.combined_root / "c16384_v01024_t01024"
        final_view_root.mkdir(parents=True)
        self.dataset_contract = {
            "schema": "synthetic_paper_corpus_dataset_contract_v1",
            "delivered_grid": {"length": 2.0 * math.pi, "nx": 1_024},
        }
        self.dataset_contract_fingerprint = runner._canonical_sha256(
            self.dataset_contract
        )
        self.semantic_plan = self._semantic_plan(
            namespace="semantic_sources",
            case_id_base=1_000,
        )
        self.final_semantic_plan = self._semantic_plan(
            namespace="final_semantic_sources",
            case_id_base=2_000,
        )
        self.shard_files = tuple(
            chunk.batches[0].shard for chunk in self.final_semantic_plan.chunks
        )
        self.shard_rows = tuple(
            view_builder.ROWS_PER_ACCEPTED_CASE[chunk.family]
            for chunk in self.final_semantic_plan.chunks
        )
        self.training_rows = sum(self.shard_rows)
        self.training_rows_patch = patch.object(
            runner,
            "_expected_training_rows",
            return_value=self.training_rows,
        )
        self.training_rows_patch.start()
        self.cumulative_views = {
            training_cases: self._write_combined_view(training_cases)
            for training_cases in handoff.CUMULATIVE_TRAINING_CASES
        }
        self.preflight_patch = patch.object(
            handoff.combined_view_builder,
            "preflight",
            side_effect=self._semantic_preflight,
        )
        self.preflight = self.preflight_patch.start()
        final_view = self.cumulative_views[16_384]
        self.summary = final_view.summary.path
        self.manifest = final_view.manifest.path
        self.map = final_view.trajectory_map.path
        self.audits = self._audits()
        audit_artifacts = {
            name: runner._artifact(path) for name, path in self.audits.items()
        }
        dependency, generation_sources = _synthetic_generation_bindings(audit_artifacts)
        self.release = runner.ReleaseBinding(
            combined=final_view,
            audits=audit_artifacts,
            dependency=dependency,
            generation_sources=generation_sources,
        )
        self.root = self.base / self.release.combined.summary.sha256
        self.root.mkdir()
        self.identity_path = self.root / "release_identity.json"
        _write_json(self.identity_path, self.release.identity_record())
        self._renderer()
        self._order()
        self._training()
        self._figure()
        self.status_path = self.root / "status.json"
        self._status()

    def tearDown(self) -> None:
        self.training_rows_patch.stop()
        self.preflight_patch.stop()
        self.gate_sha_patch.stop()
        self.gate_path_patch.stop()
        self.temporary.cleanup()

    def _view_chunks(self, training_cases: int) -> tuple[dict[str, object], ...]:
        return tuple(
            dict(chunk)
            for chunk in self.chunks
            if chunk["split"] != "train"
            or int(chunk["accepted_after"]) <= training_cases
        )

    def _semantic_plan(
        self,
        *,
        namespace: str,
        case_id_base: int,
    ) -> view_builder.CombinedCorpusPlan:
        """Build a tiny real source plan for canonical manifest/map validation."""

        chunks: list[view_builder.CompletedChunk] = []
        for index, family in enumerate(view_builder.FAMILY_ORDER):
            root = self.inputs / "source_00" / namespace / family
            paths = BatchPaths.under(
                root,
                family=family,
                split=SplitId.TRAIN.value,
                batch_id=0,
            )
            for path in (paths.proposal, paths.shard, paths.result):
                path.parent.mkdir(parents=True, exist_ok=True)
            fingerprint = f"{index + 101:064x}"
            family_id = int(view_builder.FAMILY_IDS[family])
            revision_id = runner.REVISION_BY_FAMILY[family]
            case_id = case_id_base + index
            np.savez(
                paths.proposal,
                family_id=np.asarray(family_id, dtype=np.int16),
                revision_id=np.asarray(revision_id, dtype=np.int16),
                split_id=np.asarray(0, dtype=np.uint8),
                batch_id=np.asarray(0, dtype=np.int64),
                case_id=np.asarray([case_id], dtype=np.int64),
                cell_id=np.asarray([index], dtype=np.int32),
                config_fingerprint=np.asarray(fingerprint),
            )
            row_count = view_builder.ROWS_PER_ACCEPTED_CASE[family]
            np.savez(
                paths.shard,
                case_local_index=np.zeros(row_count, dtype=np.int32),
                frame_index=np.arange(row_count, dtype=np.int32),
            )
            _write_json(
                paths.result,
                {
                    "schema": "paper_corpus_batch_result_v1",
                    "config_fingerprint": fingerprint,
                    "proposal_sha256": runner._sha256(paths.proposal),
                    "shard_sha256": runner._sha256(paths.shard),
                    "cases": [
                        {
                            "case_id": case_id,
                            "accepted": True,
                            "required_bits": 0,
                            "evaluated_bits": 0,
                            "failed_bits": 0,
                            "first_row": 0,
                            "row_count": row_count,
                        }
                    ],
                },
            )
            source_summary = root / "source.summary.json"
            _write_json(source_summary, {"family": family})
            chunks.append(
                view_builder.CompletedChunk(
                    summary_path=source_summary,
                    summary_sha256=runner._sha256(source_summary),
                    root=root,
                    family=family,
                    revision_id=revision_id,
                    split=SplitId.TRAIN,
                    stream_id=index,
                    accepted_before=0,
                    accepted_count=1,
                    accepted_after=1,
                    attempted_count=1,
                    fingerprint=fingerprint,
                    dependency_fingerprint="d" * 64,
                    execution_fingerprint=f"{index + 201:064x}",
                    generation_compatibility_id=None,
                    source_fingerprint=f"{index + 301:064x}",
                    source_sha256={},
                    execution_platform="cpu",
                    batches=(paths,),
                )
            )
        fingerprints = tuple(sorted(chunk.fingerprint for chunk in chunks))
        return view_builder.CombinedCorpusPlan(
            chunks=tuple(chunks),
            splits=(SplitId.TRAIN,),
            accepted_cases_per_family_by_split={"train": 1},
            attempted_cases_by_split={"train": len(chunks)},
            attempted_cases=len(chunks),
            accepted_cases=len(chunks),
            expected_rows=sum(view_builder.ROWS_PER_ACCEPTED_CASE.values()),
            fingerprints=fingerprints,
        )

    def _expected_preflight(self, training_cases: int) -> dict[str, object]:
        tag = f"{training_cases:05d}"
        chunks = self._view_chunks(training_cases)
        cases_by_split = {
            "train": training_cases,
            "validation": 1_024,
            "test": 1_024,
        }
        rows_by_split_and_family = handoff._cumulative_rows_by_split_and_family(
            training_cases
        )
        attempted_by_split = {
            split: sum(
                int(chunk["attempted_count"])
                for chunk in chunks
                if chunk["split"] == split
            )
            for split in runner.SPLIT_ORDER
        }
        return {
            "schema": runner.COMBINED_PREFLIGHT_SCHEMA,
            "mode": "dry_run",
            "no_view_written": True,
            "output_root": str(
                (self.combined_root / f"c{tag}_v01024_t01024").resolve()
            ),
            "view_name": f"paper_corpus_all_splits_c{tag}",
            "splits": list(runner.SPLIT_ORDER),
            "accepted_cases_per_family_by_split": cases_by_split,
            "accepted_cases_total": len(runner.FAMILY_ORDER)
            * sum(cases_by_split.values()),
            "attempted_cases_by_split": attempted_by_split,
            "attempted_cases_total": sum(attempted_by_split.values()),
            "expected_rows": sum(
                sum(family_rows.values())
                for family_rows in rows_by_split_and_family.values()
            ),
            "expected_rows_by_split_and_family": rows_by_split_and_family,
            "configuration_fingerprints": sorted(
                str(chunk["configuration_fingerprint"]) for chunk in chunks
            ),
            "chunks": list(chunks),
        }

    def _semantic_preflight(
        self,
        summary_paths: Sequence[Path],
        *,
        output_root: Path,
        name: str | None,
    ) -> tuple[
        view_builder.CombinedCorpusPlan,
        Path,
        str,
        dict[str, object],
    ]:
        assert name is not None
        training_cases = int(name.rsplit("c", maxsplit=1)[1])
        plan = (
            self.final_semantic_plan if training_cases == 16_384 else self.semantic_plan
        )
        expected_paths = tuple(
            Path(str(chunk["summary_path"])).resolve()
            for chunk in self._view_chunks(training_cases)
        )
        if tuple(Path(path).resolve() for path in summary_paths) != expected_paths:
            raise RuntimeError(
                "canonical preflight received the wrong source summaries"
            )
        return (
            plan,
            Path(output_root).resolve(),
            name,
            self._expected_preflight(training_cases),
        )

    def _write_semantic_view(
        self,
        *,
        root: Path,
        name: str,
        plan: view_builder.CombinedCorpusPlan,
    ) -> tuple[Path, Path, dict[str, object]]:
        trajectory_map = root / f"{name}.trajectory_map.npz"
        row_counts = np.asarray(
            [
                view_builder.ROWS_PER_ACCEPTED_CASE[chunk.family]
                for chunk in plan.chunks
            ],
            dtype=np.int32,
        )
        case_ids = []
        for chunk in plan.chunks:
            with np.load(chunk.batches[0].proposal, allow_pickle=False) as proposal:
                case_ids.append(int(np.asarray(proposal["case_id"])[0]))
        first_rows = np.cumsum(row_counts, dtype=np.int64) - row_counts
        row_owners = np.repeat(
            np.arange(len(row_counts), dtype=np.int32),
            row_counts,
        )
        frame_index = np.concatenate(
            tuple(np.arange(count, dtype=np.int32) for count in map(int, row_counts))
        )
        shard_index = np.repeat(
            np.arange(len(row_counts), dtype=np.int32),
            row_counts,
        )
        shard_row = np.concatenate(
            tuple(np.arange(count, dtype=np.int64) for count in map(int, row_counts))
        )
        np.savez(
            trajectory_map,
            schema_version=np.asarray(2, dtype=np.int16),
            trajectory_index=row_owners,
            frame_index=frame_index,
            shard_index=shard_index,
            shard_row=shard_row,
            trajectory_family_id=np.asarray(
                [int(view_builder.FAMILY_IDS[chunk.family]) for chunk in plan.chunks],
                dtype=np.int16,
            ),
            trajectory_revision_id=np.asarray(
                [chunk.revision_id for chunk in plan.chunks],
                dtype=np.int16,
            ),
            trajectory_split_id=np.zeros(len(row_counts), dtype=np.uint8),
            trajectory_case_id=np.asarray(case_ids, dtype=np.int64),
            trajectory_cell_id=np.arange(len(row_counts), dtype=np.int32),
            trajectory_accepted=np.ones(len(row_counts), dtype=np.bool_),
            trajectory_required_bits=np.zeros(len(row_counts), dtype=np.uint32),
            trajectory_evaluated_bits=np.zeros(len(row_counts), dtype=np.uint32),
            trajectory_failed_bits=np.zeros(len(row_counts), dtype=np.uint32),
            trajectory_first_row=first_rows,
            trajectory_row_count=row_counts,
        )
        batches, shards = self._semantic_source_records(root, plan=plan)
        manifest = root / f"{name}.dataset.json"
        _write_json(
            manifest,
            {
                "schema_version": 2,
                "configuration_fingerprint": None,
                "configuration_fingerprints": list(plan.fingerprints),
                "trajectory_map_npz": trajectory_map.name,
                "trajectory_map_sha256": runner._sha256(trajectory_map),
                "requires_trajectory_map": True,
                "n_rows": plan.expected_rows,
                "n_trajectories": plan.attempted_cases,
                "n_accepted_trajectories": plan.accepted_cases,
                "n_accepted_rows": plan.expected_rows,
                "grid": {"length": 2.0 * math.pi, "nx": 1_024},
                "split_counts": {
                    "train": {"attempted": 4, "accepted": 4},
                    "validation": {"attempted": 0, "accepted": 0},
                    "test": {"attempted": 0, "accepted": 0},
                },
                "dataset_contract": self.dataset_contract,
                "dataset_contract_fingerprint": (self.dataset_contract_fingerprint),
                "dataset_batches": batches,
                "dataset_shards": shards,
            },
        )
        paths = view_builder.DatasetViewPaths(
            manifest=manifest,
            trajectory_map=trajectory_map,
        )
        return (
            manifest,
            trajectory_map,
            view_builder._validate_view(paths, plan=plan),
        )

    def _semantic_source_records(
        self,
        root: Path,
        *,
        plan: view_builder.CombinedCorpusPlan,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return canonical records for the tiny physical source transactions."""

        batches: list[dict[str, object]] = []
        shards: list[dict[str, object]] = []
        for index, chunk in enumerate(plan.chunks):
            paths = chunk.batches[0]
            row_count = view_builder.ROWS_PER_ACCEPTED_CASE[chunk.family]
            batches.append(
                {
                    "proposal_path": os.path.relpath(paths.proposal, start=root),
                    "proposal_sha256": runner._sha256(paths.proposal),
                    "result_path": os.path.relpath(paths.result, start=root),
                    "result_sha256": runner._sha256(paths.result),
                    "shard_index": index,
                    "configuration_fingerprint": chunk.fingerprint,
                    "family_id": int(view_builder.FAMILY_IDS[chunk.family]),
                    "revision_id": chunk.revision_id,
                    "split_id": 0,
                    "batch_id": 0,
                    "n_attempted_trajectories": 1,
                    "n_accepted_trajectories": 1,
                    "n_rows": row_count,
                }
            )
            shards.append(
                {
                    "path": os.path.relpath(paths.shard, start=root),
                    "sha256": runner._sha256(paths.shard),
                    "n_rows": row_count,
                    "batch_index": index,
                    "configuration_fingerprint": chunk.fingerprint,
                }
            )
        return batches, shards

    def _write_combined_view(
        self,
        training_cases: int,
    ) -> runner.CombinedBinding:
        tag = f"{training_cases:05d}"
        name = f"paper_corpus_all_splits_c{tag}"
        root = self.combined_root / f"c{tag}_v01024_t01024"
        root.mkdir(parents=True, exist_ok=True)
        chunks = self._view_chunks(training_cases)
        preflight = self._expected_preflight(training_cases)
        plan = (
            self.final_semantic_plan if training_cases == 16_384 else self.semantic_plan
        )
        manifest, trajectory_map, view_record = self._write_semantic_view(
            root=root,
            name=name,
            plan=plan,
        )
        summary = root / f"{name}.summary.json"
        _write_json(
            summary,
            {
                "schema": runner.COMBINED_SUMMARY_SCHEMA,
                "status": "complete",
                "preflight": preflight,
                "dataset_view": view_record,
            },
        )
        return runner.CombinedBinding(
            summary=runner._artifact(summary),
            manifest=runner._artifact(manifest),
            trajectory_map=runner._artifact(trajectory_map),
            chunks=chunks,
        )

    def _chunks(self) -> tuple[dict[str, object], ...]:
        chunks: list[dict[str, object]] = []
        source_index = 0
        for family in runner.FAMILY_ORDER:
            for split, before, count, stream in runner.expected_layout(family):
                source_root = self.inputs / f"source_{source_index:02d}"
                source_root.mkdir()
                stem = f"paper_corpus_{family}_{split}"
                source = source_root / f"{stem}.summary.json"
                source_manifest = source_root / f"{stem}.dataset.json"
                source_map = source_root / f"{stem}.trajectory_map.npz"
                proposal = source_root / "transactions/batch_000000.proposal.npz"
                result = source_root / "transactions/batch_000000.result.json"
                shard = source_root / "shards/shard_000000.npz"
                proposal.parent.mkdir(parents=True)
                shard.parent.mkdir(parents=True)
                rejected = 0
                if before == 0 and family == "benjamin_feir":
                    rejected = {"train": 20, "validation": 0, "test": 3}[split]
                elif (
                    before == 0
                    and family in {"tanaka", "jonswap_tma"}
                    and split == "train"
                ):
                    rejected = 1
                attempted = count + rejected
                cell_count = handoff.CATEGORY_COUNT_BY_FAMILY[family]
                quotient, remainder = divmod(count, cell_count)
                by_cell = {}
                cell_ids = (
                    sorted(handoff.STOKES_CATEGORY_IDS)
                    if family == "stokes"
                    else [f"{family}_cell_{index:02d}" for index in range(cell_count)]
                )
                selected_category = runner.CENTRAL_VALIDATION_CATEGORY[family]
                if selected_category not in cell_ids:
                    cell_ids[0] = selected_category
                for index, cell_id in enumerate(cell_ids):
                    accepted = quotient + (index < remainder)
                    cell_rejected = rejected if index == 0 else 0
                    by_cell[cell_id] = {
                        "target_accepted": accepted,
                        "attempted": accepted + cell_rejected,
                        "accepted": accepted,
                        "rejected": cell_rejected,
                    }
                tiny_case_count = runner.RENDERER_TOP_COUNT
                case_ids = np.arange(
                    10_000 * source_index,
                    10_000 * source_index + tiny_case_count,
                    dtype=np.int64,
                )
                rows_per_case = runner.ROWS_PER_CASE[family]
                retained = tiny_case_count * rows_per_case
                frame_index = np.tile(
                    np.arange(rows_per_case, dtype=np.int32), tiny_case_count
                )
                trajectory_index = np.repeat(
                    np.arange(tiny_case_count, dtype=np.int32), rows_per_case
                )
                np.savez(
                    proposal,
                    batch_id=np.asarray(0, dtype=np.int64),
                    case_id=case_ids,
                    case_spec_json=np.asarray(
                        [
                            json.dumps(
                                {"cell_id": selected_category, "depth": 1.0},
                                sort_keys=True,
                            )
                            for _ in range(tiny_case_count)
                        ]
                    ),
                )
                shard_fields: dict[str, np.ndarray] = {
                    "depth": np.ones(retained, dtype=np.float64),
                    "time": frame_index.astype(np.float64),
                    "eta": np.zeros((retained, 1), dtype=np.float32),
                    "xi": np.zeros((retained, 1), dtype=np.float32),
                    "gxi": np.zeros((retained, 1), dtype=np.float32),
                }
                if family == "jonswap_tma":
                    gxi = np.zeros((retained, 1_024), dtype=np.float32)
                    x = 2.0 * np.pi * np.arange(1_024) / 1_024
                    for case_position in range(tiny_case_count):
                        amplitude = 0.1 * (case_position + 1 + source_index)
                        profile = np.sin(x) + amplitude * np.sin(100.0 * x)
                        rows = slice(
                            case_position * rows_per_case,
                            (case_position + 1) * rows_per_case,
                        )
                        gxi[rows] = profile.astype(np.float32)
                    shard_fields["gxi"] = gxi
                np.savez(shard, **shard_fields)
                _write_json(
                    result,
                    {
                        "proposal_sha256": runner._sha256(proposal),
                        "shard_sha256": runner._sha256(shard),
                        "cases": [
                            {
                                "accepted": True,
                                "case_id": int(case_id),
                                "first_row": index * rows_per_case,
                                "row_count": rows_per_case,
                            }
                            for index, case_id in enumerate(case_ids)
                        ],
                    },
                )
                np.savez(
                    source_map,
                    trajectory_accepted=np.ones(tiny_case_count, dtype=np.bool_),
                    trajectory_case_id=case_ids,
                    trajectory_cell_id=np.zeros(tiny_case_count, dtype=np.int32),
                    trajectory_first_row=(
                        np.arange(tiny_case_count, dtype=np.int64) * rows_per_case
                    ),
                    trajectory_row_count=np.full(
                        tiny_case_count, rows_per_case, dtype=np.int32
                    ),
                    trajectory_index=trajectory_index,
                    shard_index=np.zeros(retained, dtype=np.int32),
                    shard_row=np.arange(retained, dtype=np.int64),
                    frame_index=frame_index,
                )
                _write_json(
                    source_manifest,
                    {
                        "schema_version": 2,
                        "dataset_batches": [
                            {
                                "batch_id": 0,
                                "shard_index": 0,
                                "proposal_path": str(proposal.relative_to(source_root)),
                                "proposal_sha256": runner._sha256(proposal),
                                "result_path": str(result.relative_to(source_root)),
                                "result_sha256": runner._sha256(result),
                            }
                        ],
                        "dataset_shards": [
                            {
                                "batch_index": 0,
                                "path": str(shard.relative_to(source_root)),
                                "sha256": runner._sha256(shard),
                                "n_rows": retained,
                            }
                        ],
                    },
                )
                numerical = {"length": 2.0 * math.pi, "gravity": 9.81}
                configuration: dict[str, object] = {
                    "source_sha256": {
                        path: runner._sha256(runner.ROOT / path)
                        for path in (
                            "solver/gen_data/pipeline/reference.py",
                            "solver/solvers/dno_series_jax.py",
                        )
                    }
                }
                if family == "stokes":
                    configuration["contract"] = numerical
                else:
                    configuration["trajectory_execution"] = {"numerical": numerical}
                _write_json(
                    source,
                    {
                        "schema": handoff.QUOTA_SUMMARY_SCHEMA,
                        "status": "complete",
                        "output_root": str(source_root),
                        "run_spec": {
                            "family_name": family,
                            "split_id": split,
                            "revision_id": runner.REVISION_BY_FAMILY[family],
                            "cell_codes": {selected_category: 0},
                            "configuration": configuration,
                        },
                        "dataset_view": {
                            "grid": {"length": 2.0 * math.pi, "nx": 1_024},
                            "manifest": {
                                "path": source_manifest.name,
                                "bytes": source_manifest.stat().st_size,
                                "sha256": runner._sha256(source_manifest),
                            },
                            "trajectory_map": {
                                "path": source_map.name,
                                "bytes": source_map.stat().st_size,
                                "sha256": runner._sha256(source_map),
                            },
                        },
                        "counts": {
                            "accepted": count,
                            "attempted": attempted,
                            "rejected": rejected,
                            "rejection_reasons": {},
                            "by_cell": by_cell,
                        },
                    },
                )
                chunks.append(
                    {
                        "family": family,
                        "revision_id": runner.REVISION_BY_FAMILY[family],
                        "split": split,
                        "accepted_before": before,
                        "accepted_count": count,
                        "accepted_after": before + count,
                        "attempted_count": attempted,
                        "stream_id": stream,
                        "configuration_fingerprint": f"{source_index + 1:064x}",
                        "label": f"{split}_{before:05d}_{before + count:05d}",
                        "summary_path": str(source),
                        "summary_sha256": runner._sha256(source),
                        "manifest_path": str(source_manifest),
                        "manifest_sha256": runner._sha256(source_manifest),
                        "trajectory_map_path": str(source_map),
                        "trajectory_map_sha256": runner._sha256(source_map),
                        "retained_rows": count * runner.ROWS_PER_CASE[family],
                    }
                )
                source_index += 1
        self.assertEqual(source_index, runner.FINAL_SOURCE_COUNT)
        return tuple(chunks)

    def _family_category_counts(
        self,
        family: str,
        split: str | None = None,
    ) -> dict[str, dict[str, int | float]]:
        totals: dict[str, dict[str, int]] = {}
        for chunk in self.chunks:
            if chunk["family"] != family or (
                split is not None and chunk["split"] != split
            ):
                continue
            source = runner._strict_json(Path(str(chunk["summary_path"])))
            by_cell = source["counts"]["by_cell"]
            for name, record in by_cell.items():
                target = totals.setdefault(
                    name,
                    {
                        "target_accepted": 0,
                        "attempted": 0,
                        "accepted": 0,
                        "rejected": 0,
                    },
                )
                for field in target:
                    target[field] += int(record[field])
        return {
            name: {
                **record,
                "rejection_rate": record["rejected"] / record["attempted"],
            }
            for name, record in sorted(totals.items())
        }

    def _audits(self) -> dict[str, Path]:
        paths = {
            name: self.inputs / f"{name}.json"
            for name in (*handoff.PRIMARY_AUDITS, "stokes_legacy")
        }
        _write_json(
            paths["stokes_legacy"],
            {
                "schema": handoff.STOKES_LEGACY_AUDIT_SCHEMA,
                "status": "pass",
                "finite_float_values_checked": 56_659_968,
                "transaction_artifact_bytes_checked": 400_263_499,
            },
        )
        split_counts = dict(runner.CASES_PER_FAMILY_BY_SPLIT)
        for family in handoff.PRIMARY_AUDITS:
            attempted = sum(
                int(chunk["attempted_count"])
                for chunk in self.chunks
                if chunk["family"] == family
            )
            counts = {
                "accepted": 18_432,
                "attempted": attempted,
                "rejected": attempted - 18_432,
                "retained_rows": runner.EXPECTED_ROWS_BY_FAMILY[family],
                "accepted_by_split": split_counts,
            }
            record: dict[str, object] = {
                "schema": handoff.FAMILY_AUDIT_SCHEMAS[family],
                "status": "pass",
                "chunks": [
                    {
                        name: chunk[name]
                        for name in (
                            "label",
                            "split",
                            "accepted_count",
                            "attempted_count",
                            "retained_rows",
                            "summary_path",
                            "summary_sha256",
                            "manifest_path",
                            "manifest_sha256",
                            "trajectory_map_path",
                            "trajectory_map_sha256",
                        )
                    }
                    for chunk in self.chunks
                    if chunk["family"] == family
                ],
            }
            if family == "stokes":
                record["counts"] = counts
                legacy = runner._artifact(paths["stokes_legacy"])
                record["legacy_numerical_audit"] = {
                    "path": str(legacy.path),
                    "sha256": legacy.sha256,
                    "schema": handoff.STOKES_LEGACY_AUDIT_SCHEMA,
                    "status": "pass",
                    "finite_float_values_checked": 56_659_968,
                    "transaction_artifact_bytes_checked": 400_263_499,
                }
            else:
                record.update(counts)
                if family in {"tanaka", "benjamin_feir"}:
                    record["accepted_by_split_and_cell"] = {
                        split: {
                            name: int(cell["accepted"])
                            for name, cell in self._family_category_counts(
                                family,
                                split,
                            ).items()
                        }
                        for split in runner.SPLIT_ORDER
                    }
                support_counts = {
                    metric: (
                        27_648
                        if family == "tanaka"
                        and metric in {"crest_alpha", "physical_crest_amplitude"}
                        else attempted
                    )
                    for metric in handoff.SUPPORT_EXTREMA_METRICS[family]
                }
                record["support"] = {
                    "extrema": {
                        metric: _extrema(count, 0.01, 2.0)
                        for metric, count in support_counts.items()
                    }
                }
                if family == "tanaka":
                    record["requested_crests_checked"] = 27_648
                    record["numerical_extrema"] = {
                        "accepted_maximum_stage_residual": _extrema(
                            18_432, 1.0e-12, 9.9e-9
                        )
                    }
                    record["checks"] = {
                        "achieved_per_crest_amplitudes_persisted": False
                    }
                else:
                    metrics = handoff.HEALTH_EXTREMA_METRICS[family]
                    record["numerical_extrema"] = {
                        "accepted": {
                            metric: _extrema(18_432, 1.0e-12, 9.9e-4)
                            for metric in metrics["accepted"]
                        },
                        "all_finite_attempted": {
                            metric: _extrema(attempted, 1.0e-12, 1.1e-3)
                            for metric in metrics["all_finite_attempted"]
                        },
                    }
                if family == "jonswap_tma":
                    candidates = []
                    for chunk in self.chunks:
                        if chunk["family"] != "jonswap_tma":
                            continue
                        summary = runner._strict_json(Path(str(chunk["summary_path"])))
                        manifest_path = Path(str(chunk["manifest_path"]))
                        manifest = runner._strict_json(manifest_path)
                        shard_record = manifest["dataset_shards"][0]
                        shard_path = manifest_path.parent / shard_record["path"]
                        with np.load(shard_path, allow_pickle=False) as shard:
                            gxi = np.asarray(shard["gxi"], dtype=np.float64)
                        fractions = order_diagnostic.high_band_fraction(gxi)
                        per_case = fractions.reshape(runner.RENDERER_TOP_COUNT, 16)
                        for local_index, values in enumerate(per_case):
                            frame = int(np.argmax(values))
                            candidates.append(
                                (
                                    float(values[frame]),
                                    chunk,
                                    summary,
                                    local_index,
                                    frame,
                                    shard_path,
                                )
                            )
                    maximum = max(candidates, key=lambda item: item[0])
                    value, chunk, summary, local_index, frame, shard_path = maximum
                    case_id = 10_000 * self.chunks.index(chunk) + local_index
                    record["population_conditioning"] = {
                        **handoff.JONSWAP_POPULATION_CONDITIONING,
                        "cells": self._family_category_counts(family),
                    }
                    record["identity"] = {
                        "source_sha256_fingerprint": "source",
                        "dependency_environment_fingerprint": "dependency",
                        "execution_record_fingerprint": "execution",
                        "current_support_source_sha256": {"source.py": "hash"},
                    }
                    record["trajectory_quality_diagnostics"] = {
                        "metric_values_affect_acceptance_or_audit_status": False,
                        "accepted_trajectories_checked": 18_432,
                        "metrics": {
                            "gxi_high_band_energy_fraction_maximum": {
                                "maximum": {
                                    "value": value,
                                    "identity": {
                                        "case_id": case_id,
                                        "chunk_label": chunk["label"],
                                        "split": chunk["split"],
                                        "batch_id": 0,
                                        "local_index": local_index,
                                        "frame_index": frame,
                                        "cell_id": runner.CENTRAL_VALIDATION_CATEGORY[
                                            "jonswap_tma"
                                        ],
                                        "shard_path": str(
                                            shard_path.relative_to(
                                                Path(str(chunk["summary_path"])).parent
                                            )
                                        ),
                                    },
                                }
                            }
                        },
                    }
            _write_json(paths[family], record)
        return paths

    def _renderer(self) -> None:
        output = self.root / "worst_cases"
        output.mkdir()
        names = {
            f"{family}_worst_{ranking}.{suffix}"
            for family in runner.FAMILY_ORDER
            for ranking in runner.RENDERER_RANKINGS
            for suffix in ("gif", "pdf", "png")
        } | {f"all_families_worst_overview.{suffix}" for suffix in ("pdf", "png")}
        artifacts: dict[str, object] = {}
        gif_templates: dict[tuple[int, int], bytes] = {}
        for name in names:
            path = output / name
            if path.suffix == ".gif":
                family = name.split("_worst_", maxsplit=1)[0]
                frames = min(
                    runner.ROWS_PER_CASE[family],
                    renderer_producer.GIF_MAXIMUM_FRAMES,
                )
                duration = 250 if frames <= 20 else 80
                key = frames, duration
                template = gif_templates.get(key)
                if template is None:
                    images = [
                        Image.new("P", renderer_producer.GIF_DIMENSIONS, 0)
                        for _ in range(frames)
                    ]
                    for index, image in enumerate(images):
                        image.putpixel((index, 0), 1)
                    template_path = output / f".template_{frames}.gif"
                    images[0].save(
                        template_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=duration,
                        loop=0,
                        optimize=True,
                    )
                    template = template_path.read_bytes()
                    template_path.unlink()
                    gif_templates[key] = template
                path.write_bytes(template)
            else:
                path.write_bytes(name.encode("ascii"))
            artifacts[name] = runner._artifact(path).record()
        source_records = runner._release_source_records(self.release)
        proofs = runner._renderer_source_proofs(self.release)
        families: dict[str, object] = {}
        for family in runner.FAMILY_ORDER:
            candidates = []
            for source_index, (source, proof) in enumerate(
                zip(source_records, proofs, strict=True)
            ):
                if source["family"] != family:
                    continue
                trajectories = proof["trajectories"]
                for trajectory in trajectories:
                    high_fraction = 0.5
                    high_frame = 0
                    if family == "jonswap_tma":
                        shard_path = proof["shards"][trajectory.shard_index][0]
                        with np.load(shard_path, allow_pickle=False) as shard:
                            rows = slice(
                                trajectory.first_shard_row,
                                trajectory.first_shard_row + trajectory.row_count,
                            )
                            fractions = order_diagnostic.high_band_fraction(
                                np.asarray(shard["gxi"][rows], dtype=np.float64)
                            )
                        high_frame = int(np.argmax(fractions))
                        high_fraction = float(fractions[high_frame])
                    candidates.append(
                        (high_fraction, source_index, trajectory, high_frame)
                    )
            candidates.sort(key=lambda item: (-item[0], item[1], item[2].case_id))
            selected = candidates[: runner.RENDERER_TOP_COUNT]
            cases = []
            for rank, (
                high_fraction,
                source_index,
                trajectory,
                high_frame,
            ) in enumerate(selected):
                score = runner.RENDERER_TOP_COUNT - rank
                cases.append(
                    {
                        "source_index": source_index,
                        "accepted_index": trajectory.accepted_index,
                        "trajectory_index": trajectory.trajectory_index,
                        "family": family,
                        "split": source_records[source_index]["split"],
                        "case_id": trajectory.case_id,
                        "category": trajectory.category,
                        "shard_index": trajectory.shard_index,
                        "first_shard_row": trajectory.first_shard_row,
                        "row_count": trajectory.row_count,
                        "depth": 1.0,
                        "all_frames_finite": True,
                        "constant_depth": True,
                        "ordered_time": True,
                        "minimum_water_column": 0.5,
                        "minimum_water_fraction": 0.5,
                        "maximum_eta_slope": float(score),
                        "maximum_eta_slope_frame": 0,
                        "maximum_gxi_high_band_fraction": high_fraction,
                        "maximum_gxi_high_band_fraction_frame": high_frame,
                        "maximum_thresholded_gxi_sign_changes": score,
                        "maximum_thresholded_gxi_sign_changes_frame": 0,
                        "maximum_relative_stored_band_quadratic_energy_drift": (
                            float(score) * 1.0e-6
                        ),
                        "maximum_relative_stored_band_quadratic_energy_drift_frame": 0,
                        "source_root": source_records[source_index]["root"],
                        "combined_empirical_rank": 1.0 - rank * 0.1,
                    }
                )
            quantile_maxima = {
                metric: max(float(case[field]) for case in cases)
                for metric, field in {
                    "eta_slope": "maximum_eta_slope",
                    "gxi_high_band": "maximum_gxi_high_band_fraction",
                    "gxi_sign_changes": "maximum_thresholded_gxi_sign_changes",
                    "stored_band_quadratic_energy_drift": (
                        "maximum_relative_stored_band_quadratic_energy_drift"
                    ),
                    "minimum_water_fraction": "minimum_water_fraction",
                }.items()
            }
            families[family] = {
                "accepted_cases": 18_432,
                "retained_rows": runner.EXPECTED_ROWS_BY_FAMILY[family],
                "splits": runner.EXPECTED_SPLIT_COUNTS,
                "quantiles": {
                    metric: {
                        quantile: (
                            (
                                quantile_maxima[metric]
                                if metric == "minimum_water_fraction"
                                else 0.0
                            )
                            if quantile == "q000"
                            else quantile_maxima[metric]
                        )
                        for quantile in handoff.RENDERER_QUANTILES
                    }
                    for metric in handoff.RENDERER_QUANTILE_METRICS
                },
                "rankings": {
                    ranking: [dict(case) for case in cases]
                    for ranking in runner.RENDERER_RANKINGS
                },
            }
        animations: dict[str, object] = {}
        for family in runner.FAMILY_ORDER:
            for ranking in runner.RENDERER_RANKINGS:
                case = families[family]["rankings"][ranking][0]
                proof = proofs[case["source_index"]]
                shard_path = proof["shards"][case["shard_index"]][0]
                first = case["first_shard_row"]
                count = case["row_count"]
                with np.load(shard_path, allow_pickle=False) as shard:
                    fields = {
                        name: np.asarray(
                            shard[name][first : first + count], dtype=np.float64
                        )
                        for name in (*renderer_producer.FIELD_NAMES, "time")
                    }
                animation = renderer_producer.animation_record(
                    renderer_producer.LoadedTrajectory(
                        eta=fields["eta"],
                        xi=fields["xi"],
                        gxi=fields["gxi"],
                        depth=np.full(count, case["depth"]),
                        time=fields["time"],
                    )
                )
                name = f"{family}_worst_{ranking}.gif"
                animations[name] = {
                    "family": family,
                    "ranking": ranking,
                    "rank": 1,
                    **{
                        field: case[field]
                        for field in (
                            "source_index",
                            "accepted_index",
                            "trajectory_index",
                            "case_id",
                            "category",
                            "split",
                        )
                    },
                    **animation,
                }
        _write_json(
            output / "summary.json",
            {
                "schema": runner.RENDERER_SCHEMA,
                "status": "complete",
                "interpretation": renderer_producer.INTERPRETATION,
                "source_binding": {
                    "mode": "combined_summary",
                    "combined_summary_path": str(self.summary),
                    "combined_summary_sha256": runner._sha256(self.summary),
                    "expected_sources": runner.FINAL_SOURCE_COUNT,
                    "expected_accepted_cases": runner.FINAL_ACCEPTED_CASES,
                    "expected_retained_rows": runner.FINAL_RETAINED_ROWS,
                },
                "parameters": dict(renderer_producer.PARAMETERS),
                "definitions": dict(renderer_producer.DEFINITIONS),
                "population": {
                    "sources": runner.FINAL_SOURCE_COUNT,
                    "accepted_cases": runner.FINAL_ACCEPTED_CASES,
                    "retained_rows": runner.FINAL_RETAINED_ROWS,
                },
                "sources": source_records,
                "families": families,
                "renderer_implementation": (
                    renderer_producer.renderer_implementation_record()
                ),
                "animations": animations,
                "artifacts": artifacts,
            },
        )

    def _order_case(self, renderer_case: dict[str, object]) -> dict[str, object]:
        source_root = Path(str(renderer_case["source_root"]))
        chunk = next(
            value
            for value in self.chunks
            if Path(str(value["summary_path"])).parent == source_root
        )
        manifest = runner._strict_json(Path(str(chunk["manifest_path"])))
        batch = manifest["dataset_batches"][0]
        shard = manifest["dataset_shards"][0]
        proposal_path = source_root / batch["proposal_path"]
        result_path = source_root / batch["result_path"]
        shard_path = source_root / shard["path"]
        zeros = [0.0] * 16
        ones = [1.0] * 16
        return {
            "rank_metric": renderer_case["maximum_gxi_high_band_fraction"],
            "identity": {
                "case_id": renderer_case["case_id"],
                "chunk_label": chunk["label"],
                "split": renderer_case["split"],
                "batch_id": 0,
                "local_index": renderer_case["trajectory_index"],
                "frame_index": renderer_case["maximum_gxi_high_band_fraction_frame"],
                "cell_id": renderer_case["category"],
                "shard_path": str(shard_path.relative_to(source_root)),
            },
            "trajectory_index_within_chunk": renderer_case["trajectory_index"],
            "depth": 1.0,
            "sources": {
                "summary_path": chunk["summary_path"],
                "summary_sha256": chunk["summary_sha256"],
                "manifest_path": chunk["manifest_path"],
                "manifest_sha256": chunk["manifest_sha256"],
                "trajectory_map_path": chunk["trajectory_map_path"],
                "trajectory_map_sha256": chunk["trajectory_map_sha256"],
                "proposal_path": str(proposal_path),
                "proposal_sha256": runner._sha256(proposal_path),
                "result_path": str(result_path),
                "result_sha256": runner._sha256(result_path),
                "shard_path": str(shard_path),
                "shard_sha256": runner._sha256(shard_path),
            },
            "orders": {
                str(order): {
                    "projected_l2_norm_by_frame": ones,
                    "projected_high_band_energy_fraction_by_frame": zeros,
                    "maximum_projected_high_band_energy_fraction": 0.0,
                    "maximum_projected_l2_norm": 1.0,
                }
                for order in range(4, 9)
            },
            "successive_order_changes": [
                {
                    "from_order": order,
                    "to_order": order + 1,
                    "projected_relative_l2_difference_by_frame": zeros,
                    "maximum_projected_relative_l2_difference": 0.0,
                    "raw_preprojection_relative_l2_difference_by_frame": zeros,
                    "maximum_raw_preprojection_relative_l2_difference": 0.0,
                    "absolute_high_band_fraction_change_by_frame": zeros,
                    "maximum_absolute_high_band_fraction_change": 0.0,
                }
                for order in range(4, 8)
            ],
            "stored_float32_vs_recomputed_order6": {
                "relative_l2_difference_by_frame": [1.0e-6] * 16,
                "maximum_relative_l2_difference": 1.0e-6,
            },
            "order6_to_order8": {
                "projected_relative_l2_difference_by_frame": [1.0e-5] * 16,
                "maximum_projected_relative_l2_difference": 1.0e-5,
            },
        }

    def _order(self) -> None:
        renderer = self.root / "worst_cases/summary.json"
        renderer_record = runner._strict_json(renderer)
        top_cases = renderer_record["families"]["jonswap_tma"]["rankings"][
            "gxi_high_band"
        ][:3]
        audit = runner._artifact(self.audits["jonswap_tma"])
        audit_record = runner._strict_json(audit.path)
        audit_identity = audit_record["identity"]
        implementation = order_diagnostic.diagnostic_implementation_record()
        target_sources = {
            path: self.release.generation_sources["jonswap_tma"].source_sha256[path]
            for path in (
                "solver/gen_data/pipeline/reference.py",
                "solver/solvers/dno_series_jax.py",
            )
        }
        chunk_bindings = []
        for chunk in audit_record["chunks"]:
            manifest = runner._strict_json(Path(str(chunk["manifest_path"])))
            chunk_bindings.append(
                {
                    "label": chunk["label"],
                    "split": chunk["split"],
                    "accepted_scanned": chunk["accepted_count"],
                    "rows_scanned": chunk["retained_rows"],
                    "shards_scanned": len(manifest["dataset_shards"]),
                    "summary_path": chunk["summary_path"],
                    "summary_sha256": chunk["summary_sha256"],
                    "manifest_path": chunk["manifest_path"],
                    "manifest_sha256": chunk["manifest_sha256"],
                    "trajectory_map_path": chunk["trajectory_map_path"],
                    "trajectory_map_sha256": chunk["trajectory_map_sha256"],
                    "generation_target_source_sha256": target_sources,
                }
            )
        audit_maximum = audit_record["trajectory_quality_diagnostics"]["metrics"][
            "gxi_high_band_energy_fraction_maximum"
        ]["maximum"]["value"]
        independent_maximum = top_cases[0]["maximum_gxi_high_band_fraction"]
        _write_json(
            self.root / "jonswap_order_convergence.json",
            {
                "schema": runner.ORDER_SCHEMA,
                "status": "pass",
                "diagnostic_only": True,
                "affects_corpus_acceptance": False,
                "affects_corpus_release": False,
                "completion_audit": {
                    "path": str(audit.path),
                    "sha256": audit.sha256,
                    "schema": order_diagnostic.AUDIT_SCHEMA,
                    "accepted": 18_432,
                    "retained_rows": runner.EXPECTED_ROWS_BY_FAMILY["jonswap_tma"],
                },
                "selection": {
                    "definition": order_diagnostic.SELECTION_DEFINITION,
                    "independently_rescanned": True,
                    "completion_audit_global_maximum_reproduced": True,
                    "completion_audit_global_maximum_value": audit_maximum,
                    "independent_global_maximum_value": independent_maximum,
                    "maximum_value_absolute_difference": abs(
                        independent_maximum - audit_maximum
                    ),
                    "maximum_value_comparison_tolerance": 1.0e-7,
                    "accepted_trajectories_scanned": 18_432,
                    "stored_rows_scanned": runner.EXPECTED_ROWS_BY_FAMILY[
                        "jonswap_tma"
                    ],
                    "shards_scanned": sum(
                        binding["shards_scanned"] for binding in chunk_bindings
                    ),
                    "selected_count": 3,
                    "final_renderer_cross_check": {
                        "path": str(renderer),
                        "sha256": runner._sha256(renderer),
                        "schema": runner.RENDERER_SCHEMA,
                        "status": "complete",
                        "combined_summary_path": str(self.summary),
                        "combined_summary_sha256": runner._sha256(self.summary),
                        "exact_final_population_verified": True,
                        "independent_top_selection_reproduced": True,
                    },
                },
                "numerical_definition": {
                    "execution_platform": "cpu",
                    "dtype": "float64",
                    "nx": 1_024,
                    "length": 2.0 * runner.math.pi,
                    "input_and_output_projection": "sharp |k| <= 128",
                    "zero_output_mean": True,
                    "pad_factor": 8,
                    "cumulative_orders": [4, 5, 6, 7, 8],
                    "relative_l2_definition": (order_diagnostic.RELATIVE_L2_DEFINITION),
                },
                "identity": {
                    "generation_source_sha256_fingerprint": audit_identity[
                        "source_sha256_fingerprint"
                    ],
                    "generation_dependency_environment_fingerprint": audit_identity[
                        "dependency_environment_fingerprint"
                    ],
                    "generation_execution_record_fingerprint": audit_identity[
                        "execution_record_fingerprint"
                    ],
                    "generation_support_source_sha256": audit_identity[
                        "current_support_source_sha256"
                    ],
                    "diagnostic_source_sha256": {
                        item["path"]: item["sha256"] for item in implementation["files"]
                    },
                },
                "diagnostic_implementation": implementation,
                "chunk_bindings": chunk_bindings,
                "cases": [self._order_case(case) for case in top_cases],
                "interpretation_scope": order_diagnostic.INTERPRETATION_SCOPE,
                "timing_seconds": 1.0,
                "invocation_started_at": "2026-08-09T00:00:00+00:00",
                "invocation_finished_at": "2026-08-09T00:00:01+00:00",
            },
        )

    def _training(self) -> None:
        shards = [
            {
                "index": index,
                "rows": self.shard_rows[index],
                "selected_training_rows": (
                    self.shard_rows[index]
                    if self.final_semantic_plan.chunks[index].split is SplitId.TRAIN
                    else 0
                ),
                **runner._artifact(path).record(),
            }
            for index, path in enumerate(self.shard_files)
        ]
        identity_payload = [
            {key: item[key] for key in ("index", "path", "bytes", "sha256", "rows")}
            for item in shards
        ]
        selection = {"count": self.training_rows, "sha256": "a" * 64}
        stats = {
            "num_examples": self.training_rows,
            "storage_num_examples": runner.FINAL_RETAINED_ROWS,
            "index_selection": selection,
            "feature_min": [-1.0, -2.0],
            "feature_max": [1.0, 2.0],
            "feature_absmax": [1.0, 2.0],
            "target_min": -3.0,
            "target_max": 3.0,
            "target_absmax": 3.0,
            "depth_min": 0.1,
            "depth_max": 10.0,
            "log_depth_min": -2.3,
            "log_depth_max": 2.3,
            "domain_length": 2.0 * runner.math.pi,
        }
        sampler = {
            "schema": "paper_corpus_schema_v2_sampler_contract_v1",
            "seed": 0,
            "test_rows_in_train_or_validation": 0,
            "train_epoch_0": {
                "count": 65_536,
                "sha256": "b" * 64,
                "one_row_per_case": True,
                "per_family_count": {family: 16_384 for family in runner.FAMILY_ORDER},
            },
            "fixed_validation_epoch_0": {
                "count": 4_096,
                "sha256": "c" * 64,
                "one_row_per_case": True,
                "per_family_count": {family: 1_024 for family in runner.FAMILY_ORDER},
            },
        }
        _write_json(
            self.root / "training_handoff_audit.json",
            {
                "schema": runner.TRAINING_SCHEMA,
                "status": "complete",
                "purpose": "training_handoff_normalization_audit",
                "audit_only": True,
                "acceptance_or_generation_effect": False,
                "training_implementation": _training_implementation_record(),
                "combined_summary": self.release.combined.summary.record(),
                "execution": {"gpu_used": False, "training_split_seed": 0},
                "dataset_view": {
                    "manifest": self.release.combined.manifest.record(),
                    "trajectory_map": self.release.combined.trajectory_map.record(),
                    "dataset_contract_fingerprint": (self.dataset_contract_fingerprint),
                    "shard_count": len(shards),
                    "shard_rows": sum(int(item["rows"]) for item in shards),
                    "shard_bytes": sum(int(item["bytes"]) for item in shards),
                    "shard_identity_fingerprint": runner._canonical_sha256(
                        identity_payload
                    ),
                    "shards": shards,
                },
                "final_contract": {
                    "expected": {
                        "families": list(runner.FAMILY_ORDER),
                        "revision_by_family": runner.REVISION_BY_FAMILY,
                        "source_count_by_family": runner.SOURCE_COUNT_BY_FAMILY,
                        "source_count": runner.FINAL_SOURCE_COUNT,
                        "accepted_cases_per_family_by_split": (
                            runner.CASES_PER_FAMILY_BY_SPLIT
                        ),
                        "rows_per_accepted_case": runner.ROWS_PER_CASE,
                        "accepted_cases": runner.FINAL_ACCEPTED_CASES,
                        "accepted_rows": runner.FINAL_RETAINED_ROWS,
                    },
                    "observed": {
                        "accepted_cases": runner.FINAL_ACCEPTED_CASES,
                        "accepted_rows": runner.FINAL_RETAINED_ROWS,
                    },
                },
                "training_normalization": {
                    "accepted_training_row_selection": {"seed": 0, **selection},
                    "stats": stats,
                    "stats_fingerprint": runner._canonical_sha256(stats),
                    "schema_v2_sampler": sampler,
                },
            },
        )

    def _figure(self) -> None:
        stem = self.root / "family_case_examples"
        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        pdf.write_bytes(b"pdf")
        png.write_bytes(b"png")
        cases = runner._expected_figure_cases(self.release)
        release_contract = {
            "source_count": runner.FINAL_SOURCE_COUNT,
            "source_count_by_family": dict(
                figure_producer.EXPECTED_SOURCE_COUNT_BY_FAMILY
            ),
            "accepted_cases": runner.FINAL_ACCEPTED_CASES,
            "retained_rows": runner.FINAL_RETAINED_ROWS,
            "family_revisions": dict(figure_producer.EXPECTED_REVISIONS),
            "ordered_cell_ids": dict(figure_producer.EXPECTED_CELL_IDS),
            "central_validation_categories": dict(
                figure_producer.CENTRAL_VALIDATION_CATEGORIES
            ),
            "source_intervals": [
                {
                    "family": item.family,
                    "split": item.split.value,
                    "stream_id": item.stream_id,
                    "accepted_before": item.accepted_before,
                    "accepted_after": item.accepted_after,
                }
                for item in figure_producer._expected_chunk_layout()
            ],
        }
        _write_json(
            stem.with_suffix(".json"),
            {
                "schema": runner.FIGURE_SCHEMA,
                "status": "complete",
                "description": figure_producer.SIDECAR_DESCRIPTION,
                "combined_summary": self.release.combined.summary.record(),
                "combined_view": {
                    "manifest": self.release.combined.manifest.record(),
                    "trajectory_map": self.release.combined.trajectory_map.record(),
                },
                "release_contract": release_contract,
                "publication": {
                    "commit_marker": str(stem.with_suffix(".json")),
                    "rule": figure_producer.PUBLICATION_RULE,
                },
                "cases": cases,
                "figure_implementation": (
                    figure_producer.figure_implementation_record()
                ),
                "artifacts": {
                    "pdf": runner._artifact(pdf).record(),
                    "png": runner._artifact(png).record(),
                },
            },
        )

    def _stage_artifacts(self) -> dict[str, dict[str, object]]:
        return {
            "final_renderer": dict(
                runner.validate_renderer_output(self.root / "worst_cases", self.release)
            ),
            "jonswap_order_diagnostic": dict(
                runner.validate_order_output(
                    self.root / "jonswap_order_convergence.json",
                    release=self.release,
                    renderer_summary=self.root / "worst_cases/summary.json",
                )
            ),
            "training_handoff_audit": dict(
                runner.validate_training_output(
                    self.root / "training_handoff_audit.json",
                    self.release.combined.summary,
                )
            ),
            "family_case_figure": dict(
                runner.validate_figure_output(
                    self.root / "family_case_examples", self.release
                )
            ),
        }

    def _status(self) -> None:
        identity = self.release.identity_record()
        _write_json(
            self.status_path,
            {
                "schema": runner.RUNNER_SCHEMA,
                "status": "complete",
                "release_artifacts_mutated": False,
                "release_artifacts_mutated_by_runner": False,
                "release_evidence_unchanged_after_stages": True,
                "diagnostics_affect_corpus_release": False,
                "diagnostic_failures": [],
                "combined_summary": self.release.combined.summary.record(),
                "release_evidence": identity,
                "release_identity": runner._artifact(self.identity_path).record(),
                "release_reauthentication": {
                    "status": "pass",
                    "verified_unchanged": True,
                    "combined_summary_sha256": self.release.combined.summary.sha256,
                    "release_identity_fingerprint": runner._canonical_sha256(identity),
                },
                "stages": {
                    name: {
                        "status": "complete",
                        "attempts": 1,
                        "artifacts": artifacts,
                    }
                    for name, artifacts in self._stage_artifacts().items()
                },
            },
        )

    def _replace_audit(self, name: str, record: object) -> None:
        _write_json(self.audits[name], record)
        audits = dict(self.release.audits)
        audits[name] = runner._artifact(self.audits[name])
        self.release = runner.ReleaseBinding(
            combined=self.release.combined,
            audits=audits,
            dependency=self.release.dependency,
            generation_sources=self.release.generation_sources,
        )
        _write_json(self.identity_path, self.release.identity_record())
        if name == "jonswap_tma":
            order_path = self.root / "jonswap_order_convergence.json"
            order = runner._strict_json(order_path)
            audit = self.release.audits[name]
            order["completion_audit"]["path"] = str(audit.path)
            order["completion_audit"]["sha256"] = audit.sha256
            _write_json(order_path, order)
        self._status()

    def _authenticate(
        self, combined: Path, audits: runner.AuditPaths
    ) -> runner.ReleaseBinding:
        self.assertEqual(combined.resolve(), self.summary.resolve())
        expected_paths = {
            "stokes": audits.stokes_binding,
            "stokes_legacy": audits.stokes_legacy,
            "benjamin_feir": audits.benjamin_feir,
            "jonswap_tma": audits.jonswap_tma,
            "tanaka": audits.tanaka,
        }
        for name, artifact in self.release.audits.items():
            self.assertEqual(expected_paths[name].resolve(), artifact.path)
            observed = runner._artifact(artifact.path)
            if observed != artifact:
                raise RuntimeError(f"synthetic audit changed: {name}")
        return self.release

    def _rebind_cumulative_view(self, training_cases: int) -> None:
        binding = self.cumulative_views[training_cases]
        manifest = runner._strict_json(binding.manifest.path)
        manifest["trajectory_map_sha256"] = runner._sha256(binding.trajectory_map.path)
        _write_json(binding.manifest.path, manifest)
        summary = runner._strict_json(binding.summary.path)
        summary["dataset_view"]["manifest"] = runner._artifact(
            binding.manifest.path
        ).record()
        summary["dataset_view"]["trajectory_map"] = runner._artifact(
            binding.trajectory_map.path
        ).record()
        _write_json(binding.summary.path, summary)

    def _current_cumulative_binding(
        self,
        training_cases: int,
    ) -> runner.CombinedBinding:
        binding = self.cumulative_views[training_cases]
        return runner.CombinedBinding(
            summary=runner._artifact(binding.summary.path),
            manifest=runner._artifact(binding.manifest.path),
            trajectory_map=runner._artifact(binding.trajectory_map.path),
            chunks=binding.chunks,
        )

    def _assert_after_second_fingerprint_failure(
        self,
        mutation: Callable[[], None],
        *,
        exception: type[Exception],
        pattern: str,
    ) -> None:
        original_fingerprint = handoff._authentication_fingerprint
        calls = 0

        def mutate_after_fingerprint(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> str:
            nonlocal calls
            fingerprint = original_fingerprint(authenticated)
            calls += 1
            if calls == 2:
                mutation()
            return fingerprint

        with patch.object(
            handoff,
            "_authentication_fingerprint",
            side_effect=mutate_after_fingerprint,
        ):
            with self.assertRaisesRegex(exception, pattern):
                handoff.build_document_handoff(
                    self.root,
                    release_authenticator=self._authenticate,
                )
        self.assertEqual(calls, 2)
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def _bind_fake_training_repository(self, repository: Path) -> None:
        file_sets = (
            runner.TRAINING_IMPLEMENTATION_FILES,
            renderer_producer.IMPLEMENTATION_FILES,
            order_diagnostic.IMPLEMENTATION_FILES,
            figure_producer.IMPLEMENTATION_FILES,
        )
        for relative, _ in sorted(set().union(*map(set, file_sets))):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
        training_path = self.root / "training_handoff_audit.json"
        training_record = runner._strict_json(training_path)
        training_record["training_implementation"] = _training_implementation_record(
            repository
        )
        _write_json(training_path, training_record)
        renderer_path = self.root / "worst_cases/summary.json"
        renderer_record = runner._strict_json(renderer_path)
        renderer_record["renderer_implementation"] = _implementation_record(
            repository,
            schema=renderer_producer.IMPLEMENTATION_SCHEMA,
            files_and_roles=renderer_producer.IMPLEMENTATION_FILES,
            semantic_relationship=dict(renderer_producer.IMPLEMENTATION_RELATIONSHIP),
        )
        _write_json(renderer_path, renderer_record)
        order_path = self.root / "jonswap_order_convergence.json"
        order_record = runner._strict_json(order_path)
        order_implementation = _implementation_record(
            repository,
            schema=order_diagnostic.IMPLEMENTATION_SCHEMA,
            files_and_roles=order_diagnostic.IMPLEMENTATION_FILES,
            semantic_relationship=dict(order_diagnostic.IMPLEMENTATION_RELATIONSHIP),
        )
        order_record["diagnostic_implementation"] = order_implementation
        order_record["identity"]["diagnostic_source_sha256"] = {
            item["path"]: item["sha256"] for item in order_implementation["files"]
        }
        order_record["selection"]["final_renderer_cross_check"]["sha256"] = (
            runner._sha256(renderer_path)
        )
        _write_json(order_path, order_record)
        figure_path = self.root / "family_case_examples.json"
        figure_record = runner._strict_json(figure_path)
        figure_record["figure_implementation"] = _implementation_record(
            repository,
            schema=figure_producer.IMPLEMENTATION_SCHEMA,
            files_and_roles=figure_producer.IMPLEMENTATION_FILES,
            semantic_relationship=dict(figure_producer.IMPLEMENTATION_RELATIONSHIP),
        )
        _write_json(figure_path, figure_record)
        with patch.object(runner, "ROOT", repository):
            self._status()

    def _bind_fake_generation_repository(self, repository: Path) -> None:
        dependency, generation_sources = _synthetic_generation_bindings(
            dict(self.release.audits),
            repository_root=repository,
        )
        self.release = runner.ReleaseBinding(
            combined=self.release.combined,
            audits=self.release.audits,
            dependency=dependency,
            generation_sources=generation_sources,
            historical_source_snapshots=(self.release.historical_source_snapshots),
        )
        _write_json(self.identity_path, self.release.identity_record())
        self._order()
        self._status()

    def _mutate_cumulative_map(
        self,
        training_cases: int,
        mutation: str,
    ) -> None:
        path = self.cumulative_views[training_cases].trajectory_map.path
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
        if mutation == "trajectory_case_id":
            arrays[mutation][0] += 10
        elif mutation == "trajectory_cell_id":
            arrays[mutation][0] += 10
        elif mutation == "quality_bits":
            arrays["trajectory_required_bits"][0] = 1
            arrays["trajectory_evaluated_bits"][0] = 1
        elif mutation == "frame_index":
            arrays[mutation][0] += 1
        elif mutation == "trajectory_first_row":
            arrays[mutation][0] += 1
        else:
            raise ValueError(f"unknown synthetic map mutation {mutation}")
        np.savez(path, **arrays)
        self._rebind_cumulative_view(training_cases)

    def test_success_and_idempotent_identical_output(self) -> None:
        output = handoff.build_document_handoff(
            self.root, release_authenticator=self._authenticate
        )
        before = output.read_bytes()
        second = handoff.build_document_handoff(
            self.status_path, release_authenticator=self._authenticate
        )
        self.assertEqual(second, output)
        self.assertEqual(output.read_bytes(), before)
        record = runner._strict_json(output)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["release"]["totals"]["accepted_cases"], 73_728)
        self.assertEqual(record["release"]["totals"]["attempted_cases"], 73_753)
        self.assertEqual(
            record["release"]["totals"]["rejection_rate"],
            25 / 73_753,
        )
        family_counts = record["release"]["family_counts"]
        self.assertEqual(
            record["release"]["population_conditioning"],
            handoff.RELEASE_POPULATION_CONDITIONING,
        )

        self.assertEqual(family_counts["stokes"]["rejection_rate"], 0.0)
        self.assertTrue(
            all(
                split["rejection_rate"] == 0.0
                for split in family_counts["stokes"]["splits"].values()
            )
        )
        benjamin_feir = family_counts["benjamin_feir"]
        self.assertEqual(benjamin_feir["rejection_rate"], 23 / 18_455)
        self.assertEqual(
            benjamin_feir["splits"]["train"]["rejection_rate"],
            20 / 16_404,
        )
        self.assertEqual(
            benjamin_feir["splits"]["validation"]["rejection_rate"],
            0.0,
        )
        self.assertEqual(
            benjamin_feir["splits"]["test"]["rejection_rate"],
            3 / 1_027,
        )
        for family in ("tanaka", "jonswap_tma"):
            counts = family_counts[family]
            self.assertEqual(counts["rejection_rate"], 1 / 18_433)
            self.assertEqual(
                counts["splits"]["train"]["rejection_rate"],
                1 / 16_385,
            )
            self.assertEqual(counts["splits"]["validation"]["rejection_rate"], 0.0)
            self.assertEqual(counts["splits"]["test"]["rejection_rate"], 0.0)
        for family, category_count in handoff.CATEGORY_COUNT_BY_FAMILY.items():
            counts = family_counts[family]
            self.assertEqual(len(counts["categories"]), category_count)
            for split in runner.SPLIT_ORDER:
                categories = counts["splits"][split]["categories"]
                self.assertEqual(len(categories), category_count)
                self.assertEqual(
                    sum(cell["accepted"] for cell in categories.values()),
                    counts["splits"][split]["accepted"],
                )
                self.assertEqual(
                    sum(cell["attempted"] for cell in categories.values()),
                    counts["splits"][split]["attempted"],
                )
        conditioning = family_counts["jonswap_tma"]["population_conditioning"]
        for name, expected in handoff.JONSWAP_POPULATION_CONDITIONING.items():
            self.assertEqual(conditioning[name], expected)
        self.assertEqual(
            conditioning["cells"],
            family_counts["jonswap_tma"]["categories"],
        )
        views = record["release"]["cumulative_combined_views"]
        self.assertEqual(set(views), {"c02048", "c04096", "c08192", "c16384"})
        self.assertEqual(
            [views[key]["source_count"] for key in sorted(views)],
            [12, 16, 20, 26],
        )
        for training_cases in handoff.CUMULATIVE_TRAINING_CASES:
            view = views[f"c{training_cases:05d}"]
            self.assertEqual(view["training_cases_per_family"], training_cases)
            self.assertEqual(
                view["accepted_cases_per_family_by_split"],
                {"train": training_cases, "validation": 1_024, "test": 1_024},
            )
            self.assertEqual(
                view["attempted_cases"],
                sum(view["attempted_cases_by_split"].values()),
            )
            self.assertEqual(
                view["rejected_cases"],
                view["attempted_cases"] - view["accepted_cases"],
            )
        self.assertTrue(
            all(
                math.isfinite(counts["rejection_rate"])
                for counts in (*family_counts.values(), record["release"]["totals"])
            )
        )
        adoption = record["validated_manuscript_evidence"][
            "jonswap_tma_revision4_prebulk_adoption_gate"
        ]
        self.assertEqual(
            adoption["decision_scope"],
            "revision_4_constructor_prebulk_adoption_only",
        )
        self.assertFalse(adoption["applies_to_final_bulk_release"])
        self.assertEqual(
            adoption["observed"],
            {
                "accepted": 540,
                "attempted": 548,
                "rejected": 8,
                "rejection_rate": 8 / 548,
            },
        )
        self.assertEqual(adoption["artifact"]["path"], str(self.adoption_gate))
        self.assertEqual(
            adoption["artifact"]["sha256"], runner._sha256(self.adoption_gate)
        )
        extrema = record["release"]["family_numerical_extrema"]
        self.assertFalse(extrema["stokes"]["reported"])
        self.assertNotIn("support_parameter_extrema", extrema["stokes"])
        self.assertEqual(
            set(extrema["benjamin_feir"]["support_parameter_extrema"]),
            set(handoff.SUPPORT_EXTREMA_METRICS["benjamin_feir"]),
        )
        self.assertEqual(
            extrema["tanaka"]["numerical_health_extrema"][
                "accepted_maximum_stage_residual"
            ]["count"],
            18_432,
        )
        self.assertFalse(extrema["tanaka"]["achieved_per_crest_amplitudes_persisted"])
        self.assertNotIn("achieved_per_crest_amplitudes", extrema["tanaka"])
        self.assertEqual(
            record["validated_stage_outputs"]["training_handoff"]["measured_storage"][
                "combined_shard_bytes"
            ],
            sum(path.stat().st_size for path in self.shard_files),
        )
        self.assertEqual(
            record["validated_stage_outputs"]["training_handoff"][
                "training_implementation"
            ],
            _training_implementation_record(),
        )
        selected = record["validated_stage_outputs"]["family_case_figure"][
            "selected_cases"
        ]
        self.assertEqual(
            [case["case_id"] for case in selected],
            [case["case_id"] for case in runner._expected_figure_cases(self.release)],
        )
        stage_outputs = record["validated_stage_outputs"]
        self.assertEqual(
            stage_outputs["worst_case_renderer"]["authenticated_proof"],
            runner._strict_json(self.root / "worst_cases/summary.json"),
        )
        self.assertEqual(
            len(stage_outputs["worst_case_renderer"]["authenticated_proof"]["sources"]),
            26,
        )
        self.assertEqual(
            stage_outputs["jonswap_order_diagnostic"]["authenticated_proof"],
            runner._strict_json(self.root / "jonswap_order_convergence.json"),
        )
        figure_proof = stage_outputs["family_case_figure"]["authenticated_proof"]
        self.assertEqual(
            figure_proof,
            runner._strict_json(self.root / "family_case_examples.json"),
        )
        self.assertEqual(
            set(figure_proof["cases"][0]),
            {
                "family",
                "revision_id",
                "split",
                "category",
                "case_id",
                "time",
                "depth",
                "gravity",
                "domain_length",
                "stored_nx",
                "selection",
                "owned_row",
                "dimensionless_variables",
                "source_summary",
                "source_manifest",
                "source_trajectory_map",
                "selected_shard",
            },
        )

    def test_builder_holds_runner_lock_through_authentication_and_commit(
        self,
    ) -> None:
        original_authenticate = handoff.authenticate_postcompletion
        original_write = handoff._write_immutable_json
        observed: list[str] = []

        def require_lock(label: str) -> None:
            with (self.root / "runner.lock").open("a+b") as contender:
                with self.assertRaises(BlockingIOError):
                    handoff.fcntl.flock(
                        contender.fileno(),
                        handoff.fcntl.LOCK_EX | handoff.fcntl.LOCK_NB,
                    )
            observed.append(label)

        def authenticate(
            source: Path,
            *,
            release_authenticator: handoff.ReleaseAuthenticator,
        ) -> handoff.AuthenticatedPostcompletion:
            require_lock("authenticate")
            return original_authenticate(
                source,
                release_authenticator=release_authenticator,
            )

        def write(path: Path, payload: Mapping[str, object]) -> None:
            require_lock("commit")
            original_write(path, payload)

        with (
            patch.object(
                handoff,
                "authenticate_postcompletion",
                side_effect=authenticate,
            ),
            patch.object(handoff, "_write_immutable_json", side_effect=write),
        ):
            output = handoff.build_document_handoff(
                self.root,
                release_authenticator=self._authenticate,
            )

        self.assertEqual(observed, ["authenticate", "authenticate", "commit"])
        self.assertEqual(output, self.root / handoff.DEFAULT_NAME)

    def test_missing_cumulative_view_fails_closed(self) -> None:
        self.cumulative_views[4_096].summary.path.unlink()

        with self.assertRaises(FileNotFoundError):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_all_views_use_canonical_preflight_and_deep_validation(self) -> None:
        canonical_validate = handoff.combined_view_builder._validate_view
        with patch.object(
            handoff.combined_view_builder,
            "_validate_view",
            wraps=canonical_validate,
        ) as validate:
            handoff.build_document_handoff(
                self.root,
                release_authenticator=self._authenticate,
            )

        self.assertEqual(self.preflight.call_count, 8)
        self.assertEqual(validate.call_count, 8)

    def test_cumulative_view_must_use_exact_source_prefix(self) -> None:
        path = self.cumulative_views[4_096].summary.path
        record = runner._strict_json(path)
        record["preflight"]["chunks"][0]["attempted_count"] += 1
        _write_json(path, record)

        with self.assertRaisesRegex(RuntimeError, "preflight differs from canonical"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_count_preserving_cumulative_map_corruptions_fail_closed(self) -> None:
        binding = self.cumulative_views[4_096]
        originals = {
            path: path.read_bytes()
            for path in (
                binding.summary.path,
                binding.manifest.path,
                binding.trajectory_map.path,
            )
        }
        for mutation in (
            "trajectory_case_id",
            "trajectory_cell_id",
            "quality_bits",
            "frame_index",
            "trajectory_first_row",
        ):
            with self.subTest(mutation=mutation):
                for path, content in originals.items():
                    path.write_bytes(content)
                self._mutate_cumulative_map(4_096, mutation)
                with self.assertRaises(RuntimeError):
                    handoff.build_document_handoff(
                        self.root,
                        release_authenticator=self._authenticate,
                    )
                self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rebound_cumulative_manifest_source_corruptions_fail_closed(self) -> None:
        binding = self.cumulative_views[4_096]
        originals = {
            path: path.read_bytes()
            for path in (binding.summary.path, binding.manifest.path)
        }
        for field, mutation in (
            ("dataset_batches", "reordered"),
            ("dataset_batches", "deleted"),
            ("dataset_shards", "reordered"),
            ("dataset_shards", "deleted"),
        ):
            with self.subTest(field=field, mutation=mutation):
                for path, content in originals.items():
                    path.write_bytes(content)
                manifest = runner._strict_json(binding.manifest.path)
                records = manifest[field]
                if mutation == "reordered":
                    manifest[field] = list(reversed(records))
                else:
                    manifest[field] = records[:-1]
                _write_json(binding.manifest.path, manifest)
                self._rebind_cumulative_view(4_096)
                with self.assertRaisesRegex(RuntimeError, "manifest .* records differ"):
                    handoff.build_document_handoff(
                        self.root,
                        release_authenticator=self._authenticate,
                    )
                self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rehashed_smaller_dataset_contract_must_equal_final(self) -> None:
        binding = self.cumulative_views[4_096]
        manifest = runner._strict_json(binding.manifest.path)
        manifest["dataset_contract"]["coherent_but_wrong"] = True
        manifest["dataset_contract_fingerprint"] = runner._canonical_sha256(
            manifest["dataset_contract"]
        )
        _write_json(binding.manifest.path, manifest)
        summary = runner._strict_json(binding.summary.path)
        summary["dataset_view"]["dataset_contract_fingerprint"] = manifest[
            "dataset_contract_fingerprint"
        ]
        _write_json(binding.summary.path, summary)
        self._rebind_cumulative_view(4_096)

        with self.assertRaisesRegex(
            RuntimeError,
            "dataset contract differs from final release",
        ):
            handoff.build_document_handoff(
                self.root,
                release_authenticator=self._authenticate,
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rehashed_final_map_semantic_corruption_fails_closed(self) -> None:
        self._mutate_cumulative_map(16_384, "trajectory_case_id")
        final = self._current_cumulative_binding(16_384)

        with self.assertRaisesRegex(RuntimeError, "trajectory-map field"):
            handoff._authenticate_cumulative_views(final)

    def test_rehashed_final_manifest_reorder_and_delete_fail_closed(self) -> None:
        binding = self.cumulative_views[16_384]
        originals = {
            path: path.read_bytes()
            for path in (binding.summary.path, binding.manifest.path)
        }
        for field, mutation in (
            ("dataset_batches", "reordered"),
            ("dataset_batches", "deleted"),
            ("dataset_shards", "reordered"),
            ("dataset_shards", "deleted"),
        ):
            with self.subTest(field=field, mutation=mutation):
                for path, content in originals.items():
                    path.write_bytes(content)
                manifest = runner._strict_json(binding.manifest.path)
                records = manifest[field]
                manifest[field] = (
                    list(reversed(records)) if mutation == "reordered" else records[:-1]
                )
                _write_json(binding.manifest.path, manifest)
                self._rebind_cumulative_view(16_384)
                final = self._current_cumulative_binding(16_384)
                with self.assertRaisesRegex(RuntimeError, "manifest .* records differ"):
                    handoff._authenticate_cumulative_views(final)

    def test_rehashed_final_coherent_wrong_dataset_contract_fails_closed(self) -> None:
        binding = self.cumulative_views[16_384]
        manifest = runner._strict_json(binding.manifest.path)
        manifest["dataset_contract"]["coherent_but_wrong"] = True
        manifest["dataset_contract_fingerprint"] = runner._canonical_sha256(
            manifest["dataset_contract"]
        )
        _write_json(binding.manifest.path, manifest)
        summary = runner._strict_json(binding.summary.path)
        summary["dataset_view"]["dataset_contract_fingerprint"] = manifest[
            "dataset_contract_fingerprint"
        ]
        _write_json(binding.summary.path, summary)
        self._rebind_cumulative_view(16_384)
        final = self._current_cumulative_binding(16_384)

        with self.assertRaisesRegex(
            RuntimeError,
            "dataset contract differs from final release",
        ):
            handoff._authenticate_cumulative_views(final)

    def test_cumulative_view_mutation_before_commit_fails_closed(self) -> None:
        artifact = self.cumulative_views[4_096].manifest.path
        original_build_record = handoff.build_record

        def mutate_after_build(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> dict[str, object]:
            record = original_build_record(authenticated)
            artifact.write_bytes(b"changed after initial authentication")
            return record

        with patch.object(handoff, "build_record", side_effect=mutate_after_build):
            with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_cumulative_mutation_after_second_fingerprint_fails_closed(self) -> None:
        artifact = self.cumulative_views[4_096].manifest.path
        original_fingerprint = handoff._authentication_fingerprint
        calls = 0

        def mutate_after_fingerprint(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> str:
            nonlocal calls
            fingerprint = original_fingerprint(authenticated)
            calls += 1
            if calls == 2:
                artifact.write_bytes(b"changed after second fingerprint")
            return fingerprint

        with patch.object(
            handoff,
            "_authentication_fingerprint",
            side_effect=mutate_after_fingerprint,
        ):
            with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertEqual(calls, 2)
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_generation_source_mutation_after_second_fingerprint_fails_closed(
        self,
    ) -> None:
        repository = self.base / "generation_repository_after_fingerprint"
        self._bind_fake_generation_repository(repository)
        target = repository / "solver/gen_data/jonswap_tma.py"
        self._assert_after_second_fingerprint_failure(
            lambda: target.write_text("# changed late\n", encoding="utf-8"),
            exception=RuntimeError,
            pattern="artifact SHA-256 differs",
        )

    def test_source_manifest_mutation_during_terminal_provenance_fails_closed(
        self,
    ) -> None:
        repository = self.base / "generation_repository_terminal_race"
        self._bind_fake_generation_repository(repository)
        target = Path(str(self.chunks[0]["manifest_path"]))
        trigger = min(
            artifact.path
            for binding in self.release.generation_sources.values()
            for artifact in binding.current_sources.values()
        )
        original_artifact = runner._artifact
        original_terminal = runner._terminal_reauthenticate_artifact_groups
        terminal_active = False
        mutated = False

        def artifact_and_mutate(
            path: Path,
            *,
            expected_sha256: object | None = None,
        ) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, expected_sha256=expected_sha256)
            if terminal_active and observed.path == trigger and not mutated:
                target.write_bytes(target.read_bytes() + b"changed during provenance")
                mutated = True
            return observed

        def terminal_with_race(
            groups: dict[str, tuple[Path, dict[str, runner.Artifact]]],
        ) -> None:
            nonlocal terminal_active
            terminal_active = True
            try:
                original_terminal(groups)
            finally:
                terminal_active = False

        with (
            patch.object(runner, "_artifact", side_effect=artifact_and_mutate),
            patch.object(
                runner,
                "_terminal_reauthenticate_artifact_groups",
                side_effect=terminal_with_race,
            ),
            self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"),
        ):
            handoff.build_document_handoff(
                self.root,
                release_authenticator=self._authenticate,
            )
        self.assertTrue(mutated)
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_preserved_figure_shard_mutation_during_provenance_fails_closed(
        self,
    ) -> None:
        repository = self.base / "generation_repository_figure_proof_race"
        self._bind_fake_generation_repository(repository)
        figure = runner._strict_json(self.root / "family_case_examples.json")
        target = Path(figure["cases"][0]["selected_shard"]["path"])
        trigger = min(
            artifact.path
            for binding in self.release.generation_sources.values()
            for artifact in binding.current_sources.values()
        )
        original_artifact = runner._artifact
        original_terminal = runner._terminal_reauthenticate_artifact_groups
        terminal_active = False
        mutated = False

        def artifact_and_mutate(
            path: Path,
            *,
            expected_sha256: object | None = None,
        ) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, expected_sha256=expected_sha256)
            if terminal_active and observed.path == trigger and not mutated:
                target.write_bytes(target.read_bytes() + b"changed during provenance")
                mutated = True
            return observed

        def terminal_with_race(
            groups: dict[str, tuple[Path, dict[str, runner.Artifact]]],
        ) -> None:
            nonlocal terminal_active
            terminal_active = True
            try:
                original_terminal(groups)
            finally:
                terminal_active = False

        with (
            patch.object(runner, "_artifact", side_effect=artifact_and_mutate),
            patch.object(
                runner,
                "_terminal_reauthenticate_artifact_groups",
                side_effect=terminal_with_race,
            ),
            self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"),
        ):
            handoff.build_document_handoff(
                self.root,
                release_authenticator=self._authenticate,
            )
        self.assertTrue(mutated)
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_training_code_mutation_after_second_fingerprint_fails_closed(self) -> None:
        repository = self.base / "training_repository_after_fingerprint"
        self._bind_fake_training_repository(repository)
        target = repository / "train-jax-10m/util.py"

        with (
            patch.object(runner, "ROOT", repository),
            patch.object(handoff, "REPOSITORY_ROOT", repository),
        ):
            self._assert_after_second_fingerprint_failure(
                lambda: target.write_text("# changed\n", encoding="utf-8"),
                exception=RuntimeError,
                pattern="SHA-256 differs",
            )

    def test_training_code_symlink_substitution_after_second_fingerprint_fails_closed(
        self,
    ) -> None:
        repository = self.base / "training_repository_symlink_after_fingerprint"
        self._bind_fake_training_repository(repository)
        for substitution in ("direct", "ancestor"):
            with self.subTest(substitution=substitution):
                if substitution == "direct":
                    target = repository / "train-jax-10m/util.py"
                    backup = target.with_suffix(".original.py")

                    def mutate() -> None:
                        target.rename(backup)
                        target.symlink_to(backup.name)

                    def restore() -> None:
                        target.unlink()
                        backup.rename(target)

                else:
                    target = repository / "train-jax-10m"
                    backup = repository / "trainer-original"

                    def mutate() -> None:
                        target.rename(backup)
                        target.symlink_to(backup.name, target_is_directory=True)

                    def restore() -> None:
                        target.unlink()
                        backup.rename(target)

                try:
                    with (
                        patch.object(runner, "ROOT", repository),
                        patch.object(handoff, "REPOSITORY_ROOT", repository),
                    ):
                        self._assert_after_second_fingerprint_failure(
                            mutate,
                            exception=ValueError,
                            pattern="symbolic link",
                        )
                finally:
                    restore()

    def test_proposal_mutation_after_second_fingerprint_fails_closed(self) -> None:
        artifact = self.semantic_plan.chunks[0].batches[0].proposal

        def mutate() -> None:
            artifact.write_bytes(b"proposal changed after second fingerprint")

        self._assert_after_second_fingerprint_failure(
            mutate,
            exception=RuntimeError,
            pattern="artifact SHA-256 differs",
        )

    def test_result_mutation_after_second_fingerprint_fails_closed(self) -> None:
        artifact = self.semantic_plan.chunks[0].batches[0].result

        def mutate() -> None:
            artifact.write_bytes(b"result changed after second fingerprint")

        self._assert_after_second_fingerprint_failure(
            mutate,
            exception=RuntimeError,
            pattern="artifact SHA-256 differs",
        )

    def test_smaller_only_shard_mutation_after_second_fingerprint_fails_closed(
        self,
    ) -> None:
        artifact = self.semantic_plan.chunks[0].batches[0].shard
        self.assertNotIn(artifact, self.shard_files)

        def mutate() -> None:
            artifact.write_bytes(b"smaller-only shard changed after fingerprint")

        self._assert_after_second_fingerprint_failure(
            mutate,
            exception=RuntimeError,
            pattern="artifact SHA-256 differs",
        )

    def test_transitive_source_symlink_after_second_fingerprint_fails_closed(
        self,
    ) -> None:
        artifact = self.semantic_plan.chunks[0].batches[0].proposal
        substitute = self.base / "proposal-substitute.npz"

        def mutate() -> None:
            artifact.replace(substitute)
            artifact.symlink_to(substitute)

        self._assert_after_second_fingerprint_failure(
            mutate,
            exception=ValueError,
            pattern="must not be a symbolic link",
        )

    def test_transitive_source_path_substitution_after_second_fingerprint_fails_closed(
        self,
    ) -> None:
        source_root = self.semantic_plan.chunks[0].root
        substitute = self.base / "source-tree-substitute"

        def mutate() -> None:
            source_root.replace(substitute)
            source_root.symlink_to(substitute, target_is_directory=True)

        self._assert_after_second_fingerprint_failure(
            mutate,
            exception=ValueError,
            pattern="symbolic-link or path substitution",
        )

    def test_parent_mutation_during_transitive_hash_fails_final_rehash(self) -> None:
        source = self.semantic_plan.chunks[0].batches[0].proposal.resolve()
        parent = self.cumulative_views[2_048].manifest.path
        original_artifact = handoff.postcompletion._artifact
        mutated = False

        def mutate_parent_after_source_hash(
            path: Path,
            *,
            expected_sha256: object | None = None,
        ) -> runner.Artifact:
            nonlocal mutated
            observed = original_artifact(path, expected_sha256=expected_sha256)
            if observed.path == source and not mutated:
                parent.write_bytes(b"parent changed while children were hashed")
                mutated = True
            return observed

        with patch.object(
            handoff.postcompletion,
            "_artifact",
            side_effect=mutate_parent_after_source_hash,
        ):
            with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
                handoff.build_document_handoff(
                    self.root,
                    release_authenticator=self._authenticate,
                )
        self.assertTrue(mutated)
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_final_currentness_deduplicates_transitive_and_training_shards(
        self,
    ) -> None:
        authenticated = handoff.authenticate_postcompletion(
            self.root,
            release_authenticator=self._authenticate,
        )
        original_artifact = handoff.postcompletion._artifact
        calls_by_path: dict[Path, int] = {}

        def count_physical_hash(
            path: Path,
            *,
            expected_sha256: object | None = None,
        ) -> runner.Artifact:
            observed = original_artifact(path, expected_sha256=expected_sha256)
            calls_by_path[observed.path] = calls_by_path.get(observed.path, 0) + 1
            return observed

        with patch.object(
            handoff.postcompletion,
            "_artifact",
            side_effect=count_physical_hash,
        ):
            handoff._verify_authenticated_artifacts_current(authenticated)

        proposal = self.semantic_plan.chunks[0].batches[0].proposal.resolve()
        smaller_shard = self.semantic_plan.chunks[0].batches[0].shard.resolve()
        training_shard = self.shard_files[0].resolve()
        self.assertEqual(calls_by_path[proposal], 1)
        self.assertEqual(calls_by_path[smaller_shard], 1)
        self.assertEqual(calls_by_path[training_shard], 1)
        for view in authenticated.cumulative_combined_views.values():
            self.assertEqual(calls_by_path[view.summary.path], 3)
            self.assertEqual(calls_by_path[view.manifest.path], 3)
            self.assertEqual(calls_by_path[view.trajectory_map.path], 3)

    def test_source_category_counts_fail_closed(self) -> None:
        chunk = self.chunks[0]
        path = Path(str(chunk["summary_path"]))
        record = runner._strict_json(path)
        first = next(iter(record["counts"]["by_cell"].values()))
        first["attempted"] += 1
        _write_json(path, record)
        rebound = {
            **chunk,
            "summary_sha256": runner._sha256(path),
        }

        with self.assertRaisesRegex(RuntimeError, "counts do not close"):
            handoff._chunk_category_counts(
                rebound,
                family=str(chunk["family"]),
                split=str(chunk["split"]),
            )

    def test_non_jonswap_category_identities_fail_closed(self) -> None:
        arbitrary_categories = {
            f"arbitrary_{index}": {"accepted": 1}
            for index in range(len(handoff.STOKES_CATEGORY_IDS))
        }
        arbitrary_splits = {
            split: {"categories": arbitrary_categories} for split in runner.SPLIT_ORDER
        }
        with self.assertRaisesRegex(RuntimeError, "declared taxonomy"):
            handoff._validate_declared_category_identities(
                family="stokes",
                audit={},
                split_records=arbitrary_splits,
                categories=arbitrary_categories,
            )

        for family in ("tanaka", "benjamin_feir"):
            with (
                self.subTest(family=family),
                self.assertRaisesRegex(
                    RuntimeError,
                    "source category identities differ from its audit",
                ),
            ):
                handoff._validate_declared_category_identities(
                    family=family,
                    audit=runner._strict_json(self.audits[family]),
                    split_records={
                        split: {"categories": {"arbitrary": {"accepted": 1}}}
                        for split in runner.SPLIT_ORDER
                    },
                    categories={"arbitrary": {"accepted": 3}},
                )

    def test_jonswap_population_conditioning_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["jonswap_tma"])
        cell = next(iter(record["population_conditioning"]["cells"].values()))
        cell["attempted"] += 1
        cell["rejected"] += 1
        cell["rejection_rate"] = cell["rejected"] / cell["attempted"]
        self._replace_audit("jonswap_tma", record)

        with self.assertRaisesRegex(
            RuntimeError,
            "conditioning cells differ from authenticated source summaries",
        ):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rejection_rate_rejects_malformed_counts(self) -> None:
        malformed = (
            {"attempted": 0, "rejected": 0},
            {"attempted": True, "rejected": 0},
            {"attempted": 1, "rejected": False},
            {"attempted": float("nan"), "rejected": 0},
            {"attempted": 1, "rejected": float("inf")},
            {"attempted": 1, "rejected": 2},
        )
        for counts in malformed:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                handoff._rejection_rate(context="test", **counts)

    def test_family_audit_schema_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["benjamin_feir"])
        record["schema"] = "unexpected_schema"
        self._replace_audit("benjamin_feir", record)

        with self.assertRaisesRegex(RuntimeError, "wrong schema or status"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_adoption_gate_sha256_fails_closed(self) -> None:
        self.adoption_gate.write_bytes(b"corrupted")

        with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_adoption_gate_mutation_before_commit_fails_closed(self) -> None:
        original_build_record = handoff.build_record

        def mutate_after_build(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> dict[str, object]:
            record = original_build_record(authenticated)
            self.adoption_gate.write_bytes(b"changed after initial authentication")
            return record

        with patch.object(handoff, "build_record", side_effect=mutate_after_build):
            with self.assertRaisesRegex(RuntimeError, "artifact SHA-256 differs"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rebound_adoption_gate_criterion_fails_closed(self) -> None:
        record = runner._strict_json(self.adoption_gate)
        record["predeclared_gate"]["maximum_rejection_rate"] = 0.03
        _write_json(self.adoption_gate, record)

        with patch.object(
            handoff,
            "JONSWAP_ADOPTION_GATE_SHA256",
            runner._sha256(self.adoption_gate),
        ):
            with self.assertRaisesRegex(RuntimeError, "criterion differs"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rebound_adoption_gate_counts_fail_closed(self) -> None:
        record = runner._strict_json(self.adoption_gate)
        record["observed"]["accepted"] = 539
        _write_json(self.adoption_gate, record)

        with patch.object(
            handoff,
            "JONSWAP_ADOPTION_GATE_SHA256",
            runner._sha256(self.adoption_gate),
        ):
            with self.assertRaisesRegex(RuntimeError, "counts differ"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_rebound_failed_adoption_gate_fails_closed(self) -> None:
        record = runner._strict_json(self.adoption_gate)
        record["status"] = "fail"
        record["passed"] = False
        _write_json(self.adoption_gate, record)

        with patch.object(
            handoff,
            "JONSWAP_ADOPTION_GATE_SHA256",
            runner._sha256(self.adoption_gate),
        ):
            with self.assertRaisesRegex(RuntimeError, "not the passed v1 record"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_missing_or_extra_extrema_metric_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["benjamin_feir"])
        del record["support"]["extrema"]["focused_steepness"]
        record["support"]["extrema"]["undocumented_metric"] = _extrema(18_433, 0.0, 1.0)
        self._replace_audit("benjamin_feir", record)

        with self.assertRaisesRegex(RuntimeError, "wrong fields"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_malformed_extrema_shape_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["benjamin_feir"])
        del record["numerical_extrema"]["accepted"]["stage_residual"]["maximum"]
        self._replace_audit("benjamin_feir", record)

        with self.assertRaisesRegex(RuntimeError, "wrong fields"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_extrema_population_count_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["benjamin_feir"])
        record["numerical_extrema"]["accepted"]["stage_residual"]["count"] = 1
        self._replace_audit("benjamin_feir", record)

        with self.assertRaisesRegex(RuntimeError, "count differs"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_tanaka_achieved_crest_scope_fails_closed(self) -> None:
        record = runner._strict_json(self.audits["tanaka"])
        record["checks"]["achieved_per_crest_amplitudes_persisted"] = True
        self._replace_audit("tanaka", record)

        with self.assertRaisesRegex(RuntimeError, "achieved per-crest"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_stokes_unreported_extrema_require_bound_legacy_audit(self) -> None:
        record = runner._strict_json(self.audits["stokes"])
        record["legacy_numerical_audit"]["sha256"] = "f" * 64
        self._replace_audit("stokes", record)

        with self.assertRaisesRegex(RuntimeError, "different legacy numerical audit"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )

    def test_corrupted_stage_artifact_fails_closed(self) -> None:
        artifact = self.root / "worst_cases/all_families_worst_overview.png"
        artifact.write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_stage_mutation_before_commit_fails_closed(self) -> None:
        artifact = self.root / "worst_cases/all_families_worst_overview.png"
        original_build_record = handoff.build_record
        calls = 0

        def mutate_after_build(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> dict[str, object]:
            nonlocal calls
            record = original_build_record(authenticated)
            calls += 1
            if calls == 1:
                artifact.write_bytes(b"changed after initial authentication")
            return record

        with patch.object(handoff, "build_record", side_effect=mutate_after_build):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_corrupted_physical_training_shard_fails_closed(self) -> None:
        self.shard_files[0].write_bytes(b"corrupted shard")
        with self.assertRaises((RuntimeError, ValueError)):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_training_shard_mutation_after_record_build_fails_closed(self) -> None:
        original_build_record = handoff.build_record

        def mutate_after_build(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> dict[str, object]:
            record = original_build_record(authenticated)
            self.shard_files[0].write_bytes(b"changed after record build")
            return record

        with patch.object(handoff, "build_record", side_effect=mutate_after_build):
            with self.assertRaises((RuntimeError, ValueError)):
                handoff.build_document_handoff(
                    self.root, release_authenticator=self._authenticate
                )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_training_code_mutation_after_record_build_fails_closed(self) -> None:
        repository = self.base / "training_repository"
        self._bind_fake_training_repository(repository)
        target = repository / "train-jax-10m/util.py"

        original_build_record = handoff.build_record

        def mutate_after_build(
            authenticated: handoff.AuthenticatedPostcompletion,
        ) -> dict[str, object]:
            record = original_build_record(authenticated)
            target.write_text("# changed after record build\n", encoding="utf-8")
            return record

        with (
            patch.object(runner, "ROOT", repository),
            patch.object(handoff, "REPOSITORY_ROOT", repository),
            patch.object(
                handoff,
                "build_record",
                side_effect=mutate_after_build,
            ),
            self.assertRaisesRegex(RuntimeError, "SHA-256 differs"),
        ):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_missing_physical_training_shard_fails_closed(self) -> None:
        self.shard_files[0].unlink()
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_training_shard_path_must_match_combined_manifest(self) -> None:
        alternate = self.inputs / "alternate.npz"
        alternate.write_bytes(self.shard_files[0].read_bytes())
        record = runner._strict_json(self.root / "training_handoff_audit.json")
        shard = record["dataset_view"]["shards"][0]
        shard.update(runner._artifact(alternate).record())
        identity_payload = [
            {key: item[key] for key in ("index", "path", "bytes", "sha256", "rows")}
            for item in record["dataset_view"]["shards"]
        ]
        record["dataset_view"]["shard_identity_fingerprint"] = runner._canonical_sha256(
            identity_payload
        )
        _write_json(self.root / "training_handoff_audit.json", record)
        with self.assertRaisesRegex(RuntimeError, "path differs"):
            runner.validate_training_output(
                self.root / "training_handoff_audit.json",
                self.release.combined.summary,
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_stage_symlink_fails_closed(self) -> None:
        artifact = self.root / "family_case_examples.png"
        target = self.base / "external.png"
        artifact.replace(target)
        artifact.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_nested_stage_artifact_extras_fail_closed(self) -> None:
        renderer_path = self.root / "worst_cases/summary.json"
        renderer = runner._strict_json(renderer_path)
        original_renderer = deepcopy(renderer)
        artifact = next(iter(renderer["artifacts"].values()))
        artifact["sentinel_extra"] = "must be rejected"
        _write_json(renderer_path, renderer)
        with self.assertRaisesRegex(RuntimeError, "fields differ"):
            runner.validate_renderer_output(renderer_path.parent, self.release)

        _write_json(renderer_path, original_renderer)
        figure_path = self.root / "family_case_examples.json"
        figure = runner._strict_json(figure_path)
        figure["combined_view"]["manifest"]["sentinel_extra"] = "must be rejected"
        _write_json(figure_path, figure)
        with self.assertRaisesRegex(RuntimeError, "fields differ"):
            runner.validate_figure_output(
                self.root / "family_case_examples", self.release
            )

    def test_partial_stage_output_fails_closed(self) -> None:
        (self.root / "family_case_examples.png").unlink()
        with self.assertRaises(FileNotFoundError):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self.assertFalse((self.root / handoff.DEFAULT_NAME).exists())

    def test_mismatch_and_unrelated_output_are_never_overwritten(self) -> None:
        status = runner._strict_json(self.status_path)
        status["release_reauthentication"]["combined_summary_sha256"] = "f" * 64
        _write_json(self.status_path, status)
        with self.assertRaisesRegex(RuntimeError, "reauthentication differs"):
            handoff.build_document_handoff(
                self.root, release_authenticator=self._authenticate
            )
        self._status()
        unrelated = self.root / "unrelated.json"
        unrelated.write_text('{"owner":"someone_else"}\n', encoding="utf-8")
        before = unrelated.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "unrelated"):
            handoff.build_document_handoff(
                self.identity_path,
                output=unrelated,
                release_authenticator=self._authenticate,
            )
        self.assertEqual(unrelated.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
