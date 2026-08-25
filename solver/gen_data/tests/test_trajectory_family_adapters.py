"""CPU wiring tests for the three trajectory-family adapters.

The reduced ``N=64``, ``M=0`` GL2 runs below test software wiring only.  They
are not spatial-, order-, or full-horizon validation of the paper contract.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_SAMPLE_CELL_IDS,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RELATIVE_FREQUENCY_WINDOW,
    finite_depth_angular_frequency,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_SAMPLING_REVISION_V4,
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    BatchPaths,
    BatchStatus,
    CaseCommitRecord,
    commit_batch,
    ensure_shard,
    inspect_batch,
    record_fatal_failure,
)
from solver.gen_data.pipeline.manifest import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
    paper_dataset_revision_id,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    PAPER_JONSWAP_GL2_CONTRACT,
    ResidualControlledGL2Contract,
    execute_production_trajectory,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
    outcomes_from_production,
)
from solver.gen_data.pipeline.writer import (  # noqa: E402
    commit_case_outcomes,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_SAMPLE_CELL_IDS,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
    persist_sampled_trajectory_proposal,
    resolved_band_for_contract,
    sample_benjamin_feir_trajectory_cases,
    sample_jonswap_tma_trajectory_cases,
    sample_tanaka_trajectory_cases,
)

jax.config.update("jax_enable_x64", True)


def wiring_contract() -> ResidualControlledGL2Contract:
    """Return a deliberately reduced real-GL2 software-wiring contract."""

    return ResidualControlledGL2Contract(
        nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        production_dt=0.02,
        saved_dt=0.02,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        target_time_chunk_size=2,
    )


def assignment(
    *,
    family_id: int,
    cell_id: str,
    attempt_index: int,
    revision_id: int | None = None,
) -> AttemptAssignment:
    """Return one deterministic test-split attempt."""

    selected_revision = (
        paper_dataset_revision_id(PhysicalFamilyId(family_id))
        if revision_id is None
        else revision_id
    )
    return AttemptAssignment(
        case_key=CaseKey(
            family_id=family_id,
            revision_id=selected_revision,
            split_id=SplitId.TEST,
            stream_id=7,
            attempt_index=attempt_index,
        ),
        cell_id=cell_id,
    )


def assert_valid_initial_batch(
    test: unittest.TestCase,
    batch: TrajectoryInitialBatch,
    *,
    contract: ResidualControlledGL2Contract,
) -> None:
    """Check dtype, graph, band, mean, and JSON invariants."""

    test.assertEqual(batch.eta0.dtype, np.float64)
    test.assertEqual(batch.xi0.dtype, np.float64)
    test.assertEqual(batch.depths.dtype, np.float64)
    test.assertEqual(batch.eta0.shape, (1, contract.nx))
    test.assertEqual(batch.xi0.shape, (1, contract.nx))
    test.assertTrue(np.isfinite(batch.eta0).all())
    test.assertTrue(np.isfinite(batch.xi0).all())
    test.assertTrue(np.all(batch.depths[:, None] + batch.eta0 > 0.0))
    test.assertLess(float(np.max(np.abs(np.mean(batch.xi0, axis=-1)))), 1.0e-14)

    modes = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            contract.nx,
            d=contract.length / contract.nx,
        )
    )
    for field in (batch.eta0, batch.xi0):
        coefficients = np.fft.fft(field, axis=-1)
        outside = coefficients[:, np.abs(modes) > contract.maximum_wavenumber]
        test.assertLess(float(np.max(np.abs(outside))), 1.0e-11)
    json.dumps(
        batch.specification_records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_single_row_shard(
    paths: BatchPaths,
    *,
    proposal_sha256: str,
    config_fingerprint: str,
    batch: TrajectoryInitialBatch,
) -> None:
    """Write one valid orphaned shard for adapter replay tests."""

    ensure_shard(
        paths,
        {
            "eta": np.asarray(batch.eta0[:1], dtype=np.float32),
            "xi": np.asarray(batch.xi0[:1], dtype=np.float32),
            "gxi": np.zeros_like(batch.eta0[:1], dtype=np.float32),
            "depth": np.asarray(batch.depths[:1], dtype=np.float64),
            "time": np.asarray([0.0], dtype=np.float64),
            "case_local_index": np.asarray([0], dtype=np.int32),
            "frame_index": np.asarray([0], dtype=np.int32),
            "selected_dense_index": np.asarray([0], dtype=np.int32),
            "config_fingerprint": np.asarray(config_fingerprint),
            "proposal_sha256": np.asarray(proposal_sha256),
        },
    )


class TrajectoryFamilyAdapterTest(unittest.TestCase):
    def test_paper_jonswap_constructs_on_target_band_before_wide_evolution(
        self,
    ) -> None:
        contract = PAPER_JONSWAP_GL2_CONTRACT
        attempted = assignment(
            family_id=4,
            revision_id=JONSWAP_TMA_SAMPLING_REVISION_V4,
            cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
            attempt_index=29,
        )
        sampled = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        record = sampled.specification_records[0]
        band = resolved_band_for_contract(contract)

        self.assertEqual(contract.nx, 2048)
        self.assertEqual(contract.maximum_wavenumber, 704.0)
        self.assertEqual(contract.target_definition.nx, 1024)
        self.assertEqual(contract.target_definition.maximum_wavenumber, 128.0)
        self.assertEqual(band.maximum_wavenumber, 128.0)
        self.assertEqual(band.transition_wavenumber, 96.0)
        self.assertEqual(len(record["phase_right"]), 128)
        self.assertEqual(len(record["phase_left"]), 128)
        self.assertEqual(
            record["initial_projection"]["maximum_wavenumber"],
            128.0,
        )
        self.assertEqual(
            record["constructor_settings"]["density_window"],
            PAPER_RELATIVE_FREQUENCY_WINDOW,
        )
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="9" * 64,
                metadata={"test_scope": "paper_target_band_before_evolution"},
            )
            initial = construct_jonswap_tma_trajectory_batch(proposed)

        metrics = initial.construction_metrics[0]
        self.assertEqual(metrics["initial_discrete_peak_wavenumber"], 6.0)
        self.assertGreaterEqual(
            int(metrics["initial_half_maximum_spectral_cell_count"]),
            1,
        )
        self.assertAlmostEqual(
            float(metrics["initial_realized_height_ratio"]),
            1.0,
            places=13,
        )
        self.assertLess(
            float(metrics["initial_linear_hamiltonian_relative_error"]),
            1.0e-12,
        )
        self.assertGreater(
            float(metrics["initial_minimum_water_column"]),
            0.0,
        )

        wavenumbers = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                contract.nx,
                d=contract.length / contract.nx,
            )
        )
        outside_target = np.abs(wavenumbers) > 128.0
        for field in (initial.eta0, initial.xi0):
            coefficients = np.fft.fft(field, axis=-1)
            self.assertLess(
                float(np.max(np.abs(coefficients[:, outside_target]))),
                1.0e-10,
            )

    def test_revision_4_jonswap_uses_the_published_relative_frequency_band(
        self,
    ) -> None:
        contract = PAPER_JONSWAP_GL2_CONTRACT
        attempted = assignment(
            family_id=4,
            revision_id=JONSWAP_TMA_SAMPLING_REVISION_V4,
            cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
            attempt_index=31,
        )
        sampled = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        record = sampled.specification_records[0]
        self.assertEqual(
            record["initial_condition_constructor"],
            "relative_frequency_jonswap_tma_linear_state_v2",
        )
        self.assertEqual(
            record["constructor_settings"]["density_window"],
            PAPER_RELATIVE_FREQUENCY_WINDOW,
        )
        self.assertEqual(
            record["constructor_settings"]["relative_frequency_minimum"],
            PAPER_RELATIVE_FREQUENCY_MINIMUM,
        )
        self.assertEqual(
            record["constructor_settings"]["relative_frequency_maximum"],
            PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )

        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="4" * 64,
                metadata={"test_scope": "revision_4_relative_frequency_band"},
            )
            initial = construct_jonswap_tma_trajectory_batch(proposed)

        parameters = sampled.samples[0].parameters
        wavenumbers = (
            2.0
            * np.pi
            * np.fft.rfftfreq(
                contract.nx,
                d=contract.length / contract.nx,
            )
        )
        frequencies = finite_depth_angular_frequency(
            wavenumbers,
            depth=parameters.depth,
            gravity=contract.gravity,
        )
        peak_frequency = finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber]),
            depth=parameters.depth,
            gravity=contract.gravity,
        )[0]
        coefficients = np.fft.rfft(initial.eta0[0])
        far_outside = (frequencies > 2.6 * peak_frequency) | (
            frequencies < 0.4 * peak_frequency
        )
        self.assertLess(float(np.max(np.abs(coefficients[far_outside]))), 1.0e-10)

    def test_buffered_tanaka_reuses_the_exact_delivered_band_initial_state(
        self,
    ) -> None:
        hard = wiring_contract()
        buffered = replace(
            hard,
            maximum_wavenumber=24.0,
            target_maximum_wavenumber=16.0,
            post_step_state_filter="hou_li",
            post_step_maximum_wavenumber=24.0,
        )
        attempted = assignment(
            family_id=2,
            cell_id=TANAKA_SAMPLE_CELL_IDS[0],
            attempt_index=19,
        )
        hard_sampled = sample_tanaka_trajectory_cases(
            (attempted,),
            contract=hard,
        )
        buffered_sampled = sample_tanaka_trajectory_cases(
            (attempted,),
            contract=buffered,
        )
        self.assertEqual(
            hard_sampled.specification_records,
            buffered_sampled.specification_records,
        )
        self.assertEqual(
            buffered_sampled.specification_records[0]["initial_projection"][
                "maximum_wavenumber"
            ],
            16.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hard_proposal = persist_sampled_trajectory_proposal(
                hard_sampled,
                root=root / "hard",
                family_name="tanaka",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="1" * 64,
                metadata={"test_scope": "hard_initial_state"},
            )
            buffered_proposal = persist_sampled_trajectory_proposal(
                buffered_sampled,
                root=root / "buffered",
                family_name="tanaka",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="2" * 64,
                metadata={"test_scope": "buffered_initial_state"},
            )
            hard_initial = construct_tanaka_trajectory_batch(hard_proposal)
            buffered_initial = construct_tanaka_trajectory_batch(buffered_proposal)

        np.testing.assert_array_equal(
            hard_initial.eta0,
            buffered_initial.eta0,
        )
        np.testing.assert_array_equal(
            hard_initial.xi0,
            buffered_initial.xi0,
        )
        np.testing.assert_array_equal(
            hard_initial.depths,
            buffered_initial.depths,
        )

    def test_all_families_construct_finite_graph_states_with_exact_replay(
        self,
    ) -> None:
        contract = wiring_contract()
        tanaka_assignment = assignment(
            family_id=2,
            cell_id=TANAKA_SAMPLE_CELL_IDS[0],
            attempt_index=19,
        )
        bf_assignment = assignment(
            family_id=3,
            cell_id=BENJAMIN_FEIR_SAMPLE_CELL_IDS[0],
            attempt_index=23,
        )
        finite_random_sea_cell = JONSWAP_TMA_SAMPLE_CELL_IDS[9]
        jonswap_assignment = assignment(
            family_id=4,
            cell_id=finite_random_sea_cell,
            attempt_index=29,
        )

        families = (
            (
                "tanaka",
                sample_tanaka_trajectory_cases,
                construct_tanaka_trajectory_batch,
                tanaka_assignment,
            ),
            (
                "benjamin_feir",
                sample_benjamin_feir_trajectory_cases,
                construct_benjamin_feir_trajectory_batch,
                bf_assignment,
            ),
            (
                "jonswap_tma",
                sample_jonswap_tma_trajectory_cases,
                construct_jonswap_tma_trajectory_batch,
                jonswap_assignment,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constructed: dict[str, TrajectoryInitialBatch] = {}
            for batch_id, (family, sampler, constructor, attempted) in enumerate(
                families
            ):
                with self.subTest(family=family):
                    first_sampled = sampler((attempted,), contract=contract)
                    replayed_sampled = sampler((attempted,), contract=contract)
                    self.assertEqual(
                        first_sampled.specification_records,
                        replayed_sampled.specification_records,
                    )
                    if family == "benjamin_feir":
                        bf_record = first_sampled.specification_records[0]
                        self.assertNotIn("schema", bf_record)
                        self.assertEqual(
                            bf_record["initial_condition_constructor"],
                            ("jcp09_equation_33_with_project_fifth_order_carrier_v2"),
                        )
                    proposed = persist_sampled_trajectory_proposal(
                        first_sampled,
                        root=root,
                        family_name=family,
                        batch_id=batch_id,
                        cell_codes={attempted.cell_id: 0},
                        config_fingerprint="5" * 64,
                        metadata={
                            "test_scope": "reduced_N64_M0_wiring_only",
                        },
                    )
                    self.assertTrue(proposed.paths.proposal.is_file())
                    first = constructor(proposed)
                    second = constructor(proposed)
                    assert_valid_initial_batch(self, first, contract=contract)
                    np.testing.assert_array_equal(first.eta0, second.eta0)
                    np.testing.assert_array_equal(first.xi0, second.xi0)
                    np.testing.assert_array_equal(first.depths, second.depths)
                    self.assertEqual(
                        first.specification_records,
                        second.specification_records,
                    )
                    constructed[family] = first

            random_sea = constructed["jonswap_tma"]
            record = random_sea.specification_records[0]
            band = resolved_band_for_contract(contract)
            self.assertEqual(
                len(record["phase_right"]),
                int(np.floor(band.maximum_wavenumber)),
            )
            self.assertEqual(
                len(record["phase_left"]),
                len(record["phase_right"]),
            )
            self.assertEqual(
                record["constructor_settings"]["density_window"],
                PAPER_RELATIVE_FREQUENCY_WINDOW,
            )

    def test_construction_refuses_an_absent_or_changed_proposal(self) -> None:
        contract = wiring_contract()
        attempted = assignment(
            family_id=4,
            cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
            attempt_index=31,
        )
        sampled = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="6" * 64,
                metadata={"test_scope": "transaction_order"},
            )
            proposed.paths.proposal.unlink()
            with self.assertRaisesRegex(
                RuntimeError,
                "proposal",
            ):
                construct_jonswap_tma_trajectory_batch(proposed)

        replayed = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                replayed,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="6" * 64,
                metadata={"test_scope": "transaction_order"},
            )
            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                replace(proposed, proposal_sha256="0" * 64)
            replayed.samples[0].phase_right[0] += 0.125
            with self.assertRaisesRegex(
                RuntimeError,
                "sampled specification changed",
            ):
                construct_jonswap_tma_trajectory_batch(proposed)

    def test_jonswap_construction_refuses_a_changed_density_window(self) -> None:
        contract = wiring_contract()
        attempted = assignment(
            family_id=4,
            cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
            attempt_index=35,
        )
        sampled = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        changed_settings = {
            **sampled.construction_settings,
            "density_window": "changed_window",
        }
        changed = replace(sampled, construction_settings=changed_settings)
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                changed,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="7" * 64,
                metadata={"test_scope": "density_window_contract"},
            )
            with self.assertRaisesRegex(ValueError, "density window"):
                construct_jonswap_tma_trajectory_batch(proposed)

    def test_shard_written_replay_rechecks_hash_and_specification(self) -> None:
        contract = wiring_contract()
        attempted = assignment(
            family_id=4,
            cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
            attempt_index=37,
        )
        sampled = sample_jonswap_tma_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={attempted.cell_id: 0},
                config_fingerprint="8" * 64,
                metadata={"test_scope": "shard_written_replay"},
            )
            initial = construct_jonswap_tma_trajectory_batch(proposed)
            write_single_row_shard(
                proposed.paths,
                proposal_sha256=proposed.proposal_sha256,
                config_fingerprint=proposed.config_fingerprint,
                batch=initial,
            )
            self.assertEqual(
                inspect_batch(proposed.paths).status,
                BatchStatus.SHARD_WRITTEN,
            )

            replayed_token = replace(proposed)
            replayed = construct_jonswap_tma_trajectory_batch(replayed_token)
            np.testing.assert_array_equal(replayed.eta0, initial.eta0)
            np.testing.assert_array_equal(replayed.xi0, initial.xi0)
            np.testing.assert_array_equal(replayed.depths, initial.depths)

            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                replace(proposed, proposal_sha256="0" * 64)
            original_phase = float(sampled.samples[0].phase_right[0])
            sampled.samples[0].phase_right[0] = original_phase + 0.125
            with self.assertRaisesRegex(
                RuntimeError,
                "sampled specification changed",
            ):
                construct_jonswap_tma_trajectory_batch(proposed)
            sampled.samples[0].phase_right[0] = original_phase

            commit_batch(
                proposed.paths,
                cases=(
                    CaseCommitRecord(
                        case_id=attempted.case_key.case_id,
                        accepted=True,
                        required_bits=0,
                        evaluated_bits=0,
                        failed_bits=0,
                        first_row=0,
                        row_count=1,
                        metrics={},
                    ),
                ),
                metadata={"test_scope": "terminal_status"},
            )
            self.assertEqual(
                inspect_batch(proposed.paths).status,
                BatchStatus.COMMITTED,
            )
            with self.assertRaisesRegex(RuntimeError, "proposed or shard-written"):
                construct_jonswap_tma_trajectory_batch(proposed)
            with self.assertRaisesRegex(RuntimeError, "proposed or shard-written"):
                replace(proposed)

    def test_construction_refuses_failed_and_corrupt_batches(self) -> None:
        contract = wiring_contract()
        cell_id = JONSWAP_TMA_SAMPLE_CELL_IDS[9]

        failed_attempt = assignment(
            family_id=4,
            cell_id=cell_id,
            attempt_index=39,
        )
        failed_sampled = sample_jonswap_tma_trajectory_cases(
            (failed_attempt,),
            contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            failed = persist_sampled_trajectory_proposal(
                failed_sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={cell_id: 0},
                config_fingerprint="9" * 64,
                metadata={"test_scope": "failed_status"},
            )
            record_fatal_failure(
                failed.paths,
                phase="construct",
                exception_type="RuntimeError",
                message="injected",
                telemetry={},
            )
            self.assertEqual(
                inspect_batch(failed.paths).status,
                BatchStatus.FAILED,
            )
            with self.assertRaisesRegex(RuntimeError, "proposed or shard-written"):
                construct_jonswap_tma_trajectory_batch(failed)
            with self.assertRaisesRegex(RuntimeError, "proposed or shard-written"):
                replace(failed)

        corrupt_attempt = assignment(
            family_id=4,
            cell_id=cell_id,
            attempt_index=40,
        )
        corrupt_sampled = sample_jonswap_tma_trajectory_cases(
            (corrupt_attempt,),
            contract=contract,
        )
        with tempfile.TemporaryDirectory() as directory:
            corrupt = persist_sampled_trajectory_proposal(
                corrupt_sampled,
                root=Path(directory),
                family_name="jonswap_tma",
                batch_id=0,
                cell_codes={cell_id: 0},
                config_fingerprint="a" * 64,
                metadata={"test_scope": "corrupt_status"},
            )
            initial = construct_jonswap_tma_trajectory_batch(corrupt)
            write_single_row_shard(
                corrupt.paths,
                proposal_sha256=corrupt.proposal_sha256,
                config_fingerprint=corrupt.config_fingerprint,
                batch=initial,
            )
            with np.load(corrupt.paths.shard, allow_pickle=False) as archive:
                shard = {name: np.asarray(archive[name]) for name in archive.files}
            shard["proposal_sha256"] = np.asarray("0" * 64)
            np.savez(corrupt.paths.shard, **shard)
            with self.assertRaisesRegex(ValueError, "different proposal hash"):
                construct_jonswap_tma_trajectory_batch(corrupt)
            with self.assertRaisesRegex(ValueError, "different proposal hash"):
                replace(corrupt)

    def test_bf_and_jonswap_real_gl2_to_committed_manifest_wiring(
        self,
    ) -> None:
        """Exercise the real CPU executor; numerical settings are smoke-only."""

        contract = wiring_contract()
        cases = (
            (
                "benjamin_feir",
                assignment(
                    family_id=3,
                    cell_id=BENJAMIN_FEIR_SAMPLE_CELL_IDS[0],
                    attempt_index=41,
                ),
                sample_benjamin_feir_trajectory_cases,
                construct_benjamin_feir_trajectory_batch,
            ),
            (
                "jonswap_tma",
                assignment(
                    family_id=4,
                    cell_id=JONSWAP_TMA_SAMPLE_CELL_IDS[9],
                    attempt_index=43,
                ),
                sample_jonswap_tma_trajectory_cases,
                construct_jonswap_tma_trajectory_batch,
            ),
        )
        fingerprint = "7" * 64
        saved_times = np.asarray([0.0, 0.02, 0.04], dtype=np.float64)
        policy = StoredTimePolicy(
            tanaka_count=3,
            benjamin_feir_count=3,
            random_sea_count=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for family, attempted, sampler, constructor in cases:
                sampled = sampler((attempted,), contract=contract)
                proposed = persist_sampled_trajectory_proposal(
                    sampled,
                    root=root,
                    family_name=family,
                    batch_id=0,
                    cell_codes={attempted.cell_id: 0},
                    config_fingerprint=fingerprint,
                    metadata={
                        "test_scope": "reduced_N64_M0_wiring_only",
                    },
                )
                self.assertTrue(proposed.paths.proposal.is_file())
                self.assertFalse(proposed.paths.shard.exists())
                self.assertFalse(proposed.paths.result.exists())
                initial = constructor(proposed)
                execution = execute_production_trajectory(
                    initial.eta0,
                    initial.xi0,
                    initial.depths,
                    saved_times,
                    contract=contract,
                )
                self.assertTrue(execution.cases[0].accepted)
                outcomes = outcomes_from_production(
                    execution,
                    initial.depths,
                    family=family,
                    length=contract.length,
                    policy=policy,
                )
                commit_case_outcomes(
                    proposed.paths,
                    proposed.proposal_arrays,
                    outcomes,
                    metadata={
                        "test_scope": "reduced_N64_M0_wiring_only",
                    },
                )
                paths.append(proposed.paths)

            view = build_dataset_view(
                root,
                tuple(paths),
                name="trajectory_family_adapter_smoke",
                length=contract.length,
                expected_fingerprint=fingerprint,
            )
            manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_trajectories"], 2)
            self.assertEqual(manifest["n_rows"], 6)
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_accepted"],
                    np.asarray([True, True]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([3, 3]),
                )


if __name__ == "__main__":
    unittest.main()
