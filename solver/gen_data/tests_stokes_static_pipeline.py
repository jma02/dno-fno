"""CPU end-to-end tests for the common static Stokes writer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.archive import BatchStatus, inspect_batch  # noqa: E402
from solver.gen_data.pipeline.manifest import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.reference import DiscreteDnoTarget  # noqa: E402
from solver.gen_data.stokes_sampling import (  # noqa: E402
    STOKES_SAMPLE_CELLS,
    StokesSample,
    sample_stokes_case,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    STATIC_STOKES_REQUIRED_CHECKS,
    StaticStokesContract,
    construct_stokes_state,
    evaluate_static_stokes_sample,
    write_static_stokes_batch,
)

jax.config.update("jax_enable_x64", True)

FINGERPRINT = "7" * 64


def assignment(
    cell_index: int,
    *,
    attempt_index: int,
    split_id: SplitId = SplitId.VALIDATION,
) -> AttemptAssignment:
    """Return a fixed Stokes assignment for integration tests."""

    return AttemptAssignment(
        case_key=CaseKey(
            family_id=PhysicalFamilyId.STOKES,
            revision_id=1,
            split_id=split_id,
            stream_id=9,
            attempt_index=attempt_index,
        ),
        cell_id=STOKES_SAMPLE_CELLS[cell_index].cell_id,
    )


def reduced_contract() -> StaticStokesContract:
    """Return the explicitly labeled CPU wiring contract."""

    return StaticStokesContract.reduced_wiring_evidence(
        DiscreteDnoTarget(
            nx=64,
            length=2.0 * math.pi,
            dno_order=2,
            pad_factor=2,
            maximum_wavenumber=24.0,
        )
    )


class StokesStaticPipelineTest(unittest.TestCase):
    def test_contract_requires_an_explicit_reduced_evidence_label(self) -> None:
        self.assertEqual(
            (
                PAPER_STATIC_STOKES_CONTRACT.target.nx,
                PAPER_STATIC_STOKES_CONTRACT.target.dno_order,
                PAPER_STATIC_STOKES_CONTRACT.target.pad_factor,
                PAPER_STATIC_STOKES_CONTRACT.target.maximum_wavenumber,
                PAPER_STATIC_STOKES_CONTRACT.role,
            ),
            (1024, 6, 8, 128.0, "paper_dataset"),
        )
        with self.assertRaisesRegex(ValueError, "must be labeled"):
            StaticStokesContract(target=reduced_contract().target)
        self.assertEqual(
            reduced_contract().role,
            "reduced_wiring_evidence_only",
        )

    def test_real_finite_and_deep_cases_reach_schema_v2_view(self) -> None:
        samples = (
            sample_stokes_case(assignment(0, attempt_index=10)),
            sample_stokes_case(assignment(2, attempt_index=11)),
            sample_stokes_case(assignment(0, attempt_index=12)),
            sample_stokes_case(assignment(2, attempt_index=13)),
        )
        contract = reduced_contract()

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            expected_proposal = (
                root
                / "proposals/stokes/validation/batch_000003.npz"
            )
            proposal_existed_during_construction: list[bool] = []

            def injected_constructor(
                sample: StokesSample,
                active_contract: StaticStokesContract,
            ) -> tuple[jax.Array, jax.Array]:
                proposal_existed_during_construction.append(
                    expected_proposal.exists()
                )
                eta, xi = construct_stokes_state(sample, active_contract)
                if sample.assignment.case_key.attempt_index == 12:
                    eta = eta.at[0].set(jnp.nan)
                if sample.assignment.case_key.attempt_index == 13:
                    eta = jnp.full_like(eta, -sample.depth)
                return eta, xi

            result = write_static_stokes_batch(
                root,
                samples,
                batch_id=3,
                config_fingerprint=FINGERPRINT,
                contract=contract,
                metadata={"purpose": "CPU wiring evidence"},
                state_constructor=injected_constructor,
            )

            self.assertTrue(all(proposal_existed_during_construction))
            self.assertEqual(
                inspect_batch(
                    result.paths,
                    expected_fingerprint=FINGERPRINT,
                ).status,
                BatchStatus.COMMITTED,
            )
            self.assertEqual(
                [outcome.decision.accepted for outcome in result.outcomes],
                [True, True, False, False],
            )
            self.assertTrue(
                result.outcomes[2].decision.failed
                & QualityReason.NONFINITE_STATE
            )
            self.assertTrue(
                result.outcomes[3].decision.failed
                & QualityReason.BOTTOM_CLEARANCE
            )

            with np.load(
                result.paths.proposal,
                allow_pickle=False,
            ) as proposal:
                np.testing.assert_array_equal(
                    proposal["case_id"],
                    np.asarray(
                        [
                            sample.assignment.case_key.case_id
                            for sample in samples
                        ],
                        dtype=np.int64,
                    ),
                )
                np.testing.assert_array_equal(
                    proposal["root_seed"],
                    np.full(4, 2026072204, dtype=np.uint64),
                )
                specifications = [
                    json.loads(value)
                    for value in proposal["case_spec_json"]
                ]
                self.assertEqual(
                    specifications,
                    [sample.to_json_record() for sample in samples],
                )
                self.assertIn(
                    "amplitude_attempts",
                    specifications[0],
                )
                proposal_metadata = json.loads(
                    str(proposal["metadata_json"])
                )
                self.assertEqual(
                    proposal_metadata["contract"]["role"],
                    "reduced_wiring_evidence_only",
                )

            with np.load(result.paths.shard, allow_pickle=False) as shard:
                self.assertEqual(shard["eta"].shape, (2, 64))
                np.testing.assert_array_equal(
                    shard["case_local_index"],
                    np.asarray([0, 1], dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    shard["frame_index"],
                    np.asarray([0, 0], dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    shard["time"],
                    np.asarray([0.0, 0.0], dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    shard["selected_dense_index"],
                    np.asarray([0, 0], dtype=np.int32),
                )
                self.assertLess(
                    float(np.max(np.abs(np.mean(shard["xi"], axis=1)))),
                    1.0e-8,
                )
                self.assertLess(
                    float(np.max(np.abs(np.mean(shard["gxi"], axis=1)))),
                    1.0e-8,
                )

            result_payload = json.loads(
                result.paths.result.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [case["row_count"] for case in result_payload["cases"]],
                [1, 1, 0, 0],
            )
            self.assertEqual(
                [case["first_row"] for case in result_payload["cases"]],
                [0, 1, -1, -1],
            )
            self.assertTrue(
                all(
                    case["required_bits"]
                    == int(STATIC_STOKES_REQUIRED_CHECKS)
                    for case in result_payload["cases"]
                )
            )

            view = build_dataset_view(
                root,
                (result.paths,),
                name="stokes_wiring",
                expected_fingerprint=FINGERPRINT,
            )
            manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["n_rows"], 2)
            self.assertEqual(manifest["n_trajectories"], 4)
            self.assertEqual(manifest["n_accepted_trajectories"], 2)
            self.assertEqual(
                manifest["split_counts"]["validation"],
                {"attempted": 4, "accepted": 2},
            )
            with np.load(view.trajectory_map, allow_pickle=False) as mapping:
                np.testing.assert_array_equal(
                    mapping["trajectory_split_id"],
                    np.full(4, 1, dtype=np.uint8),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_accepted"],
                    np.asarray([True, True, False, False]),
                )
                np.testing.assert_array_equal(
                    mapping["trajectory_row_count"],
                    np.asarray([1, 1, 0, 0], dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    mapping["frame_index"],
                    np.asarray([0, 0], dtype=np.int32),
                )

    def test_nonfinite_target_is_a_zero_row_rejection(self) -> None:
        sample = sample_stokes_case(assignment(2, attempt_index=30))
        contract = reduced_contract()

        def nonfinite_target(
            eta: jax.Array | np.ndarray,
            xi: jax.Array | np.ndarray,
            depth: float | jax.Array | np.ndarray,
            *,
            definition: DiscreteDnoTarget,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del depth, definition
            eta_array = jnp.asarray(eta)
            xi_array = jnp.asarray(xi)
            return eta_array, xi_array, jnp.full_like(eta_array, jnp.nan)

        outcome = evaluate_static_stokes_sample(
            sample,
            contract=contract,
            target_evaluator=nonfinite_target,
        )
        self.assertFalse(outcome.decision.accepted)
        self.assertIsNone(outcome.rows)
        self.assertTrue(
            outcome.decision.failed & QualityReason.NONFINITE_TARGET
        )
        self.assertEqual(outcome.metrics["target_finite"], False)

    def test_constructor_and_target_exceptions_propagate(self) -> None:
        sample = sample_stokes_case(assignment(2, attempt_index=31))
        contract = reduced_contract()

        def failing_constructor(
            sample: StokesSample,
            contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            del sample, contract
            raise ValueError("injected constructor bug")

        with self.assertRaisesRegex(ValueError, "injected constructor bug"):
            evaluate_static_stokes_sample(
                sample,
                contract=contract,
                state_constructor=failing_constructor,
            )

        def failing_target(
            eta: jax.Array | np.ndarray,
            xi: jax.Array | np.ndarray,
            depth: float | jax.Array | np.ndarray,
            *,
            definition: DiscreteDnoTarget,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del eta, xi, depth, definition
            raise RuntimeError("injected target bug")

        with self.assertRaisesRegex(RuntimeError, "injected target bug"):
            evaluate_static_stokes_sample(
                sample,
                contract=contract,
                target_evaluator=failing_target,
            )


if __name__ == "__main__":
    unittest.main()
