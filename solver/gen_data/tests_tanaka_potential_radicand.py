"""Focused CPU tests for fail-closed Tanaka potential construction."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA,
    TanakaPotentialRadicandError,
    _validate_tanaka_surface_potential_radicand,
    build_per_case_initial_conditions,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    ResidualControlledGL2Contract,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_SAMPLE_CELL_IDS,
    TanakaCrest,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    construct_tanaka_trajectory_batch,
    persist_sampled_trajectory_proposal,
    sample_tanaka_trajectory_cases,
)
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    make_default_tanaka_template,
)

jax.config.update("jax_enable_x64", True)

LENGTH = 2.0 * math.pi


def validate(
    radical: np.ndarray,
    *,
    speeds: np.ndarray,
    depths: np.ndarray,
    specs: list[TanakaCrest],
    case_ids: np.ndarray,
    components_within_case: tuple[int, ...],
) -> None:
    """Apply the production validator on a small deterministic grid."""
    _validate_tanaka_surface_potential_radicand(
        radical,
        x_grid=np.linspace(0.0, 1.5, radical.shape[1], dtype=np.float64),
        speed_per_crest=speeds,
        case_h_ref=depths,
        flat_specs=specs,
        crest_case_ids=case_ids,
        components_within_case=components_within_case,
    )


class TanakaPotentialRadicandValidationTest(unittest.TestCase):
    def test_positive_exact_zero_and_negative_zero_are_accepted(self) -> None:
        validate(
            np.asarray(((1.0, 0.0, -0.0, 4.0),), dtype=np.float64),
            speeds=np.asarray((2.0,), dtype=np.float64),
            depths=np.asarray((0.2,), dtype=np.float64),
            specs=[TanakaCrest(0.25, 0.5, 1)],
            case_ids=np.asarray((0,), dtype=np.int32),
            components_within_case=(0,),
        )

    def test_smallest_representable_negative_value_is_rejected(self) -> None:
        smallest_negative = np.nextafter(0.0, -np.inf)
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            validate(
                np.asarray(
                    ((1.0, 0.0, smallest_negative, 4.0),),
                    dtype=np.float64,
                ),
                speeds=np.asarray((2.0,), dtype=np.float64),
                depths=np.asarray((0.2,), dtype=np.float64),
                specs=[TanakaCrest(0.25, 0.5, -1)],
                case_ids=np.asarray((0,), dtype=np.int32),
                components_within_case=(0,),
            )

        record = caught.exception.failure_record
        self.assertEqual(
            record["reason"],
            "negative_or_nonfinite_surface_potential_radicand",
        )
        component = record["components"][0]
        self.assertEqual(component["minimum_radicand"], smallest_negative)
        self.assertEqual(component["minimum_radicand_grid_index"], 2)
        self.assertEqual(component["minimum_radicand_x"], 1.0)
        self.assertEqual(component["negative_count"], 1)
        self.assertEqual(component["nonfinite_count"], 0)

    def test_nonfinite_values_and_mixed_crests_have_strict_records(self) -> None:
        specifications = [
            TanakaCrest(0.10, 0.1, 1),
            TanakaCrest(0.20, 0.2, -1),
            TanakaCrest(0.30, 0.3, 1),
        ]
        radical = np.asarray(
            (
                (1.0, 2.0, 3.0, 4.0),
                (1.0, -0.25, 3.0, 4.0),
                (1.0, np.nan, np.inf, -np.inf),
            ),
            dtype=np.float64,
        )
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            validate(
                radical,
                speeds=np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
                depths=np.asarray((0.15, 0.25), dtype=np.float64),
                specs=specifications,
                case_ids=np.asarray((0, 0, 1), dtype=np.int32),
                components_within_case=(0, 1, 0),
            )

        record = caught.exception.failure_record
        self.assertEqual(
            record["schema"],
            TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA,
        )
        self.assertEqual(record["invalid_case_indices"], [0, 1])
        components = record["components"]
        self.assertEqual(
            [component["global_component_index"] for component in components],
            [1, 2],
        )
        self.assertEqual(
            [
                (
                    component["local_case_index"],
                    component["component_within_case"],
                )
                for component in components
            ],
            [(0, 1), (1, 0)],
        )
        self.assertEqual(components[0]["alpha"], specifications[1].alpha)
        self.assertEqual(components[0]["depth"], 0.15)
        self.assertEqual(components[0]["speed_squared"], 4.0)
        self.assertEqual(components[1]["alpha"], specifications[2].alpha)
        self.assertEqual(components[1]["negative_count"], 0)
        self.assertEqual(components[1]["nonfinite_count"], 3)
        self.assertEqual(components[1]["first_nonfinite_grid_index"], 1)
        self.assertEqual(components[1]["first_nonfinite_x"], 0.5)
        self.assertIsNone(components[1]["minimum_radicand"])
        self.assertIsNone(components[1]["minimum_radicand_over_speed_squared"])
        self.assertIsNone(components[1]["minimum_radicand_grid_index"])
        self.assertIsNone(components[1]["minimum_radicand_x"])
        json.dumps(record, sort_keys=True, allow_nan=False)
        self.assertNotIn("NaN", str(caught.exception))
        self.assertNotIn("Infinity", str(caught.exception))

    def test_nonfinite_or_nonpositive_speed_squared_is_rejected(self) -> None:
        with self.assertRaises(TanakaPotentialRadicandError) as caught:
            validate(
                np.ones((3, 4), dtype=np.float64),
                speeds=np.asarray((0.0, np.nan, np.inf), dtype=np.float64),
                depths=np.asarray((0.1, 0.2), dtype=np.float64),
                specs=[
                    TanakaCrest(0.1, 0.1, 1),
                    TanakaCrest(0.2, 0.2, 1),
                    TanakaCrest(0.3, 0.3, -1),
                ],
                case_ids=np.asarray((0, 0, 1), dtype=np.int32),
                components_within_case=(0, 1, 0),
            )

        record = caught.exception.failure_record
        self.assertEqual(
            record["reason"],
            "nonpositive_or_nonfinite_surface_potential_speed_squared",
        )
        self.assertEqual(record["invalid_case_indices"], [0, 1])
        components = record["components"]
        self.assertEqual(components[0]["speed_squared"], 0.0)
        self.assertIsNone(components[1]["unsigned_speed"])
        self.assertIsNone(components[1]["speed_squared"])
        self.assertIsNone(components[2]["unsigned_speed"])
        self.assertIsNone(components[2]["speed_squared"])
        json.dumps(record, sort_keys=True, allow_nan=False)


class TanakaPotentialRadicandIntegrationTest(unittest.TestCase):
    def test_actual_upper_seam_constructor_and_symmetries_remain_valid(
        self,
    ) -> None:
        nx = 64
        template = make_default_tanaka_template(
            depth=1.0,
            gravity=1.0,
            direction=1,
            nx=nx,
            length=LENGTH,
            center=0.0,
            dno_order=6,
            pad_factor=8,
        )
        eta, xi = build_per_case_initial_conditions(
            template_params=template,
            case_h_ref=np.asarray((0.35, 0.35, 0.35), dtype=np.float64),
            case_specs=[
                [TanakaCrest(0.45, 0.0, 1)],
                [TanakaCrest(0.45, LENGTH / 2.0, 1)],
                [TanakaCrest(0.45, 0.0, -1)],
            ],
            length=LENGTH,
            nx=nx,
            gravity=1.0,
        )
        eta_host = np.asarray(eta)
        xi_host = np.asarray(xi)
        self.assertTrue(np.isfinite(eta_host).all())
        self.assertTrue(np.isfinite(xi_host).all())
        np.testing.assert_allclose(
            eta_host[1],
            np.roll(eta_host[0], nx // 2),
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            xi_host[1],
            np.roll(xi_host[0], nx // 2),
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            eta_host[2],
            eta_host[0],
            rtol=2.0e-11,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            xi_host[2],
            -xi_host[0],
            rtol=2.0e-11,
            atol=2.0e-13,
        )

    def test_durable_proposal_survives_constructor_domain_failure(self) -> None:
        contract = ResidualControlledGL2Contract(
            nx=64,
            length=LENGTH,
            gravity=1.0,
            dno_order=0,
            pad_factor=1,
            maximum_wavenumber=16.0,
            production_dt=0.02,
            saved_dt=0.02,
            gl2_residual_tolerance=1.0e-8,
            gl2_iteration_cap=8,
            refinement_tolerance=1.0e-3,
            relative_floor=1.0e-12,
            target_time_chunk_size=2,
        )
        cell_id = TANAKA_SAMPLE_CELL_IDS[0]
        attempted = AttemptAssignment(
            case_key=CaseKey(
                family_id=2,
                revision_id=3,
                split_id=SplitId.TEST,
                stream_id=7,
                attempt_index=53,
            ),
            cell_id=cell_id,
        )
        sampled = sample_tanaka_trajectory_cases(
            (attempted,),
            contract=contract,
        )
        failure = TanakaPotentialRadicandError(
            {
                "schema": TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA,
                "reason": ("negative_or_nonfinite_surface_potential_radicand"),
                "invalid_case_indices": [0],
                "components": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            proposed = persist_sampled_trajectory_proposal(
                sampled,
                root=Path(directory),
                family_name="tanaka",
                batch_id=0,
                cell_codes={cell_id: 0},
                config_fingerprint="b" * 64,
                metadata={"test_scope": "durable_proposal_before_domain_check"},
            )
            with patch(
                "solver.gen_data.trajectory_family_adapters."
                "build_per_case_initial_conditions",
                side_effect=failure,
            ):
                with self.assertRaises(TanakaPotentialRadicandError):
                    construct_tanaka_trajectory_batch(proposed)

            self.assertTrue(proposed.paths.proposal.is_file())
            self.assertFalse(proposed.paths.shard.exists())
            self.assertFalse(proposed.paths.result.exists())
            self.assertFalse(proposed.paths.failure.exists())


if __name__ == "__main__":
    unittest.main()
