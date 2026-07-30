"""Focused CPU tests for the pre-cascade spatial/order diagnostic."""
from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np  # noqa: E402

from scripts.run_precascade_spatial_order_diagnostic import (  # noqa: E402
    ArmDefinition,
    CaseProbe,
    DiagnosticContract,
    SourceCase,
    _saved_times,
    changed_arm_fields,
    production_arm_definitions,
    production_case_probes,
    resize_project_periodic_real,
    run_dynamic_arm,
    validate_arm_matrix,
)


class ArmMatrixTest(unittest.TestCase):
    def test_every_nonbaseline_arm_changes_exactly_one_factor(self) -> None:
        arms = production_arm_definitions()
        validate_arm_matrix(arms)
        baseline = arms[0]

        self.assertEqual(
            [arm.arm_id for arm in arms],
            [
                "baseline",
                "grid_2048",
                "padding_16",
                "cutoff_192",
                "order_5",
                "order_4",
            ],
        )
        self.assertEqual(changed_arm_fields(baseline, baseline), ())
        for arm in arms[1:]:
            self.assertEqual(
                changed_arm_fields(baseline, arm),
                (arm.changed_factor,),
            )

    def test_common_contract_and_arm_cutoffs_are_exact(self) -> None:
        contract = DiagnosticContract()
        self.assertEqual(contract.dt, 0.005)
        self.assertEqual(contract.saved_dt, 0.08)
        self.assertEqual(contract.gl2_residual_tolerance, 1.0e-8)
        self.assertEqual(contract.gl2_iteration_cap, 8)
        self.assertEqual(contract.common_wavenumber, 128.0)

        for arm in production_arm_definitions():
            self.assertLess(arm.delivered_wavenumber, arm.nx / 2)
            self.assertGreater(arm.delivered_wavenumber, 0.0)


class RestartGridTest(unittest.TestCase):
    def test_declared_restart_horizons_lie_on_saved_grid(self) -> None:
        contract = DiagnosticContract()
        probes = production_case_probes()
        self.assertEqual(
            [probe.restart_time for probe in probes], [80.0, 88.0]
        )
        self.assertEqual(
            [probe.terminal_time for probe in probes], [122.0, 112.0]
        )
        for probe in probes:
            times = _saved_times(probe, contract)
            self.assertEqual(float(times[0]), probe.restart_time)
            self.assertEqual(float(times[-1]), probe.terminal_time)
            np.testing.assert_allclose(
                np.diff(times), contract.saved_dt, rtol=0.0, atol=2.0e-14
            )


class SpectralResizeTest(unittest.TestCase):
    def test_resize_preserves_retained_modes_and_removes_others(self) -> None:
        length = 2.0 * math.pi
        input_nx = 256
        x = length * np.arange(input_nx, dtype=np.float64) / input_nx
        field = (
            0.4
            + 1.2 * np.cos(3.0 * x)
            - 0.7 * np.sin(11.0 * x)
            + 0.3 * np.cos(40.0 * x)
        )

        resized = resize_project_periodic_real(
            field,
            output_nx=512,
            length=length,
            maximum_wavenumber=32.0,
        )
        output_x = length * np.arange(512, dtype=np.float64) / 512
        expected = (
            0.4
            + 1.2 * np.cos(3.0 * output_x)
            - 0.7 * np.sin(11.0 * output_x)
        )

        np.testing.assert_allclose(resized, expected, rtol=0.0, atol=2.0e-14)

    def test_resize_preserves_leading_dimensions(self) -> None:
        length = 2.0 * math.pi
        x = length * np.arange(128, dtype=np.float64) / 128
        fields = np.stack((np.cos(5.0 * x), np.sin(7.0 * x)))
        resized = resize_project_periodic_real(
            fields,
            output_nx=256,
            length=length,
            maximum_wavenumber=16.0,
        )

        self.assertEqual(resized.shape, (2, 256))
        np.testing.assert_allclose(
            np.mean(resized, axis=-1),
            np.zeros(2),
            rtol=0.0,
            atol=1.0e-15,
        )


class DynamicOrchestrationSmokeTest(unittest.TestCase):
    def test_tiny_cpu_arm_records_fields_and_every_gl2_substep(self) -> None:
        contract = DiagnosticContract()
        nx = 64
        x = contract.length * np.arange(nx, dtype=np.float64) / nx
        eta0 = 1.0e-3 * np.cos(2.0 * x)
        xi0 = 1.0e-3 * np.sin(2.0 * x)
        probe = CaseProbe(
            case_id="synthetic",
            restart_time=0.0,
            terminal_time=contract.saved_dt,
            static_times=(0.0,),
        )
        source = SourceCase(
            case_id=probe.case_id,
            depth=1.0,
            source_times=np.asarray([0.0], dtype=np.float64),
            eta=eta0[None, :],
            xi=xi0[None, :],
            gxi=np.zeros((1, nx), dtype=np.float64),
            stage_converged=np.empty(0, dtype=np.bool_),
            step_times=np.empty(0, dtype=np.float64),
        )
        arm = ArmDefinition(
            arm_id="synthetic",
            nx=nx,
            pad_factor=2,
            delivered_wavenumber=8.0,
            dno_order=2,
            changed_factor="none",
        )

        arrays, timings = run_dynamic_arm(source, probe, arm, contract)

        self.assertEqual(arrays["eta"].shape, (2, nx))
        self.assertEqual(arrays["xi"].shape, (2, nx))
        self.assertEqual(arrays["gxi"].shape, (2, nx))
        self.assertEqual(arrays["gl2_converged"].shape, (16,))
        self.assertTrue(np.isfinite(arrays["eta"]).all())
        self.assertTrue(np.isfinite(arrays["xi"]).all())
        self.assertTrue(np.isfinite(arrays["gxi"]).all())
        self.assertTrue(np.all(arrays["gl2_converged"]))
        self.assertGreater(timings["compute_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
