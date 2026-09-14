"""Local full-horizon experiment; does not modify the production solver."""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["DNO_TANAKA_DTYPE"] = "float64"

import jax
import jax.numpy as jnp
import numpy as np

from solver.gen_data.jonswap_tma import JonswapTmaParameters, ResolvedBand
from solver.gen_data.jonswap_tma_sampling import JonswapTmaSample
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
from solver.gen_data.trajectory_family_adapters import construct_jonswap_tma_trajectory_batch


def multiply_real(a: jax.Array, b: jax.Array, nx: int, pad_factor: int = 8) -> jax.Array:
    fa = jnp.fft.rfft(a, axis=-1).at[..., nx // 2].set(0)
    fb = jnp.fft.rfft(b, axis=-1).at[..., nx // 2].set(0)
    ya = jnp.fft.irfft(fa, n=pad_factor * nx, axis=-1)
    yb = jnp.fft.irfft(fb, n=pad_factor * nx, axis=-1)
    product = jnp.fft.rfft(ya * yb, axis=-1)[..., :nx // 2 + 1].at[..., nx // 2].set(0)
    return pad_factor * jnp.fft.irfft(product, n=nx, axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "real2", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load("outputs/paper_dataset_regenerated_20260914/replay_inputs.npz") as archive:
        inputs = {name: archive[name] for name in archive.files}
    groups = ([1080, 1, 22, 16286, 9791], [229])
    numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]

    if args.variant == "compare":
        comparisons = []
        for group_number, indices in enumerate(groups):
            for phase in ("adjustment", "production"):
                with (
                    np.load(args.output / f"baseline_{group_number}_{phase}.npz") as baseline,
                    np.load(args.output / f"real2_{group_number}_{phase}.npz") as candidate,
                ):
                    for column, index in enumerate(indices):
                        count = int(inputs[f"{phase}_time_count"][index])
                        steps = (count - 1) * numerical.substeps_per_saved_frame
                        fields = ("eta", "xi") if phase == "adjustment" else (
                            "eta", "xi", "target_eta", "target_xi", "target_gxi",
                        )
                        for values in (baseline, candidate):
                            np.testing.assert_array_equal(values["times"][:count], numerical.saved_dt * np.arange(count))
                            assert all(values[name].dtype == np.float64
                                       and np.isfinite(values[name][:count, column]).all() for name in fields)
                            if phase == "production":
                                assert np.min(values["target_eta"][:count, column] + inputs["depth"][index]) > 0
                                np.testing.assert_allclose(values["times"][inputs["old_dense_indices"][index]],
                                                           inputs["old_times"][index], rtol=0, atol=1e-12)
                                energy = values["hamiltonian"][:count, column]
                                assert np.max(np.abs(energy - energy[0])
                                              / max(abs(energy[0]), np.finfo(float).tiny)) <= 1e-3
                        errors = {}
                        for name in fields:
                            reference = baseline[name][:count, column]
                            difference = candidate[name][:count, column] - reference
                            errors[name] = {
                                "max_relative_l2": float(np.max(
                                    np.linalg.norm(difference, axis=-1)
                                    / np.maximum(np.linalg.norm(reference, axis=-1), np.finfo(float).tiny)
                                )),
                                "max_relative_peak": float(np.max(
                                    np.max(np.abs(difference), axis=-1)
                                    / np.maximum(np.max(np.abs(reference), axis=-1), np.finfo(float).tiny)
                                )),
                                "max_absolute": float(np.max(np.abs(difference))),
                            }
                        telemetry_differences = {
                            name: int(np.count_nonzero(
                                baseline[name][:steps, column] != candidate[name][:steps, column]
                            ))
                            for name in ("gl2_iterations", "gl2_converged", "gl2_stage_finite",
                                         "gl2_state_finite", "gl2_hit_iteration_cap")
                        }
                        result = {
                            "simulation_id": int(inputs["simulation_id"][index]), "phase": phase,
                            "frames": count, "errors": errors,
                            "telemetry_differences": telemetry_differences,
                            "passed": all(
                                np.isfinite(value) and value <= 1e-6
                                for error in errors.values()
                                for name, value in error.items() if name != "max_absolute"
                            ) and not any(value for name, value in telemetry_differences.items()
                                          if name != "gl2_iterations"),
                        }
                        comparisons.append(result)
                        print(json.dumps(result), flush=True)
        runs = [json.loads((args.output / f"{variant}.json").read_text())
                for variant in ("baseline", "real2")]
        report = {"comparisons": comparisons, "runs": runs,
                  "passed": all(row["passed"] for row in comparisons)
                  and all(run["complete"] and all(row["healthy"] for row in run["phases"])
                          for run in runs)}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("Full-horizon solver comparison failed; inspect comparison.json")
    else:
        jax.config.update("jax_enable_x64", True)
        # Fresh module namespaces prevent JIT traces from reusing the other implementation.
        modules = {}
        for name in ("solver.solvers.dno_series_jax", "solver.solvers.time_integrator",
                     "solver.gen_data.pipeline.dno_target", "solver.gen_data.pipeline.trajectory_integration"):
            module = types.ModuleType(name)
            module.__package__ = name.rsplit(".", 1)[0]
            source = Path(name.replace(".", "/") + ".py")
            exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
            modules[name.rsplit(".", 1)[1]] = module
        dno, integrator, target, integration = modules.values()
        if args.variant == "real2":
            dno.multiply = multiply_real
            numerical = numerical._replace(pad_factor=2)
        integrator.dno_series_eval = dno.dno_series_eval
        target.dno_series_eval = dno.dno_series_eval
        integration.compute_dno_target = target.compute_dno_target
        payload: dict[str, jax.Array] = {}

        def capture_rollout(*positional: Any, **keywords: Any) -> dict[str, jax.Array]:
            payload.clear()
            payload.update(integrator.rollout(*positional, **keywords))
            return payload

        integration.integrate_trajectory = capture_rollout
        report = {"variant": args.variant, "device": jax.local_devices()[0].device_kind,
                  "numerical": numerical._asdict(), "complete": False, "phases": [],
                  "note": "Cold timings include compilation and target evaluation, exclude artifact writes. "
                          "Both integration and M6 target multiplication use the selected variant."}
        references = {name: np.load(f"outputs/paper_dataset/arrays/{name}.npy", mmap_mode="r")
                      for name in ("eta", "xi", "gxi")}
        for group_number, indices in enumerate(groups):
            samples = tuple(JonswapTmaSample(
                JonswapTmaParameters(*(float(inputs[name][index]) for name in JonswapTmaParameters._fields)),
                inputs["phase_right"][index], inputs["phase_left"][index],
            ) for index in indices)
            initial, valid = construct_jonswap_tma_trajectory_batch(
                samples, numerical, band=ResolvedBand(numerical.length, numerical.target_maximum_wavenumber),
            )
            assert initial is not None and valid == tuple(range(len(indices)))
            eta0, xi0 = initial.eta0, initial.xi0
            for phase in ("adjustment", "production"):
                counts = inputs[f"{phase}_time_count"][indices]
                times = numerical.saved_dt * np.arange(int(counts.max()), dtype=np.float64)
                print(json.dumps({"begin": phase, "group": group_number,
                                  "simulation_ids": inputs["simulation_id"][indices].tolist(),
                                  "saved_counts": counts.tolist()}), flush=True)
                started = perf_counter()
                kwargs = dict(eta0=eta0, xi0=xi0, depths=initial.depths, saved_times=times, config=numerical)
                if phase == "adjustment":
                    result = integration.integrate_adjustment_batch(
                        **kwargs, nonlinear_ramp_times=inputs["nonlinear_ramp_time"][indices],
                        nonlinear_ramp_order=4,
                    )
                else:
                    result = integration.integrate_batch(**kwargs)
                data = {name: np.asarray(value) for name, value in jax.device_get(payload).items()}
                if phase == "production":
                    data.update({f"target_{name}": getattr(result, name) for name in ("eta", "xi", "gxi")})
                    data.update(result.solver_grid_health._asdict())
                elapsed = perf_counter() - started
                health = []
                for column, (index, count) in enumerate(zip(indices, counts, strict=True)):
                    steps = (count - 1) * numerical.substeps_per_saved_frame
                    fields = (data["eta"][:count, column], data["xi"][:count, column])
                    row = {
                        "simulation_id": int(inputs["simulation_id"][index]),
                        "finite": all(np.isfinite(field).all().item() for field in fields),
                        "minimum_water_column": float(np.min(fields[0] + initial.depths[column])),
                        "failed_steps": int(np.count_nonzero(~data["gl2_converged"][:steps, column])),
                        "max_stage_residual": float(np.max(data["gl2_stage_residual"][:steps, column])),
                        "iteration_histogram": np.bincount(data["gl2_iterations"][:steps, column], minlength=6).tolist(),
                    }
                    if phase == "production":
                        hamiltonian = data["hamiltonian"][:count, column]
                        row["hamiltonian_drift"] = float(np.max(np.abs(hamiltonian / hamiltonian[0] - 1)))
                        row["targets_finite"] = bool(np.isfinite(data["target_gxi"][:count, column]).all()
                                                    and data["dno_output_finite"][:count, column].all())
                        archive_errors = {}
                        start = int(inputs["old_first_row"][index])
                        for name in references:
                            reference = np.asarray(references[name][start:start + 16], dtype=np.float64)
                            difference = data[f"target_{name}"][inputs["old_dense_indices"][index], column] - reference
                            archive_errors[name] = float(max(np.max(
                                np.linalg.norm(difference, axis=-1) / np.linalg.norm(reference, axis=-1)
                            ), np.max(np.max(np.abs(difference), axis=-1) / np.max(np.abs(reference), axis=-1))))
                        row["archive_errors"] = archive_errors
                    row["healthy"] = (row["finite"] and row["minimum_water_column"] > 0
                                      and row["failed_steps"] == 0
                                      and (phase == "adjustment" or (
                                          row["targets_finite"] and row["hamiltonian_drift"] <= 1e-3
                                          and all(np.isfinite(error) and error <= 1e-6
                                                  for error in row["archive_errors"].values()))))
                    health.append(row)
                phase_report = {"group": group_number, "phase": phase, "seconds": elapsed,
                                "healthy": all(row["healthy"] for row in health), "cases": health}
                report["phases"].append(phase_report)
                np.savez(args.output / f"{args.variant}_{group_number}_{phase}.npz", allow_pickle=False, **data)
                (args.output / f"{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(phase_report), flush=True)
                if not phase_report["healthy"]:
                    raise RuntimeError("A full-horizon phase failed its existing numerical checks")
                eta0 = np.stack([data["eta"][count - 1, i] for i, count in enumerate(counts)])
                xi0 = np.stack([data["xi"][count - 1, i] for i, count in enumerate(counts)])
                del data, result
                payload.clear()
        report["complete"] = True
        (args.output / f"{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")
