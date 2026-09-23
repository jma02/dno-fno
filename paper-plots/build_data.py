"""Collect epoch-40 hard-P128 C27 diagnostics; render separately from JAX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import resample  # noqa: E402

from scripts.analyze_rollout_translation_decomposition import (  # noqa: E402
    aligned_fields,
    optimal_displacement,
    unwrap_displacements,
)
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918"
LENGTH = 2 * np.pi
FAMILIES = ("Stokes", "Tanaka", "JONSWAP/TMA", "Benjamin–Feir")
FAMILY_IDS = (1, 2, 4, 3)
REGIMES = ("stokes", "tanaka", "jonswap_tma", "benjamin_feir")


def archive_path(regime: str) -> Path:
    gpu = 0 if regime in ("stokes", "tanaka") else 1
    return RUN / f"eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{gpu}" / f"{regime}_trajs.npz"


def relative_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=-1) / np.maximum(
        np.linalg.norm(truth, axis=-1), 1e-30
    )


def predict_rows(
    predict: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    eta: np.ndarray, xi: np.ndarray, depth: np.ndarray,
) -> np.ndarray:
    batches = []
    for first in range(0, len(eta), 8):
        indices = np.minimum(np.arange(first, first + 8), len(eta) - 1)
        batches.append(np.asarray(predict(
            jnp.asarray(eta[indices]), jnp.asarray(xi[indices]), jnp.log(depth[indices])
        )))
    return np.concatenate(batches)[:len(eta)]


def save_data(name: str, values: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path = OUT / "plot-data.npz"
    data = {}
    if path.exists():
        with np.load(path) as archive:
            data = dict(archive)
    data.update({f"{name}_{key}": value for key, value in values.items()})
    data[f"{name}_metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **data)


def collect_rollouts() -> None:
    """Use the current TEST panel, with independently CS-evaluated energies."""
    start = time.perf_counter()
    data: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    eta_errors, xi_errors, families, energy_truth, energy_pred = [], [], [], [], []
    _, k = build_grid(1024, LENGTH)

    cs = jax.jit(lambda eta, xi, depth: dno_series_eval(eta, xi, k, depth[:, None], order=6, pad_factor=8))

    ham_indices = np.arange(0, 251, 10)
    for family, regime in enumerate(REGIMES):
        with np.load(archive_path(regime)) as archive:
            valid = archive["truth_valid"].astype(bool)
            ids = archive["simulation_ids"][valid]
            depths = archive["depths"][valid].astype(float)
            times = archive["times"].astype(float)
            truth_eta = archive["truth_eta"][:, valid].astype(float)
            truth_xi = archive["truth_xi"][:, valid].astype(float)
            pred_eta = archive["pred_eta"][:, valid].astype(float)
            pred_xi = archive["pred_xi"][:, valid].astype(float)
        truth_xi -= truth_xi.mean(axis=-1, keepdims=True)
        pred_xi -= pred_xi.mean(axis=-1, keepdims=True)
        e_eta = relative_error(pred_eta, truth_eta)
        e_xi = relative_error(pred_xi, truth_xi)
        eta_errors.append(e_eta)
        xi_errors.append(e_xi)
        families.append(np.full(len(ids), family))
        finite = np.isfinite(pred_eta).all(axis=(0, 2)) & np.isfinite(pred_xi).all(axis=(0, 2))
        wet = (pred_eta + depths[None, :, None]).min(axis=(0, 2)) > 0
        for i, case_id in enumerate(ids):
            cases.append({
                "regime": regime, "case_id": int(case_id), "h": float(depths[i]),
                "T": float(times[-1]), "finite_saved_states": bool(finite[i]),
                "positive_depth_saved_states": bool(wet[i]),
                "terminal_eta_error": float(e_eta[-1, i]),
                "terminal_xi_error": float(e_xi[-1, i]),
                "eta_trajectory_error": float(np.linalg.norm(pred_eta[:, i] - truth_eta[:, i]) / np.linalg.norm(truth_eta[:, i])),
                "xi_trajectory_error": float(np.linalg.norm(pred_xi[:, i] - truth_xi[:, i]) / np.linalg.norm(truth_xi[:, i])),
            })

        # Evaluate the SAME physical CS Hamiltonian on both sets of states.
        for eta, xi, destination in (
            (truth_eta, truth_xi, energy_truth), (pred_eta, pred_xi, energy_pred)
        ):
            sampled_eta = eta[ham_indices].reshape(-1, 1024)
            sampled_xi = xi[ham_indices].reshape(-1, 1024)
            sampled_h = np.tile(depths, len(ham_indices))
            energies = []
            for first in range(0, len(sampled_eta), 8):
                count = min(8, len(sampled_eta) - first)
                indices = np.minimum(np.arange(first, first + 8), len(sampled_eta) - 1)
                q = np.asarray(cs(
                    jnp.asarray(sampled_eta[indices]), jnp.asarray(sampled_xi[indices]),
                    jnp.asarray(sampled_h[indices]),
                ))[:count]
                energies.append(0.5 * LENGTH * np.mean(
                    sampled_xi[first:first + count] * q + sampled_eta[first:first + count] ** 2,
                    axis=-1,
                ))
            values = np.concatenate(energies).reshape(len(ham_indices), len(ids))
            destination.append((values - values[0]) / np.abs(values[0]))

        # Select typical-error examples, subject to physical profile constraints.
        selection: dict[str, int] = {}
        if regime in ("stokes", "jonswap_tma", "benjamin_feir"):
            name = {"stokes": "stokes", "jonswap_tma": "sea", "benjamin_feir": "bf"}[regime]
            selection[name] = int(np.argsort(e_eta[-1])[len(ids) // 2])
        if regime == "tanaka":
            summary = json.loads(archive_path(regime).with_name(f"{regime}_summary.json").read_text())
            groups = np.asarray(summary["evaluation_source"]["selected_parameter_group_ids"])[valid]
            candidates = np.flatnonzero(np.char.find(groups, "_m1_") >= 0)
            selection["tanaka"] = int(candidates[np.argsort(e_eta[-1, candidates])[len(candidates) // 2]])
            candidates = np.flatnonzero(np.char.find(groups, "_m2_") >= 0)
            # A resolved interior merger, selected by its crest amplification,
            # independent of the learned model's accuracy.
            amplification = np.max(truth_eta[15:-15], axis=(0, 2)) / np.max(truth_eta[0], axis=-1)
            selection["collision"] = int(candidates[np.argmax(amplification[candidates])])
        for name, i in selection.items():
            data[f"{name}_times"] = times
            data[f"{name}_depth"] = depths[i]
            data[f"{name}_case_id"] = ids[i]
            data[f"{name}_regime"] = np.asarray(regime)
            for field, values in (
                ("truth_eta", truth_eta), ("truth_xi", truth_xi),
                ("pred_eta", pred_eta), ("pred_xi", pred_xi)
            ):
                data[f"{name}_{field}"] = values[:, i]
            data[f"{name}_ham_truth"] = energy_truth[-1][:, i]
            data[f"{name}_ham_pred"] = energy_pred[-1][:, i]
            if name in ("stokes", "tanaka"):
                displacements = np.array([
                    optimal_displacement(p, t, LENGTH)
                    for p, t in zip(pred_eta[:, i], truth_eta[:, i], strict=True)
                ])
                period = LENGTH
                if name == "stokes":
                    peak = 1 + np.argmax(np.abs(np.fft.rfft(truth_eta[0, i]))[1:])
                    period /= peak
                displacements = (displacements + period / 2) % period - period / 2
                data[f"{name}_shift"] = unwrap_displacements(displacements, period)
                aligned = aligned_fields(pred_eta[:, i], displacements, LENGTH)
                data[f"{name}_aligned"] = aligned
                data[f"{name}_aligned_error"] = relative_error(aligned, truth_eta[:, i])
                assert np.all(data[f"{name}_aligned_error"] <= e_eta[:, i] + 1e-7)
        summaries.append({
            "regime": regime, "total": int(len(valid)), "valid_reference": int(valid.sum()),
            "finite_and_wet_saved_states": int(np.sum(finite & wet)),
            "median_terminal_eta": float(np.median(e_eta[-1])),
        })
        print(f"rollouts: {regime}, {len(ids)} valid, elapsed {time.perf_counter() - start:.1f}s", flush=True)
    data.update({
        "eta_error": np.concatenate(eta_errors, axis=1),
        "xi_error": np.concatenate(xi_errors, axis=1),
        "family": np.concatenate(families), "tau": np.linspace(0, 1, 251),
        "ham_tau": np.linspace(0, 1, 251)[ham_indices],
        "ham_truth": np.concatenate(energy_truth, axis=1),
        "ham_pred": np.concatenate(energy_pred, axis=1),
    })
    save_data("rollout", data, {"regimes": summaries, "cases": cases, "seconds": time.perf_counter() - start})


def collect_snapshots(samples: int) -> None:
    from solver.evals.model_rollout import build_predict_gxi_batched, load_run

    start = time.perf_counter()
    loaded = load_run(RUN, checkpoint="best")
    predict = build_predict_gxi_batched(loaded)
    dataset = ROOT / loaded.config["dataset"]
    arrays = {key: np.load(dataset / f"{key}.npy", mmap_mode="r")
              for key in ("eta", "xi", "gxi", "family_id", "depth", "time", "dataset_split")}
    rng = np.random.default_rng(20260923)
    data: dict[str, list[np.ndarray]] = {key: [] for key in ("rows", "family", "split", "error", "kh", "h")}
    structures = []
    examples: dict[str, np.ndarray] = {}
    for family, family_id in enumerate(FAMILY_IDS):
        for split, split_name in ((0, "train"), (2, "test")):
            pool = np.flatnonzero((arrays["family_id"] == family_id) & (arrays["dataset_split"] == split_name))
            rows = np.sort(rng.choice(pool, samples, replace=False))
            eta = np.asarray(arrays["eta"][rows], dtype=float)
            xi = np.asarray(arrays["xi"][rows], dtype=float)
            xi -= xi.mean(axis=-1, keepdims=True)
            truth = np.asarray(arrays["gxi"][rows], dtype=float)
            h = np.asarray(arrays["depth"][rows], dtype=float)
            q = predict_rows(predict, eta, xi, h)
            kh = (1 + np.argmax(np.abs(np.fft.rfft(eta, axis=-1))[:, 1:], axis=-1)) * h
            for key, values in (
                ("rows", rows), ("family", np.full(samples, family)), ("split", np.full(samples, split)),
                ("error", relative_error(q, truth)), ("kh", kh), ("h", h),
            ):
                data[key].append(values)
            print(f"snapshots: {FAMILIES[family]}, split={split}, n={samples}, median={np.median(data['error'][-1]):.4g}", flush=True)
            if split == 0:
                example = int(np.argsort(data["error"][-1])[samples // 2])
                examples[f"example_{family}_eta"] = eta[example]
                examples[f"example_{family}_h"] = h[example]
                examples[f"example_{family}_time"] = arrays["time"][rows[example]]
                examples[f"example_{family}_row"] = rows[example]
                # Independent uniform draws within this family's training pool.
                trial_rows = rng.choice(pool, (3, 100), replace=True)
                surfaces = np.asarray(arrays["eta"][trial_rows[0]], dtype=float)
                x1 = np.asarray(arrays["xi"][trial_rows[1]], dtype=float)
                x2 = np.asarray(arrays["xi"][trial_rows[2]], dtype=float)
                x1 -= x1.mean(axis=-1, keepdims=True)
                x2 -= x2.mean(axis=-1, keepdims=True)
                depths = np.asarray(arrays["depth"][trial_rows[0]], dtype=float)
                a, b = rng.uniform(-1, 1, (2, 100, 1))
                q1, q2, qm = [predict_rows(predict, surfaces, inputs, depths)
                              for inputs in (x1, x2, a * x1 + b * x2)]
                residual = np.linalg.norm(qm - a * q1 - b * q2, axis=-1) / (
                    abs(a[:, 0]) * np.linalg.norm(q1, axis=-1) + abs(b[:, 0]) * np.linalg.norm(q2, axis=-1)
                )
                # Positivity uses independent matched snapshots, as in the draft.
                positive_rows = rng.choice(pool, 100, replace=False)
                positive_eta = np.asarray(arrays["eta"][positive_rows], dtype=float)
                positive_xi = np.asarray(arrays["xi"][positive_rows], dtype=float)
                positive_xi -= positive_xi.mean(axis=-1, keepdims=True)
                positive_depth = np.asarray(arrays["depth"][positive_rows], dtype=float)
                positive_q = predict_rows(predict, positive_eta, positive_xi, positive_depth)
                quadratic = np.sum(positive_xi * positive_q, axis=-1) / np.sum(positive_xi**2, axis=-1)
                structures.append({
                    "family": FAMILIES[family], "trials": 100,
                    "linearity_mean": float(np.mean(residual)), "linearity_max": float(np.max(residual)),
                    "negative_quadratic_percent": float(100 * np.mean(quadratic < -1e-10)),
                    "minimum_quadratic": float(np.min(quadratic)), "tolerance": 1e-10,
                })
    save_data("snapshot", {**{k: np.concatenate(v) for k, v in data.items()}, **examples},
              {"samples_per_family_per_split": samples, "structures": structures, "seconds": time.perf_counter() - start})


def benchmarks() -> None:
    from solver.evals.model_rollout import build_predict_gxi_batched, load_run, rollout_surrogate
    from solver.solvers import time_integrator as ti

    start = time.perf_counter()
    predict = build_predict_gxi_batched(load_run(RUN, checkpoint="best"))
    with np.load(archive_path("stokes")) as archive:
        eta0 = archive["truth_eta"][0, 0].astype(float)
        xi0 = archive["truth_xi"][0, 0].astype(float)
        depth = float(archive["depths"][0])
    xi0 -= xi0.mean()
    orders = (2, 4, 6, 8)
    latency = []
    for nx in (128, 256, 512, 1024, 2048):
        eta = jnp.asarray(resample(eta0, nx))[None]
        xi = jnp.asarray(resample(xi0, nx))[None]
        _, k = build_grid(nx, LENGTH)
        methods = {"C27": jax.jit(lambda e, x: predict(e, x, jnp.log(jnp.array([depth]))))}
        for order in orders:
            methods[f"CS-{order}"] = jax.jit(
                lambda e, x, order=order: dno_series_eval(e, x, k, depth, order, pad_factor=8)
            )
        for name, method in methods.items():
            result = method(eta, xi).block_until_ready()
            assert np.isfinite(result).all()
            durations = []
            for _ in range(7):
                tick = time.perf_counter()
                method(eta, xi).block_until_ready()
                durations.append(time.perf_counter() - tick)
            latency.append({"method": name, "nx": nx, "seconds": float(np.median(durations)), "samples_seconds": durations})
        print(f"latency: N={nx}, elapsed {time.perf_counter() - start:.1f}s", flush=True)
    nx, dt, final = 256, 0.01, 1.0
    times = jnp.linspace(0, final, 51)
    eta = jnp.asarray(resample(eta0, nx))[None]
    xi = jnp.asarray(resample(xi0, nx))[None]
    initial = ti.State(eta=eta, xi=xi)
    params = ti.make_solver_params(nx, LENGTH, depth, dno_order=10, pad_factor=8, filter_fraction=0.5)
    records, solutions = [], {}
    for order in (10, 2, 4, 6, 8, 0):
        name = f"CS-{order}" if order else "C27"
        if order:
            action = jax.jit(lambda e, x, order=order: dno_series_eval(e, x, params.k, depth, order, pad_factor=8))
        else:
            action = jax.jit(lambda e, x: predict(e, x, jnp.log(jnp.array([depth]))))
        integrate = jax.jit(lambda: rollout_surrogate(initial, times, params, action, substeps=2))
        solution = jax.block_until_ready(integrate())
        assert np.isfinite(solution["eta"]).all()
        durations = []
        for _ in range(3):
            tick = time.perf_counter()
            jax.block_until_ready(integrate())
            durations.append(time.perf_counter() - tick)
        solutions[name] = np.asarray(solution["eta"])
        records.append({"method": name, "order": order, "seconds": float(np.median(durations)), "samples_seconds": durations})
        print(f"rollout benchmark: {name}, median {records[-1]['seconds']:.3f}s", flush=True)
    for record in records:
        solution = solutions[record["method"]]
        reference = solutions["CS-10"]
        record["eta_trajectory_error"] = float(np.linalg.norm(solution - reference) / np.linalg.norm(reference))
        record["eta_terminal_error"] = float(relative_error(solution[-1], reference[-1])[0])
    save_data("benchmark", {}, {
        "latency": latency, "rollouts": records, "nx_rollout": nx, "dt": dt, "T": final,
        "depth": depth, "rollout_cutoff_mode": 64, "pad_factor": 8,
        "precision": "FP32 learned inference; FP64 CS and integration",
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "seconds": time.perf_counter() - start,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rollouts", "snapshots", "benchmarks"))
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if args.stage == "rollouts":
        collect_rollouts()
    elif args.stage == "snapshots":
        collect_snapshots(args.samples)
    else:
        benchmarks()
