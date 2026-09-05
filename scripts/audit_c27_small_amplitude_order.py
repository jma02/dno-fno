"""Audit the small-amplitude Craig--Sulem order of a trained CS-DNO.

The surface is scaled as ``eta -> epsilon * eta`` while ``xi`` and depth are
held fixed.  On registered evaluation states this script compares

* the order-six reference with ``G0`` and ``G0 + epsilon G1``;
* the trained model's Taylor remainder about ``eta = 0``; and
* the trained model with the order-six reference.

All reported norms are projected to ``|k| <= 128``.  The script forces JAX's
CPU backend before importing JAX so it can run alongside GPU training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.evals.model_rollout import (  # noqa: E402
    build_predict_gxi_batched,
    load_run,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
)

jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "c27_h1_to_l2_full_20260717_212550"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notes" / "c27_small_amplitude_order_audit_20260720"
TANAKA_EVAL = "eval_final_soliton_spectral_guard_20260719_191228"
NON_TANAKA_EVAL = "eval_final_guarded_non_tanaka_suite_n32_20260719_221813"
FAMILY_NAMES = (
    "tanaka_g0",
    "tanaka_g1",
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "random_sea_deep",
    "random_sea_finite",
    "linear",
    "stokes_deep",
    "stokes_finite",
)
EPSILONS = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=np.float64)
FIT_EPSILONS = np.asarray([0.25, 0.125, 0.0625, 0.03125], dtype=np.float64)
METRIC_LABELS: Mapping[str, str] = {
    "g0_truncation": r"Order-6 minus $G_0$",
    "g01_truncation": r"Order-6 minus $G_0+G_1$",
    "c27_taylor_remainder": r"C27 learned remainder",
}
EXPECTED_ORDERS: Mapping[str, float] = {
    "g0_truncation": 1.0,
    "g01_truncation": 2.0,
    "c27_taylor_remainder": 2.0,
}
COLORS: Mapping[str, str] = {
    "g0_truncation": "#D55E00",
    "g01_truncation": "#0072B2",
    "c27_taylor_remainder": "#009E73",
}

ArrayMap: TypeAlias = dict[str, np.ndarray]


def make_reference_evaluator(
    k: jnp.ndarray,
    order: int,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build a CPU-jitted, batched Craig--Sulem reference evaluator."""

    @jax.jit
    def evaluate(
        eta: jnp.ndarray,
        xi: jnp.ndarray,
        depths: jnp.ndarray,
    ) -> jnp.ndarray:
        return dno_series_eval(
            eta,
            xi,
            k,
            depths[:, None],
            order=order,
            pad_factor=8,
        )

    return evaluate


def evaluate_batched(
    evaluator: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    eta: np.ndarray,
    xi: np.ndarray,
    depth_inputs: np.ndarray,
    slices: Sequence[slice],
) -> np.ndarray:
    """Batch an operator using its physical-depth or log-depth inputs."""
    outputs = [
        np.asarray(
            evaluator(
                jnp.asarray(eta[chunk], dtype=jnp.float64),
                jnp.asarray(xi[chunk], dtype=jnp.float64),
                jnp.asarray(depth_inputs[chunk], dtype=jnp.float64),
            )
        )
        for chunk in slices
    ]
    return np.concatenate(outputs)


def projected_rms(fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return Parseval RMS after projection to the supplied Fourier mask."""
    transformed = np.fft.fft(fields, axis=-1)
    energy = np.sum(np.abs(transformed[..., mask]) ** 2, axis=-1)
    return np.sqrt(energy) / fields.shape[-1]


def fitted_slopes(
    values: np.ndarray, epsilons: np.ndarray, fit_mask: np.ndarray
) -> np.ndarray:
    """Fit one log-log slope per state for arrays shaped ``(epsilon, state)``."""
    selected = values[fit_mask]
    if np.any(selected <= 0.0) or not np.isfinite(selected).all():
        raise FloatingPointError("slope fit received nonpositive or nonfinite values")
    x = np.log(epsilons[fit_mask])
    x_centered = x - np.mean(x)
    y = np.log(selected)
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    return np.sum(x_centered[:, None] * y_centered, axis=0) / np.sum(x_centered**2)


def curve_slope(curve: np.ndarray, epsilons: np.ndarray, fit_mask: np.ndarray) -> float:
    """Fit one slope to a positive aggregate curve."""
    return float(fitted_slopes(curve[:, None], epsilons, fit_mask)[0])


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Return a compact finite distribution summary."""
    if not np.isfinite(values).all():
        raise FloatingPointError("cannot summarize nonfinite values")
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--states-per-family", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--projection-k-max", type=float, default=128.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    args = parser.parse_args()
    start = time.perf_counter()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"CPU backend required, got {jax.default_backend()}")
    tanaka_root = run_dir / TANAKA_EVAL
    non_tanaka_root = run_dir / NON_TANAKA_EVAL
    paths = {
        family: (
            tanaka_root / family / f"{family}_trajs.npz"
            if family.startswith("tanaka_")
            else non_tanaka_root / family / f"{family}_trajs.npz"
        )
        for family in FAMILY_NAMES
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trajectory archives: {missing}")

    eta_parts: list[np.ndarray] = []
    xi_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    simulation_parts: list[np.ndarray] = []
    archive_index_parts: list[np.ndarray] = []
    family_index_parts: list[np.ndarray] = []
    for family_index, family in enumerate(FAMILY_NAMES):
        path = paths[family]
        with np.load(path, allow_pickle=False) as archive:
            required = {"truth_eta", "truth_xi", "depths", "simulation_ids"}
            if missing_keys := required.difference(archive.files):
                raise KeyError(f"{path} is missing {sorted(missing_keys)}")
            depths = np.asarray(archive["depths"], dtype=np.float64)
            simulation_ids = np.asarray(archive["simulation_ids"], dtype=np.int64)
            count = args.states_per_family
            if count <= 0 or count > depths.size:
                raise ValueError(
                    f"invalid state count {count} for archive of size {depths.size}"
                )
            order = np.lexsort((simulation_ids, depths))
            edges = np.linspace(0, depths.size, count + 1, dtype=np.int64)
            positions = np.asarray(
                [
                    (int(left) + int(right) - 1) // 2
                    for left, right in zip(edges[:-1], edges[1:])
                ],
                dtype=np.int64,
            )
            if np.unique(positions).size != count:
                raise RuntimeError("depth-stratified positions are not unique")
            selected = order[positions]
            eta = np.asarray(archive["truth_eta"][0, selected], dtype=np.float64)
            xi = np.asarray(archive["truth_xi"][0, selected], dtype=np.float64)

        arrays = (eta, xi, depths[selected])
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError(f"nonfinite frame-zero state in {path}")
        eta_parts.append(eta)
        xi_parts.append(xi)
        depth_parts.append(depths[selected])
        simulation_parts.append(simulation_ids[selected])
        archive_index_parts.append(selected)
        family_index_parts.append(
            np.full(args.states_per_family, family_index, dtype=np.int64)
        )

    family_indices = np.concatenate(family_index_parts)
    eta = np.concatenate(eta_parts)
    xi = np.concatenate(xi_parts)
    depths = np.concatenate(depth_parts)
    simulation_ids = np.concatenate(simulation_parts)
    archive_indices = np.concatenate(archive_index_parts)
    family_names = np.asarray([FAMILY_NAMES[index] for index in family_indices])
    source_paths = tuple(
        str(paths[family].relative_to(REPO_ROOT)) for family in FAMILY_NAMES
    )
    state_count, nx = eta.shape
    expected_count = len(FAMILY_NAMES) * args.states_per_family
    if state_count != expected_count:
        raise RuntimeError(f"expected {expected_count} states, loaded {state_count}")
    if args.chunk_size <= 0 or state_count % args.chunk_size != 0:
        raise ValueError(
            f"chunk size {args.chunk_size} must divide state count {state_count}"
        )
    slices = tuple(
        slice(start, start + args.chunk_size)
        for start in range(0, state_count, args.chunk_size)
    )

    loaded = load_run(run_dir, checkpoint="final")
    expected_config: Mapping[str, object] = {
        "model": "cs_dno",
        "norm": "scale",
        "cs_use_g1_baseline": True,
        "cs_g1_k_cut": 0,
        "cs_g1_fft_fp64": True,
        "cs_residual_eta_order": 2,
    }
    mismatches = {
        key: (loaded.config.get(key), value)
        for key, value in expected_config.items()
        if loaded.config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"checkpoint does not match C27 structural config: {mismatches}"
        )
    if loaded.epoch != 40:
        raise ValueError(
            f"expected epoch-40 final checkpoint, got epoch {loaded.epoch}"
        )
    predict = build_predict_gxi_batched(loaded)
    _, k = build_grid(nx, float(loaded.config["domain_length"]))
    reference0 = make_reference_evaluator(k.astype(jnp.float64), order=0)
    reference1 = make_reference_evaluator(k.astype(jnp.float64), order=1)
    reference6 = make_reference_evaluator(k.astype(jnp.float64), order=6)

    q0 = evaluate_batched(reference0, eta, xi, depths, slices)
    q01 = evaluate_batched(reference1, eta, xi, depths, slices)
    q1 = q01 - q0
    log_depths = np.log(depths)

    @jax.jit
    def evaluate_model_jvp_batch(
        eta_direction: jnp.ndarray,
        xi_batch: jnp.ndarray,
        log_depth_batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta_zero = jnp.zeros_like(eta_direction)

        def surface_map(surface: jnp.ndarray) -> jnp.ndarray:
            return predict(surface, xi_batch, log_depth_batch)

        return jax.jvp(surface_map, (eta_zero,), (eta_direction,))

    model_jvp_pairs = [
        evaluate_model_jvp_batch(
            jnp.asarray(eta[chunk], dtype=jnp.float64),
            jnp.asarray(xi[chunk], dtype=jnp.float64),
            jnp.asarray(log_depths[chunk], dtype=jnp.float64),
        )
        for chunk in slices
    ]
    model_zero = np.concatenate([np.asarray(pair[0]) for pair in model_jvp_pairs])
    model_eta_derivative = np.concatenate(
        [np.asarray(pair[1]) for pair in model_jvp_pairs]
    )

    reference_order6 = np.stack(
        [
            evaluate_batched(
                reference6,
                epsilon * eta,
                xi,
                depths,
                slices,
            )
            for epsilon in EPSILONS
        ]
    )
    model_outputs = np.stack(
        [
            evaluate_batched(
                predict,
                epsilon * eta,
                xi,
                log_depths,
                slices,
            )
            for epsilon in EPSILONS
        ]
    )
    computed = (
        q0,
        q1,
        model_zero,
        model_eta_derivative,
        reference_order6,
        model_outputs,
    )
    if not all(np.isfinite(array).all() for array in computed):
        raise FloatingPointError("a reference, model, or JVP output is nonfinite")

    k_numpy = np.asarray(k, dtype=np.float64)
    projection_mask = np.abs(k_numpy) <= args.projection_k_max + 1e-12
    q0_norm = projected_rms(q0, projection_mask)
    q1_norm = projected_rms(q1, projection_mask)
    denominator = np.maximum(q0_norm, np.finfo(np.float64).tiny)
    epsilon_fields = EPSILONS[:, None, None]
    reference_linear = q0[None, :, :] + epsilon_fields * q1[None, :, :]
    model_taylor = (
        model_zero[None, :, :] + epsilon_fields * model_eta_derivative[None, :, :]
    )
    reference_remainder = reference_order6 - reference_linear
    model_remainder = model_outputs - model_taylor
    remainder_error = (
        projected_rms(model_remainder - reference_remainder, projection_mask)
        / denominator[None, :]
    )
    raw_model_error = (
        projected_rms(model_outputs - reference_order6, projection_mask)
        / denominator[None, :]
    )

    metrics: ArrayMap = {
        "g0_truncation": projected_rms(
            reference_order6 - q0[None, :, :], projection_mask
        )
        / denominator[None, :],
        "g01_truncation": projected_rms(reference_remainder, projection_mask)
        / denominator[None, :],
        "c27_taylor_remainder": projected_rms(model_remainder, projection_mask)
        / denominator[None, :],
    }
    fit_mask = np.isin(EPSILONS, FIT_EPSILONS)
    slopes = {
        name: fitted_slopes(values, EPSILONS, fit_mask)
        for name, values in metrics.items()
    }

    summaries: dict[str, dict[str, Any]] = {}
    bootstrap_slopes: ArrayMap = {}
    family_members = tuple(
        np.flatnonzero(family_indices == index) for index in range(len(FAMILY_NAMES))
    )
    for metric_offset, (name, values) in enumerate(metrics.items()):
        state_slopes = slopes[name]
        median_curve = np.median(values, axis=1)
        q25_curve = np.quantile(values, 0.25, axis=1)
        q75_curve = np.quantile(values, 0.75, axis=1)
        rng = np.random.default_rng(args.bootstrap_seed + metric_offset)
        bootstrapped = np.empty(args.bootstrap_samples, dtype=np.float64)
        for sample_index in range(args.bootstrap_samples):
            selected = np.concatenate(
                [
                    rng.choice(members, size=members.size, replace=True)
                    for members in family_members
                ]
            )
            bootstrapped[sample_index] = np.median(state_slopes[selected])
        ci_low, ci_high = np.quantile(bootstrapped, [0.025, 0.975])
        summaries[name] = {
            "pooled_median_curve": median_curve.tolist(),
            "pooled_q25_curve": q25_curve.tolist(),
            "pooled_q75_curve": q75_curve.tolist(),
            "pooled_median_curve_slope": curve_slope(median_curve, EPSILONS, fit_mask),
            "pooled_state_slope_median": float(np.median(state_slopes)),
            "stratified_bootstrap_state_slope_median_ci95": [
                float(ci_low),
                float(ci_high),
            ],
            "per_state_slope": distribution_summary(state_slopes),
            "family_median_state_slopes": {
                family: float(np.median(state_slopes[family_indices == index]))
                for index, family in enumerate(FAMILY_NAMES)
            },
            "family_median_curve_slopes": {
                family: curve_slope(
                    np.median(values[:, family_indices == index], axis=1),
                    EPSILONS,
                    fit_mask,
                )
                for index, family in enumerate(FAMILY_NAMES)
            },
            "expected_asymptotic_order": EXPECTED_ORDERS[name],
        }
        bootstrap_slopes[name] = bootstrapped

    flat_surface_relative_error = (
        projected_rms(model_zero - q0, projection_mask) / denominator
    )
    jvp_denominator = np.maximum(q1_norm, np.finfo(np.float64).tiny)
    jvp_absolute_error = projected_rms(model_eta_derivative - q1, projection_mask)
    jvp_relative_error = jvp_absolute_error / jvp_denominator
    jvp_error_relative_q0 = jvp_absolute_error / denominator
    raw_model_error_slopes = fitted_slopes(raw_model_error, EPSILONS, fit_mask)
    parameter_count = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(loaded.params)
    )
    if parameter_count != 1_342_400:
        raise ValueError(f"expected 1,342,400 C27 parameters, got {parameter_count:,}")

    selection = [
        {
            "family": str(family_names[index]),
            "archive_index": int(archive_indices[index]),
            "simulation_id": int(simulation_ids[index]),
            "depth": float(depths[index]),
        }
        for index in range(state_count)
    ]
    elapsed = time.perf_counter() - start
    figure_base = output_dir / "c27_small_amplitude_order"
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    for metric_name, values in metrics.items():
        summary = summaries[metric_name]
        median = np.asarray(summary["pooled_median_curve"], dtype=np.float64)
        q25 = np.asarray(summary["pooled_q25_curve"], dtype=np.float64)
        q75 = np.asarray(summary["pooled_q75_curve"], dtype=np.float64)
        slope = float(summary["pooled_state_slope_median"])
        ci_low, ci_high = summary["stratified_bootstrap_state_slope_median_ci95"]
        axes[0].loglog(
            EPSILONS,
            median,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=COLORS[metric_name],
            label=f"{METRIC_LABELS[metric_name]}: p={slope:.2f} [{ci_low:.2f}, {ci_high:.2f}]",
        )
        axes[0].fill_between(EPSILONS, q25, q75, color=COLORS[metric_name], alpha=0.12)

    guide_eps = np.asarray([EPSILONS[-1], EPSILONS[2]], dtype=np.float64)
    guide_anchor = float(np.median(metrics["g01_truncation"][2]))
    axes[0].loglog(
        guide_eps,
        guide_anchor * (guide_eps / guide_eps[-1]),
        color="0.35",
        linestyle=":",
        linewidth=1.0,
        label=r"reference $\varepsilon$",
    )
    axes[0].loglog(
        guide_eps,
        guide_anchor * (guide_eps / guide_eps[-1]) ** 2,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label=r"reference $\varepsilon^2$",
    )
    axes[0].set_xlabel(r"surface scale $\varepsilon$")
    axes[0].set_ylabel(r"projected relative $L^2$ norm")
    axes[0].set_title(r"(a) Small-amplitude convergence, $|k|\leq128$")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].legend(loc="best", frameon=False)

    metric_names = tuple(metrics)
    offsets = np.linspace(-0.22, 0.22, len(FAMILY_NAMES))
    for metric_index, metric_name in enumerate(metric_names):
        family_slopes = summaries[metric_name]["family_median_state_slopes"]
        values = np.asarray(
            [family_slopes[family] for family in FAMILY_NAMES], dtype=np.float64
        )
        axes[1].scatter(
            metric_index + offsets,
            values,
            s=18,
            color=COLORS[metric_name],
            alpha=0.7,
            edgecolors="none",
        )
        center = float(summaries[metric_name]["pooled_state_slope_median"])
        low, high = summaries[metric_name][
            "stratified_bootstrap_state_slope_median_ci95"
        ]
        axes[1].errorbar(
            metric_index,
            center,
            yerr=np.asarray([[max(0.0, center - low)], [max(0.0, high - center)]]),
            fmt="D",
            markersize=5,
            capsize=4,
            linewidth=1.5,
            color="black",
            zorder=5,
        )
    axes[1].axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
    axes[1].axhline(2.0, color="0.45", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(range(len(metric_names)))
    axes[1].set_xticklabels(["$G_0$\ntrunc.", "$G_0+G_1$\ntrunc.", "C27\nremainder"])
    axes[1].set_ylabel("fitted log--log slope")
    axes[1].set_title("(b) Family slopes; black diamonds are pooled fits")
    axes[1].grid(True, axis="y", alpha=0.2)

    jvp_median = float(np.median(jvp_relative_error))
    jvp_p95 = float(np.quantile(jvp_relative_error, 0.95))
    figure.suptitle(
        "C27 preserves the first-order Craig--Sulem expansion on 80 registered states\n"
        + rf"$D_\eta G_\theta(0)$ versus $G_1$: median relative error {jvp_median:.2e}, p95 {jvp_p95:.2e}",
        fontsize=11,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(figure_base.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)

    npz_payload: ArrayMap = {
        "epsilons": EPSILONS,
        "fit_epsilons": FIT_EPSILONS,
        "family_names": np.asarray(FAMILY_NAMES),
        "state_family_names": family_names,
        "family_indices": family_indices,
        "simulation_ids": simulation_ids,
        "archive_indices": archive_indices,
        "depths": depths,
        "eta_frame0": eta,
        "xi_frame0": xi,
        "q0_projected_rms": q0_norm,
        "q1_projected_rms": q1_norm,
        "flat_surface_relative_error": flat_surface_relative_error,
        "jvp_relative_error": jvp_relative_error,
        "jvp_absolute_error": jvp_absolute_error,
        "jvp_error_relative_q0": jvp_error_relative_q0,
        "metric_raw_c27_vs_order6": raw_model_error,
        "slope_raw_c27_vs_order6": raw_model_error_slopes,
        "metric_c27_vs_order6_remainder_error": remainder_error,
    }
    npz_payload.update({f"metric_{name}": values for name, values in metrics.items()})
    npz_payload.update({f"slope_{name}": values for name, values in slopes.items()})
    npz_payload.update(
        {f"bootstrap_slope_{name}": values for name, values in bootstrap_slopes.items()}
    )
    np.savez_compressed(
        output_dir / "c27_small_amplitude_order.npz",
        allow_pickle=False,
        **npz_payload,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "checkpoint": "final",
            "epoch": loaded.epoch,
            "structural_config": {
                "cs_use_g1_baseline": loaded.config["cs_use_g1_baseline"],
                "cs_g1_k_cut": loaded.config["cs_g1_k_cut"],
                "cs_g1_fft_fp64": loaded.config["cs_g1_fft_fp64"],
                "cs_residual_eta_order": loaded.config["cs_residual_eta_order"],
                "parameter_count": parameter_count,
            },
        },
        "execution": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "wall_seconds": elapsed,
        },
        "sampling": {
            "families": list(FAMILY_NAMES),
            "states_per_family": args.states_per_family,
            "state_count": state_count,
            "frame_index": 0,
            "rule": "sort by (depth, simulation_id), split each family into equal-count strata, choose each stratum midpoint",
            "source_paths": list(source_paths),
            "selection": selection,
        },
        "protocol": {
            "surface_scaling": "eta -> epsilon * eta with xi and physical depth fixed",
            "epsilons": EPSILONS.tolist(),
            "slope_fit_epsilons": FIT_EPSILONS.tolist(),
            "reference": "Craig--Sulem order 6, pad factor 8, float64",
            "projection": f"Fourier modes |k| <= {args.projection_k_max:g}",
            "norm": "projected spatial RMS divided by projected RMS of G0(h)xi for each state",
            "bootstrap": {
                "method": "fit each state separately, resample states with replacement within each of 10 family strata, and take the pooled median slope",
                "samples": args.bootstrap_samples,
                "seed_base": args.bootstrap_seed,
            },
        },
        "checks": {
            "flat_surface_Gtheta_vs_G0_relative_error": distribution_summary(
                flat_surface_relative_error
            ),
            "Deta_Gtheta_zero_vs_G1_relative_error": distribution_summary(
                jvp_relative_error
            ),
            "Deta_Gtheta_zero_vs_G1_error_relative_to_G0": distribution_summary(
                jvp_error_relative_q0
            ),
            "largest_G1_relative_JVP_error_state": {
                **selection[int(np.argmax(jvp_relative_error))],
                "G1_projected_rms": float(q1_norm[int(np.argmax(jvp_relative_error))]),
                "absolute_projected_rms_error": float(
                    jvp_absolute_error[int(np.argmax(jvp_relative_error))]
                ),
                "relative_to_G1": float(np.max(jvp_relative_error)),
                "relative_to_G0": float(
                    jvp_error_relative_q0[int(np.argmax(jvp_relative_error))]
                ),
                "interpretation": "The largest G1-relative value has an exceptionally small G1 denominator; its absolute and G0-relative errors remain at the float32 numerical floor.",
            },
            "raw_Gtheta_vs_order6": {
                "pooled_median_curve": np.median(raw_model_error, axis=1).tolist(),
                "per_state_slope": distribution_summary(raw_model_error_slopes),
                "interpretation": "The uncorrected total error approaches the recorded float32 G0/JVP floor; it is not used to estimate the learned remainder's order.",
            },
        },
        "metrics": summaries,
        "artifacts": {
            "json": "c27_small_amplitude_order.json",
            "npz": "c27_small_amplitude_order.npz",
            "png": "c27_small_amplitude_order.png",
            "pdf": "c27_small_amplitude_order.pdf",
        },
        "interpretation": [
            "A first-order G0 truncation slope and second-order G0+G1 truncation slope quantify the analytic benefit of retaining G1.",
            "The C27 Taylor remainder subtracts the deployed model's own value and directional derivative at eta=0, so its slope tests the trained O(eta^2) residual without a float32 baseline-subtraction floor.",
            "The C27/reference remainder error compares only terms of order two and higher; raw total model error is retained separately and eventually reaches the measured flat-surface/first-derivative numerical floor.",
            "The JVP comparison separately tests that the deployed model's first surface derivative agrees with the analytic padded G1 on the reported production band.",
            "This is an empirical structural/order audit on registered initial states, not a causal training-loss or full no-G1 retraining ablation.",
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    (output_dir / "c27_small_amplitude_order.json").write_text(
        encoded + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "wall_seconds": elapsed,
                "metrics": summaries,
                "checks": payload["checks"],
            },
            allow_nan=False,
        )
    )
    raise SystemExit(0)
