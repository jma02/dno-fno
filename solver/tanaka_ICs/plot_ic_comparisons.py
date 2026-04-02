from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..data.solitary_loader_jax import DEFAULT_SOLITON_ROOT, load_soliton_file
from .modified_tanaka import (
    REAL_DTYPE,
    ModifiedTanakaBatchSolution,
    ModifiedTanakaParams,
    ModifiedTanakaSolution,
    solve_modified_tanaka,
    solve_modified_tanaka_batched,
    solve_tanaka_branch,
)
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Tanaka IC comparisons against the provided solitary-wave data.")
    parser.add_argument("--cases", nargs="+", default=["coll.anim_s005", "coll.anim_s01", "coll.anim_s02", "coll.anim_s04"])
    parser.add_argument("--soliton_root", default=str(DEFAULT_SOLITON_ROOT))
    parser.add_argument("--output_dir", default="outputs/tanaka_ic_comparisons")
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--grid_mode", choices=("auto", "manual"), default="manual")
    parser.add_argument("--collocation_points", type=int, default=257)
    parser.add_argument("--quadrature_substeps", type=int, default=4)
    parser.add_argument("--interpolation_degree", type=int, default=3)
    parser.add_argument("--s_max", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--transform_power", type=int, default=5)
    parser.add_argument("--qc_lower", type=float, default=0.2)
    parser.add_argument("--qc_upper", type=float, default=0.999)
    parser.add_argument("--outer_iterations", type=int, default=24)
    parser.add_argument("--fixed_point_iterations", type=int, default=80)
    parser.add_argument("--f2_tolerance", type=float, default=1e-10)
    parser.add_argument("--continuation", action="store_true")
    return parser.parse_args()


def _load_initial_snapshot(path: Path) -> dict[str, np.ndarray | float | str]:
    trajectory = load_soliton_file(path)
    eta = np.asarray(trajectory["eta"][0], dtype=np.float64)
    xi = np.asarray(trajectory["xi"][0], dtype=np.float64)
    gxi = np.asarray(trajectory["gxi"][0], dtype=np.float64)
    x = np.asarray(trajectory["x"], dtype=np.float64)
    crest_index = int(np.argmax(eta))
    return {
        "name": str(trajectory["name"]),
        "x": x,
        "eta": eta,
        "xi": xi,
        "gxi": gxi,
        "amplitude": float(np.max(eta)),
        "center": float(x[crest_index]),
        "nx": int(trajectory["nx"]),
        "length": float(trajectory["length"]),
    }


def _resolve_case_path(root: Path, case_name: str) -> Path:
    direct = root / case_name
    if direct.exists():
        return direct
    prefixed = root / f"coll.anim_{case_name}"
    return prefixed


def _infer_family(case_name: str) -> str:
    stem = Path(case_name).name.replace("coll.anim_", "")
    return stem[0]


def _relative_l2(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-16) -> float:
    return float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), eps))


def _objective(snapshot: dict[str, np.ndarray | float | str], pred_eta: np.ndarray, pred_xi: np.ndarray, pred_gxi: np.ndarray) -> float:
    xi_true = np.asarray(snapshot["xi"], dtype=np.float64)
    xi_true_zero = xi_true - xi_true.mean()
    xi_pred_zero = pred_xi - pred_xi.mean()
    return (
        _relative_l2(pred_eta, np.asarray(snapshot["eta"], dtype=np.float64))
        + _relative_l2(xi_pred_zero, xi_true_zero)
        + _relative_l2(pred_gxi, np.asarray(snapshot["gxi"], dtype=np.float64))
    )


def _build_params(snapshot: dict[str, np.ndarray | float | str], args: argparse.Namespace, amplitude: float, center: float, direction: int) -> ModifiedTanakaParams:
    return ModifiedTanakaParams(
        amplitude=float(amplitude),
        depth=args.depth,
        gravity=args.gravity,
        direction=direction,
        nx=int(snapshot["nx"]),
        length=float(snapshot["length"]),
        center=float(center),
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        grid_mode=args.grid_mode,
        collocation_points=args.collocation_points,
        quadrature_substeps=args.quadrature_substeps,
        interpolation_degree=args.interpolation_degree,
        s_max=args.s_max,
        alpha=args.alpha,
        transform_power=args.transform_power,
        qc_lower=args.qc_lower,
        qc_upper=args.qc_upper,
        outer_iterations=args.outer_iterations,
        fixed_point_iterations=args.fixed_point_iterations,
        f2_tolerance=args.f2_tolerance,
    )


def _periodic_index_distance(i: int, j: int, n: int) -> int:
    delta = abs(i - j)
    return min(delta, n - delta)


def _select_crest_indices(eta: np.ndarray, n_components: int, min_separation: int) -> list[int]:
    local_maxima = np.where((eta >= np.roll(eta, 1)) & (eta > np.roll(eta, -1)))[0]
    ordered = local_maxima[np.argsort(eta[local_maxima])[::-1]]

    selected: list[int] = []
    for idx in ordered:
        if all(_periodic_index_distance(int(idx), prev, eta.size) >= min_separation for prev in selected):
            selected.append(int(idx))
        if len(selected) == n_components:
            return selected

    fallback = np.argsort(eta)[::-1]
    for idx in fallback:
        if all(_periodic_index_distance(int(idx), prev, eta.size) >= min_separation for prev in selected):
            selected.append(int(idx))
        if len(selected) == n_components:
            break
    return selected


def _assemble_model(snapshot: dict[str, np.ndarray | float | str], family: str, components: list[dict[str, object]]) -> dict[str, object]:
    pred_eta = np.sum([np.asarray(component["solution"].eta_periodic, dtype=np.float64) for component in components], axis=0)
    pred_xi = np.sum([np.asarray(component["solution"].xi_periodic, dtype=np.float64) for component in components], axis=0)
    pred_gxi = np.sum([np.asarray(component["solution"].gxi_periodic, dtype=np.float64) for component in components], axis=0)
    pred_xi = pred_xi - pred_xi.mean() + float(np.mean(np.asarray(snapshot["xi"], dtype=np.float64)))

    score = _objective(snapshot, pred_eta, pred_xi, pred_gxi)
    return {
        "family": family,
        "pred_eta": pred_eta,
        "pred_xi": pred_xi,
        "pred_gxi": pred_gxi,
        "score": float(score),
        "components": [
            {
                "amplitude": float(component["amplitude"]),
                "center": float(component["center"]),
                "direction": int(component["direction"]),
                "qc": float(component["solution"].qc),
                "froude": float(component["solution"].froude),
                "speed": float(component["solution"].speed),
            }
            for component in components
        ],
    }


def _components_from_batch(
    amplitudes: list[float],
    centers: list[float],
    directions: list[int],
    batch: ModifiedTanakaBatchSolution,
) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for idx, (amplitude, center, direction) in enumerate(zip(amplitudes, centers, directions)):
        solution = ModifiedTanakaSolution(
            amplitude=float(amplitude),
            qc=float(batch.qc[idx]),
            froude=float(batch.froude[idx]),
            speed=float(batch.speed[idx]),
            tau_profile=batch.tau_profile[idx],
            x_profile=batch.x_profile[idx],
            eta_profile=batch.eta_profile[idx],
            phi_profile=batch.phi_profile,
            q_profile=batch.q_profile[idx],
            theta_profile=batch.theta_profile[idx],
            x_periodic=batch.x_periodic,
            eta_periodic=batch.eta_periodic[idx],
            xi_periodic=batch.xi_periodic[idx],
            gxi_periodic=batch.gxi_periodic[idx],
        )
        components.append(
            {
                "amplitude": float(amplitude),
                "center": float(center),
                "direction": int(direction),
                "solution": solution,
            }
        )
    return components


def _fit_case(snapshot: dict[str, np.ndarray | float | str], args: argparse.Namespace) -> dict[str, object]:
    family = _infer_family(str(snapshot["name"]))
    eta = np.asarray(snapshot["eta"], dtype=np.float64)
    x = np.asarray(snapshot["x"], dtype=np.float64)

    if family == "s":
        template_params = _build_params(snapshot, args, 0.0, 0.0, 1)
        amplitudes = [float(snapshot["amplitude"]), float(snapshot["amplitude"])]
        centers = [float(snapshot["center"]), float(snapshot["center"])]
        directions = [-1, 1]
        batch = solve_modified_tanaka_batched(
            template_params,
            jnp.asarray(amplitudes, dtype=REAL_DTYPE),
            centers=jnp.asarray(centers, dtype=REAL_DTYPE),
            directions=jnp.asarray(directions, dtype=REAL_DTYPE),
        )
        components = _components_from_batch(amplitudes, centers, directions, batch)
        candidate_models = [
            _assemble_model(snapshot, family, [components[0]]),
            _assemble_model(snapshot, family, [components[1]]),
        ]
        return min(candidate_models, key=lambda model: model["score"])

    crest_indices = _select_crest_indices(eta, n_components=2, min_separation=max(32, eta.size // 12))
    crest_indices = sorted(crest_indices, key=lambda idx: x[idx])
    crest_amplitudes = [float(eta[idx]) for idx in crest_indices]
    crest_centers = [float(x[idx]) for idx in crest_indices]
    amplitudes = [crest_amplitudes[0], crest_amplitudes[0], crest_amplitudes[1], crest_amplitudes[1]]
    centers = [crest_centers[0], crest_centers[0], crest_centers[1], crest_centers[1]]
    directions = [-1, 1, -1, 1]
    template_params = _build_params(snapshot, args, 0.0, 0.0, 1)
    batch = solve_modified_tanaka_batched(
        template_params,
        jnp.asarray(amplitudes, dtype=REAL_DTYPE),
        centers=jnp.asarray(centers, dtype=REAL_DTYPE),
        directions=jnp.asarray(directions, dtype=REAL_DTYPE),
    )
    solved_components = _components_from_batch(amplitudes, centers, directions, batch)
    candidates = [
        {
            -1: solved_components[2 * crest_idx],
            1: solved_components[2 * crest_idx + 1],
        }
        for crest_idx in range(2)
    ]

    if family == "h":
        sign_combinations = [(1, -1), (-1, 1)]
    elif family == "f":
        sign_combinations = [(1, 1), (-1, -1)]
    else:
        sign_combinations = list(product((-1, 1), repeat=2))

    models = [
        _assemble_model(snapshot, family, [candidates[0][signs[0]], candidates[1][signs[1]]])
        for signs in sign_combinations
    ]
    return min(models, key=lambda model: model["score"])


def _make_summary(snapshot: dict[str, np.ndarray | float | str], model: dict[str, object]) -> dict[str, object]:
    eta_true = np.asarray(snapshot["eta"], dtype=np.float64)
    xi_true = np.asarray(snapshot["xi"], dtype=np.float64)
    gxi_true = np.asarray(snapshot["gxi"], dtype=np.float64)

    eta_pred = np.asarray(model["pred_eta"], dtype=np.float64)
    xi_pred = np.asarray(model["pred_xi"], dtype=np.float64)
    gxi_pred = np.asarray(model["pred_gxi"], dtype=np.float64)

    xi_true_zero = xi_true - xi_true.mean()
    xi_pred_zero = xi_pred - xi_pred.mean()

    components = list(model["components"])
    return {
        "name": str(snapshot["name"]),
        "family": str(model["family"]),
        "n_components": len(components),
        "amplitude": float(snapshot["amplitude"]),
        "center": float(snapshot["center"]),
        "component_amplitudes": [float(component["amplitude"]) for component in components],
        "component_centers": [float(component["center"]) for component in components],
        "component_directions": [int(component["direction"]) for component in components],
        "component_qc": [float(component["qc"]) for component in components],
        "component_froude": [float(component["froude"]) for component in components],
        "component_speed": [float(component["speed"]) for component in components],
        "eta_rel_l2": _relative_l2(eta_pred, eta_true),
        "xi_zero_mean_rel_l2": _relative_l2(xi_pred_zero, xi_true_zero),
        "gxi_rel_l2": _relative_l2(gxi_pred, gxi_true),
        "score": float(model["score"]),
    }


def _plot_case(snapshot: dict[str, np.ndarray | float | str], model: dict[str, object], png_path: Path, grid_mode: str) -> dict[str, object]:
    x = np.asarray(snapshot["x"], dtype=np.float64)
    eta_true = np.asarray(snapshot["eta"], dtype=np.float64)
    xi_true = np.asarray(snapshot["xi"], dtype=np.float64)
    gxi_true = np.asarray(snapshot["gxi"], dtype=np.float64)

    eta_pred = np.asarray(model["pred_eta"], dtype=np.float64)
    xi_pred = np.asarray(model["pred_xi"], dtype=np.float64)
    gxi_pred = np.asarray(model["pred_gxi"], dtype=np.float64)

    xi_true_zero = xi_true - xi_true.mean()
    xi_pred_zero = xi_pred - xi_pred.mean()

    summary = _make_summary(snapshot, model)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(x, eta_true, color="tab:blue", linewidth=2.2, label="true")
    axes[0, 0].plot(x, eta_pred, color="tab:red", linewidth=1.8, linestyle="--", label="tanaka")
    axes[0, 0].set_title(rf"$\eta(x)$ | rel L2 = {summary['eta_rel_l2']:.3e}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, xi_true_zero, color="tab:blue", linewidth=2.2, label="true")
    axes[0, 1].plot(x, xi_pred_zero, color="tab:red", linewidth=1.8, linestyle="--", label="tanaka")
    axes[0, 1].set_title(rf"$\xi(x)-\langle \xi \rangle$ | rel L2 = {summary['xi_zero_mean_rel_l2']:.3e}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x, gxi_true, color="tab:blue", linewidth=2.2, label="true")
    axes[1, 0].plot(x, gxi_pred, color="tab:red", linewidth=1.8, linestyle="--", label="tanaka")
    axes[1, 0].set_title(rf"$G(\eta)\xi$ | rel L2 = {summary['gxi_rel_l2']:.3e}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "\n".join(
            [
                str(snapshot["name"]),
                f"family = {summary['family']}",
                f"n_components = {summary['n_components']}",
                f"amplitude = {summary['amplitude']:.6f}",
                f"center = {summary['center']:.6f}",
                f"grid_mode = {grid_mode}",
                f"nx = {eta_pred.shape[0]}",
            ]
            + [
                (
                    f"c{component_index + 1}: "
                    f"a={summary['component_amplitudes'][component_index]:.6f} "
                    f"x={summary['component_centers'][component_index]:.6f} "
                    f"dir={summary['component_directions'][component_index]} "
                    f"qc={summary['component_qc'][component_index]:.6f}"
                )
                for component_index in range(summary["n_components"])
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    soliton_root = Path(args.soliton_root).expanduser().resolve()

    summaries: list[dict[str, float | str]] = []

    snapshots = [_load_initial_snapshot(_resolve_case_path(soliton_root, case_name)) for case_name in args.cases]

    if args.continuation and all(_infer_family(str(snapshot["name"])) == "s" for snapshot in snapshots):
        common_params = _build_params(
            snapshots[0],
            args,
            float(snapshots[0]["amplitude"]),
            float(snapshots[0]["center"]),
            args.direction,
        )
        amplitudes = [float(snapshot["amplitude"]) for snapshot in snapshots]
        branch = solve_tanaka_branch(amplitudes, common_params)
        models = [
            _assemble_model(
                snapshot,
                "s",
                [{"amplitude": float(snapshot["amplitude"]), "center": float(snapshot["center"]), "direction": args.direction, "solution": solution}],
            )
            for snapshot, solution in zip(snapshots, branch)
        ]
    else:
        models = [_fit_case(snapshot, args) for snapshot in snapshots]

    for snapshot, model in zip(snapshots, models):
        stem = Path(str(snapshot["name"])).name.replace("coll.anim_", "")
        png_path = output_dir / f"{stem}.png"
        npz_path = output_dir / f"{stem}.npz"
        json_path = output_dir / f"{stem}.json"

        summary = _plot_case(snapshot, model, png_path, args.grid_mode)
        summaries.append(summary)

        np.savez_compressed(
            npz_path,
            x=np.asarray(snapshot["x"], dtype=np.float64),
            truth_eta=np.asarray(snapshot["eta"], dtype=np.float64),
            truth_xi=np.asarray(snapshot["xi"], dtype=np.float64),
            truth_gxi=np.asarray(snapshot["gxi"], dtype=np.float64),
            pred_eta=np.asarray(model["pred_eta"], dtype=np.float64),
            pred_xi=np.asarray(model["pred_xi"], dtype=np.float64),
            pred_gxi=np.asarray(model["pred_gxi"], dtype=np.float64),
            params_json=json.dumps(vars(args)),
            components_json=json.dumps(model["components"]),
            summary_json=json.dumps(summary),
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
