"""Probe C25 GL2 Picard convergence on fixed saved Tanaka states.

This is a diagnostic replay, not a rollout.  At every saved state of the
seven predeclared C25 focus cases, it advances one production-size inner step
(``dt=0.01``) with four and eight Picard sweeps.  It records the last two
stage-update residuals and the difference between the resulting one-step
states.  The archived state is never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import (
    _gl2_if_step_with_residual,
    build_predict_gxi_batched,
    load_run,
)
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol


FOCUS_CASES: tuple[tuple[str, int], ...] = (
    ("tanaka_g0", 4),
    ("tanaka_g0", 21),
    ("tanaka_g0", 27),
    ("tanaka_g1", 11),
    ("tanaka_g1", 16),
    ("tanaka_g1", 24),
    ("tanaka_g1", 29),
)


def _combined_norm(eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.sum(np.square(eta), axis=-1)
        + np.sum(np.square(xi), axis=-1)
    )


def _load_focus_states(
    eval_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for output_index, (regime, case_index) in enumerate(FOCUS_CASES):
        grouped.setdefault(regime, []).append((output_index, case_index))

    times: np.ndarray | None = None
    eta_columns: list[np.ndarray | None] = [None] * len(FOCUS_CASES)
    xi_columns: list[np.ndarray | None] = [None] * len(FOCUS_CASES)
    depths = np.empty(len(FOCUS_CASES), dtype=np.float64)
    case_ids = np.empty(len(FOCUS_CASES), dtype=np.int64)
    for regime, selections in grouped.items():
        archive_path = eval_dir / regime / f"{regime}_trajs.npz"
        with np.load(archive_path) as archive:
            regime_times = np.asarray(archive["times"], dtype=np.float64)
            if times is None:
                times = regime_times
            elif not np.array_equal(times, regime_times):
                raise ValueError(f"time grid mismatch in {archive_path}")
            for output_index, case_index in selections:
                eta_columns[output_index] = np.asarray(
                    archive["pred_eta"][:, case_index], dtype=np.float64,
                )
                xi_columns[output_index] = np.asarray(
                    archive["pred_xi"][:, case_index], dtype=np.float64,
                )
                depths[output_index] = float(archive["depths"][case_index])
                case_ids[output_index] = int(archive["case_ids"][case_index])

    if times is None or any(value is None for value in eta_columns + xi_columns):
        raise RuntimeError("failed to load every focus case")
    eta = np.stack(eta_columns, axis=1)
    xi = np.stack(xi_columns, axis=1)
    return times, eta, xi, depths, case_ids


def _summary_record(
    *,
    case_position: int,
    case_id: int,
    depth: float,
    times: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    def maximum_with_time(name: str) -> dict[str, float | int]:
        values = arrays[name][:, case_position]
        index = int(np.nanargmax(values))
        return {
            "value": float(values[index]),
            "frame": index,
            "time": float(times[index]),
        }

    r4 = arrays["r4"][:, case_position]
    r3 = arrays["r3"][:, case_position]
    contraction = r4 / (r3 + 1e-300)
    return {
        "regime": FOCUS_CASES[case_position][0],
        "case_index": int(FOCUS_CASES[case_position][1]),
        "case_id": int(case_id),
        "depth": float(depth),
        "finite_all": bool(
            all(np.isfinite(value[:, case_position]).all() for value in arrays.values())
        ),
        "r4_max": maximum_with_time("r4"),
        "r4_p95": float(np.percentile(r4, 95.0)),
        "r4_median": float(np.median(r4)),
        "r8_max": maximum_with_time("r8"),
        "r4_over_r3_max": float(np.max(contraction)),
        "r4_over_r3_p95": float(np.percentile(contraction, 95.0)),
        "r4_ge_1e-6_count": int(np.count_nonzero(r4 >= 1e-6)),
        "r4_ge_1e-4_count": int(np.count_nonzero(r4 >= 1e-4)),
        "r4_ge_1e-2_count": int(np.count_nonzero(r4 >= 1e-2)),
        "four_vs_eight_over_step_max": maximum_with_time(
            "four_vs_eight_over_step",
        ),
        "four_vs_eight_over_state_max": maximum_with_time(
            "four_vs_eight_over_state",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/c25_capacity125_full_20260716_022603",
    )
    parser.add_argument(
        "--eval-dir",
        default=(
            "outputs/c25_capacity125_full_20260716_022603/"
            "eval_best_soliton_spectral_guard_20260717_133536"
        ),
    )
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument(
        "--output-prefix",
        default="notes/c25_gl2_residual_probe_20260717",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    run_dir = Path(args.run_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    times, pred_eta, pred_xi, depths, case_ids = _load_focus_states(eval_dir)
    n_frames, n_cases, nx = pred_eta.shape
    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    predict_batched = build_predict_gxi_batched(loaded)

    _, k_numpy = build_grid(nx, args.length)
    k = jnp.asarray(k_numpy, dtype=jnp.float64)
    depth_2d = jnp.asarray(depths, dtype=jnp.float64)[:, None]
    log_depth = jnp.asarray(np.log(np.maximum(depths, 1e-12)), dtype=jnp.float32)
    g0 = make_linear_dno_symbol(k, depth_2d)
    params = ti.SolverParams(
        nx=nx,
        length=float(args.length),
        depth=depth_2d,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        filter_fraction=0.25,
        k=k,
        g0=g0,
    )

    def predict(eta: jax.Array, xi: jax.Array) -> jax.Array:
        return predict_batched(
            eta.astype(jnp.float32),
            xi.astype(jnp.float32),
            log_depth,
        ).astype(jnp.float64)

    def step(
        state: ti.State,
        time: jax.Array,
        iterations: int,
    ) -> tuple[ti.State, jax.Array, jax.Array]:
        return _gl2_if_step_with_residual(
            state,
            time,
            jnp.asarray(args.dt, dtype=jnp.float64),
            params,
            predict,
            iterations=iterations,
            filter_shape="hard",
            houli_a=36.0,
            houli_m=36.0,
            cascade_gate_enabled=False,
            cascade_k_cut=32.0,
            cascade_r_threshold=1e-3,
            cascade_sharpness=10.0,
            cascade_houli_a=0.69,
            cascade_houli_m=4.0,
            cascade_k_eff=128.0,
            cascade_filter_xi=True,
        )

    step4 = jax.jit(lambda state, time: step(state, time, 4))
    step8 = jax.jit(lambda state, time: step(state, time, 8))
    arrays = {
        name: np.empty((n_frames, n_cases), dtype=np.float64)
        for name in (
            "r4",
            "r3",
            "r8",
            "r7",
            "four_vs_eight_over_step",
            "four_vs_eight_over_state",
            "eta_four_vs_eight_over_step",
            "xi_four_vs_eight_over_step",
        )
    }

    print(
        f"probing {n_cases} cases x {n_frames} saved states on {jax.devices()}",
        flush=True,
    )
    for frame in range(n_frames):
        state = ti.project_zero_mean_xi(
            ti.State(
                eta=jnp.asarray(pred_eta[frame], dtype=jnp.float64),
                xi=jnp.asarray(pred_xi[frame], dtype=jnp.float64),
            )
        )
        time = jnp.asarray(times[frame], dtype=jnp.float64)
        state4, r4, r3 = step4(state, time)
        state8, r8, r7 = step8(state, time)
        state4 = ti.project_zero_mean_xi(state4)
        state8 = ti.project_zero_mean_xi(state8)

        state_eta = np.asarray(state.eta)
        state_xi = np.asarray(state.xi)
        eta4 = np.asarray(state4.eta)
        xi4 = np.asarray(state4.xi)
        eta8 = np.asarray(state8.eta)
        xi8 = np.asarray(state8.xi)
        step_eta = eta8 - state_eta
        step_xi = xi8 - state_xi
        diff_eta = eta4 - eta8
        diff_xi = xi4 - xi8
        step_norm = _combined_norm(step_eta, step_xi)
        diff_norm = _combined_norm(diff_eta, diff_xi)
        state_norm = _combined_norm(eta8, xi8)

        arrays["r4"][frame] = np.asarray(r4)
        arrays["r3"][frame] = np.asarray(r3)
        arrays["r8"][frame] = np.asarray(r8)
        arrays["r7"][frame] = np.asarray(r7)
        arrays["four_vs_eight_over_step"][frame] = diff_norm / (
            step_norm + 1e-300
        )
        arrays["four_vs_eight_over_state"][frame] = diff_norm / (
            state_norm + 1e-300
        )
        arrays["eta_four_vs_eight_over_step"][frame] = np.linalg.norm(
            diff_eta, axis=-1,
        ) / (np.linalg.norm(step_eta, axis=-1) + 1e-300)
        arrays["xi_four_vs_eight_over_step"][frame] = np.linalg.norm(
            diff_xi, axis=-1,
        ) / (np.linalg.norm(step_xi, axis=-1) + 1e-300)

        if frame == 0 or (frame + 1) % args.progress_every == 0 or frame + 1 == n_frames:
            print(
                f"frame {frame + 1:3d}/{n_frames}: "
                f"max r4={np.max(arrays['r4'][frame]):.3e}, "
                "max |step4-step8|/|step8-step0|="
                f"{np.max(arrays['four_vs_eight_over_step'][frame]):.3e}",
                flush=True,
            )

    records = [
        _summary_record(
            case_position=position,
            case_id=int(case_ids[position]),
            depth=float(depths[position]),
            times=times,
            arrays=arrays,
        )
        for position in range(n_cases)
    ]
    payload = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(loaded.epoch),
        "dt": float(args.dt),
        "filter_fraction": 0.25,
        "filter_shape": "hard",
        "four_picard_sweeps": 4,
        "reference_picard_sweeps": 8,
        "n_frames": int(n_frames),
        "n_cases": int(n_cases),
        "devices": [str(device) for device in jax.devices()],
        "cases": records,
    }
    json_path = output_prefix.with_suffix(".json")
    npz_path = output_prefix.with_suffix(".npz")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        times=times,
        regimes=np.asarray([regime for regime, _ in FOCUS_CASES]),
        case_indices=np.asarray([index for _, index in FOCUS_CASES], dtype=np.int64),
        case_ids=case_ids,
        depths=depths,
        **arrays,
    )
    print(f"saved {json_path}", flush=True)
    print(f"saved {npz_path}", flush=True)


if __name__ == "__main__":
    main()
