"""Phase-only oracle for diagnosing secular translation error.

This module does not change the deployed Craig--Sulem backbone.  It uses the
order-six reference DNO only as a diagnostic: at each learned-rollout stage it
replaces the component of the learned ``q = G(eta) xi`` defect that changes the
instantaneous Fourier phase of ``eta``.  The learned amplitude-growth component
and every other part of the vector field are retained.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

from solver.evals.model_rollout import (
    build_predict_gxi_batched,
    load_run,
    rollout_surrogate,
)
from solver.solvers import time_integrator as ti
from solver.solvers.dno_series_jax import (
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/c21_tangent_w100_from_c20_20260714_171442"
DEFAULT_EVAL_DIR = DEFAULT_RUN_DIR / "eval_final_soliton_spectral_guard_20260714_175241"
DEFAULT_CASE_IDS = (4, 27, 1_000_002, 1_000_011, 1_000_022, 1_000_024)
MULTI_ARM_NAMES = (
    "modal",
    "localized_sigma1",
    "localized_sigma4",
    "radial_complement",
    "full_defect",
)


class FixedTailPanel(NamedTuple):
    """Fixed initial states, reference trajectories, and C21 controls."""

    times: np.ndarray
    depths: np.ndarray
    case_ids: np.ndarray
    truth_eta: np.ndarray
    truth_xi: np.ndarray
    truth_gxi: np.ndarray
    baseline_eta: np.ndarray
    baseline_xi: np.ndarray
    baseline_gxi: np.ndarray


def modal_phase_component(
    eta: jax.Array,
    defect: jax.Array,
    k: jax.Array,
    *,
    k_max: float,
    relative_energy_floor: float,
) -> jax.Array:
    """Project ``defect`` onto the instantaneous modal phase direction.

    For every active nonzero Fourier mode, the real two-dimensional plane is
    decomposed into the radial direction ``eta_hat`` and phase direction
    ``i eta_hat``.  Only the latter is returned.  The activity threshold is
    target-only: it depends on ``eta`` and never on the learned defect.
    """
    eta_hat = jnp.fft.fft(eta, axis=-1)
    defect_centered = defect - jnp.mean(defect, axis=-1, keepdims=True)
    defect_hat = jnp.fft.fft(defect_centered, axis=-1)
    eta_energy = jnp.abs(eta_hat) ** 2
    retained = jnp.logical_and(jnp.abs(k) > 0.0, jnp.abs(k) <= k_max)
    retained_energy = jnp.where(retained, eta_energy, 0.0)
    energy_floor = jnp.asarray(relative_energy_floor, dtype=eta_energy.dtype) * jnp.max(
        retained_energy, axis=-1, keepdims=True
    )
    active = jnp.logical_and(retained, eta_energy > energy_floor)
    phase_coefficient = jnp.where(
        active,
        jnp.imag(defect_hat * jnp.conj(eta_hat))
        / jnp.maximum(eta_energy, jnp.asarray(1e-30, dtype=eta_energy.dtype)),
        0.0,
    )
    component_hat = 1j * eta_hat * phase_coefficient
    component = jnp.real(jnp.fft.ifft(component_hat, axis=-1))
    return component - jnp.mean(component, axis=-1, keepdims=True)


def localized_translation_component(
    eta: jax.Array,
    defect: jax.Array,
    k: jax.Array,
    depth: jax.Array,
    *,
    window_depths: float,
    energy_floor_relative: float,
) -> jax.Array:
    """Return the local least-squares component of ``defect`` along ``eta_x``.

    The defect convention is ``reference - learned``.  Consequently a rigid
    defect ``defect = a * eta_x`` produces the correction ``a * eta_x`` (up to
    the requested denominator floor), with no additional sign change.  The
    final zero-mean projection matches the DNO gauge used by rollout.
    """
    eta_hat = jnp.fft.fft(eta, axis=-1)
    eta_x = jnp.real(jnp.fft.ifft(1j * k * eta_hat, axis=-1))
    defect_centered = defect - jnp.mean(defect, axis=-1, keepdims=True)
    sigma = jnp.asarray(window_depths, dtype=eta.dtype) * jnp.asarray(
        depth, dtype=eta.dtype
    ).reshape(-1, 1)
    smoothing_multiplier = jnp.exp(-0.5 * (sigma * jnp.abs(k)[None, :]) ** 2)

    def smooth(field: jax.Array) -> jax.Array:
        return jnp.real(
            jnp.fft.ifft(
                jnp.fft.fft(field, axis=-1) * smoothing_multiplier,
                axis=-1,
            )
        )

    local_cross = smooth(eta_x * defect_centered)
    local_energy = jnp.maximum(smooth(eta_x**2), 0.0)
    energy_floor = jnp.asarray(energy_floor_relative, dtype=eta.dtype) * jnp.max(
        local_energy, axis=-1, keepdims=True
    )
    coefficient = local_cross / (
        local_energy + energy_floor + jnp.asarray(1e-30, dtype=eta.dtype)
    )
    component = eta_x * coefficient
    return component - jnp.mean(component, axis=-1, keepdims=True)


def _load_fixed_tail_panel(
    eval_dir: Path,
    case_ids: tuple[int, ...],
    max_time: float | None,
) -> FixedTailPanel:
    arrays: dict[str, list[np.ndarray]] = {
        "depths": [],
        "case_ids": [],
        "truth_eta": [],
        "truth_xi": [],
        "truth_gxi": [],
        "baseline_eta": [],
        "baseline_xi": [],
        "baseline_gxi": [],
    }
    common_times: np.ndarray | None = None
    found: set[int] = set()
    for regime in ("tanaka_g0", "tanaka_g1"):
        path = eval_dir / regime / f"{regime}_trajs.npz"
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            if common_times is None:
                common_times = times
            elif not np.array_equal(common_times, times):
                raise ValueError(f"time grid mismatch in {path}")
            archive_ids = np.asarray(archive["case_ids"], dtype=np.int64)
            indices = [
                int(np.flatnonzero(archive_ids == case_id)[0])
                for case_id in case_ids
                if np.count_nonzero(archive_ids == case_id) == 1
            ]
            found.update(int(archive_ids[index]) for index in indices)
            if not indices:
                continue
            arrays["depths"].append(np.asarray(archive["depths"])[indices])
            arrays["case_ids"].append(archive_ids[indices])
            for prefix, archive_prefix in (("truth", "truth"), ("baseline", "pred")):
                for field in ("eta", "xi", "gxi"):
                    arrays[f"{prefix}_{field}"].append(
                        np.asarray(archive[f"{archive_prefix}_{field}"])[:, indices]
                    )
    missing = set(case_ids) - found
    if missing:
        raise ValueError(
            f"requested case IDs not found exactly once: {sorted(missing)}"
        )
    if common_times is None:
        raise ValueError(f"no trajectory archives found under {eval_dir}")
    time_mask = (
        np.ones(common_times.shape, dtype=bool)
        if max_time is None
        else common_times <= max_time + 1e-6 * max(1.0, abs(float(max_time)))
    )
    if np.count_nonzero(time_mask) < 2:
        raise ValueError("the selected time horizon must contain at least two frames")
    return FixedTailPanel(
        times=common_times[time_mask],
        depths=np.concatenate(arrays["depths"]).astype(np.float64),
        case_ids=np.concatenate(arrays["case_ids"]).astype(np.int64),
        truth_eta=np.concatenate(arrays["truth_eta"], axis=1)[time_mask],
        truth_xi=np.concatenate(arrays["truth_xi"], axis=1)[time_mask],
        truth_gxi=np.concatenate(arrays["truth_gxi"], axis=1)[time_mask],
        baseline_eta=np.concatenate(arrays["baseline_eta"], axis=1)[time_mask],
        baseline_xi=np.concatenate(arrays["baseline_xi"], axis=1)[time_mask],
        baseline_gxi=np.concatenate(arrays["baseline_gxi"], axis=1)[time_mask],
    )


def _periodic_shift(
    values: np.ndarray, displacement: float, length: float
) -> np.ndarray:
    nx = values.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(np.fft.fft(values) * np.exp(-1j * k * displacement)).real


def _optimal_displacement(
    prediction: np.ndarray,
    truth: np.ndarray,
    length: float,
) -> float:
    nx = truth.size
    dx = length / nx
    correlation = np.fft.ifft(np.fft.fft(prediction) * np.conj(np.fft.fft(truth))).real
    peak_indices = np.argpartition(correlation, -8)[-8:]

    def objective(displacement: float) -> float:
        difference = _periodic_shift(prediction, -displacement, length) - truth
        return float(np.vdot(difference, difference).real)

    candidates: list[tuple[float, float]] = []
    for peak_index in peak_indices:
        signed_index = int(peak_index)
        if signed_index > nx // 2:
            signed_index -= nx
        coarse = signed_index * dx
        result = minimize_scalar(
            objective,
            bounds=(coarse - dx, coarse + dx),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidates.append((float(result.fun), float(result.x)))
    return min(candidates)[1]


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - truth) / (np.linalg.norm(truth) + 1e-30))


def _hamiltonian(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    length: float,
) -> np.ndarray:
    dx = length / eta.shape[-1]
    return 0.5 * dx * np.sum(xi * gxi + eta**2, axis=-1)


def _terminal_case_metrics(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    truth_eta: np.ndarray,
    length: float,
) -> dict[str, float | bool]:
    finite = bool(
        np.isfinite(eta).all() and np.isfinite(xi).all() and np.isfinite(gxi).all()
    )
    if not finite:
        return {
            "finite": False,
            "raw_eta_error": float("inf"),
            "aligned_eta_error": float("inf"),
            "displacement": float("nan"),
            "displacement_grid_points": float("nan"),
            "hamiltonian_drift": float("nan"),
            "eta_band_64_128_ratio_to_truth": float("nan"),
        }
    displacement = _optimal_displacement(eta[-1], truth_eta[-1], length)
    aligned = _periodic_shift(eta[-1], -displacement, length)
    hamiltonian = _hamiltonian(eta, xi, gxi, length)
    hamiltonian_drift = (hamiltonian[-1] - hamiltonian[0]) / (
        abs(hamiltonian[0]) + 1e-30
    )
    nx = eta.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    band = (np.abs(k) >= 64.0) & (np.abs(k) <= 128.0)
    eta_band = np.linalg.norm(np.fft.fft(eta[-1])[band])
    truth_band = np.linalg.norm(np.fft.fft(truth_eta[-1])[band])
    return {
        "finite": True,
        "raw_eta_error": _relative_l2(eta[-1], truth_eta[-1]),
        "aligned_eta_error": _relative_l2(aligned, truth_eta[-1]),
        "displacement": displacement,
        "displacement_grid_points": displacement / (length / nx),
        "hamiltonian_drift": float(hamiltonian_drift),
        "eta_band_64_128_ratio_to_truth": float(eta_band / (truth_band + 1e-30)),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="final")
    parser.add_argument("--case-ids", type=int, nargs="+", default=DEFAULT_CASE_IDS)
    parser.add_argument(
        "--multi-arm-case-id",
        type=int,
        help=(
            "run the focused five-arm diagnostic on one case; when set, this "
            "overrides --case-ids without changing the default modal-only path"
        ),
    )
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--inner-dt", type=float, default=0.01)
    parser.add_argument("--gl2-iterations", type=int, default=4)
    parser.add_argument("--filter-fraction", type=float, default=0.25)
    parser.add_argument("--reference-order", type=int, default=6)
    parser.add_argument("--pad-factor", type=int, default=8)
    parser.add_argument("--k-max", type=float, default=128.0)
    parser.add_argument("--relative-energy-floor", type=float, default=1e-12)
    parser.add_argument(
        "--localized-energy-floor-relative",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)
    requested_case_ids = (
        (args.multi_arm_case_id,)
        if args.multi_arm_case_id is not None
        else tuple(args.case_ids)
    )
    panel = _load_fixed_tail_panel(
        args.eval_dir.resolve(), requested_case_ids, args.max_time
    )
    multi_arm = args.multi_arm_case_id is not None
    nx = panel.truth_eta.shape[-1]
    saved_dt = float(panel.times[1] - panel.times[0])
    substeps = int(round(saved_dt / args.inner_dt))
    if substeps <= 0 or not np.isclose(saved_dt / substeps, args.inner_dt):
        raise ValueError(
            f"saved interval {saved_dt} is not an integer multiple of inner_dt "
            f"{args.inner_dt}"
        )

    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    predict_learned = build_predict_gxi_batched(loaded)
    _, k_np = build_grid(nx, args.length)
    k = jnp.asarray(k_np, dtype=jnp.float64)
    rollout_depths_np = (
        np.repeat(panel.depths, len(MULTI_ARM_NAMES)) if multi_arm else panel.depths
    )
    depths = jnp.asarray(rollout_depths_np, dtype=jnp.float64)[:, None]
    log_depths = jnp.asarray(np.log(rollout_depths_np), dtype=jnp.float32)
    g0 = make_linear_dno_symbol(k, depths)
    params = ti.SolverParams(
        nx=nx,
        length=args.length,
        depth=depths,
        gravity=1.0,
        dno_order=args.reference_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
        k=k,
        g0=g0,
    )

    def oracle_parts(
        eta: jax.Array,
        xi: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        learned = predict_learned(
            eta.astype(jnp.float32), xi.astype(jnp.float32), log_depths
        ).astype(jnp.float64)
        reference = dno_series_eval(
            eta,
            xi,
            k,
            depths,
            args.reference_order,
            pad_factor=args.pad_factor,
        )
        reference = reference - jnp.mean(reference, axis=-1, keepdims=True)
        defect = reference - learned
        modal = modal_phase_component(
            eta,
            defect,
            k,
            k_max=args.k_max,
            relative_energy_floor=args.relative_energy_floor,
        )
        if multi_arm:
            localized_sigma1 = localized_translation_component(
                eta,
                defect,
                k,
                depths,
                window_depths=1.0,
                energy_floor_relative=args.localized_energy_floor_relative,
            )
            localized_sigma4 = localized_translation_component(
                eta,
                defect,
                k,
                depths,
                window_depths=4.0,
                energy_floor_relative=args.localized_energy_floor_relative,
            )
            defect_centered = defect - jnp.mean(defect, axis=-1, keepdims=True)
            radial = defect_centered - modal
            correction = jnp.stack(
                (
                    modal[0],
                    localized_sigma1[1],
                    localized_sigma4[2],
                    radial[3],
                    defect_centered[4],
                ),
                axis=0,
            )
        else:
            correction = modal
        return learned, reference, correction

    def predict_oracle(eta: jax.Array, xi: jax.Array) -> jax.Array:
        learned, _, correction = oracle_parts(eta, xi)
        result = learned + correction
        return result - jnp.mean(result, axis=-1, keepdims=True)

    @jax.jit
    def run_oracle() -> dict[str, jax.Array]:
        initial_eta = jnp.asarray(panel.truth_eta[0], dtype=jnp.float64)
        initial_xi = jnp.asarray(panel.truth_xi[0], dtype=jnp.float64)
        if multi_arm:
            initial_eta = jnp.repeat(initial_eta, len(MULTI_ARM_NAMES), axis=0)
            initial_xi = jnp.repeat(initial_xi, len(MULTI_ARM_NAMES), axis=0)
        return rollout_surrogate(
            ti.State(
                eta=initial_eta,
                xi=initial_xi,
            ),
            jnp.asarray(panel.times, dtype=jnp.float64),
            params,
            predict_oracle,
            substeps=substeps,
            zero_mean_xi=True,
            gl2_iterations=args.gl2_iterations,
        )

    started = time.perf_counter()
    oracle_device = run_oracle()
    jax.block_until_ready(oracle_device["eta"])
    wall_seconds = time.perf_counter() - started
    oracle = {
        field: np.asarray(jax.device_get(oracle_device[field]))
        for field in ("eta", "xi", "gxi")
    }

    final_parts = jax.jit(oracle_parts)(
        oracle_device["eta"][-1], oracle_device["xi"][-1]
    )
    learned_final, reference_final, correction_final = (
        np.asarray(jax.device_get(value)) for value in final_parts
    )
    defect_final = reference_final - learned_final

    common_payload: dict[str, object] = {
        "deployed_analytic_ceiling": (
            "G0+G1 (unchanged; reference order is diagnostic only)"
        ),
        "run_dir": str(args.run_dir.resolve()),
        "eval_dir": str(args.eval_dir.resolve()),
        "checkpoint": args.checkpoint,
        "case_ids": panel.case_ids.tolist(),
        "reference_order": args.reference_order,
        "pad_factor": args.pad_factor,
        "k_max": args.k_max,
        "relative_energy_floor": args.relative_energy_floor,
        "inner_dt": args.inner_dt,
        "saved_dt": saved_dt,
        "substeps": substeps,
        "gl2_iterations": args.gl2_iterations,
        "max_time": float(panel.times[-1]),
        "wall_seconds": wall_seconds,
    }

    if multi_arm:
        baseline_metrics = _terminal_case_metrics(
            panel.baseline_eta[:, 0],
            panel.baseline_xi[:, 0],
            panel.baseline_gxi[:, 0],
            panel.truth_eta[:, 0],
            args.length,
        )
        arms: dict[str, object] = {}
        for arm_index, arm_name in enumerate(MULTI_ARM_NAMES):
            terminal_metrics = _terminal_case_metrics(
                oracle["eta"][:, arm_index],
                oracle["xi"][:, arm_index],
                oracle["gxi"][:, arm_index],
                panel.truth_eta[:, 0],
                args.length,
            )
            defect_norm = np.linalg.norm(defect_final[arm_index]) + 1e-30
            arms[arm_name] = {
                "terminal_metrics": terminal_metrics,
                "terminal_correction_fraction_of_dno_defect": float(
                    np.linalg.norm(correction_final[arm_index]) / defect_norm
                ),
                "terminal_residual_fraction_of_dno_defect": float(
                    np.linalg.norm(
                        defect_final[arm_index] - correction_final[arm_index]
                    )
                    / defect_norm
                ),
            }
        arm_raw_errors = np.asarray(
            [
                arms[name]["terminal_metrics"]["raw_eta_error"]
                for name in MULTI_ARM_NAMES
            ],
            dtype=np.float64,
        )
        arm_aligned_errors = np.asarray(
            [
                arms[name]["terminal_metrics"]["aligned_eta_error"]
                for name in MULTI_ARM_NAMES
            ],
            dtype=np.float64,
        )
        payload = {
            "mechanism": (
                "same-state order-six defect, focused modal/local/radial/full "
                "projection comparison"
            ),
            **common_payload,
            "case_id": int(panel.case_ids[0]),
            "depth": float(panel.depths[0]),
            "arm_names": list(MULTI_ARM_NAMES),
            "localized_energy_floor_relative": (args.localized_energy_floor_relative),
            "baseline": baseline_metrics,
            "arms": arms,
            "all_arm_rollouts_finite": bool(
                all(np.isfinite(oracle[field]).all() for field in ("eta", "xi", "gxi"))
            ),
            "all_arm_raw_eta_below_0p25": bool(np.all(arm_raw_errors < 0.25)),
            "arm_raw_eta_max": float(np.max(arm_raw_errors)),
            "arm_aligned_eta_max": float(np.max(arm_aligned_errors)),
        }
    else:
        cases: dict[str, object] = {}
        for index, case_id in enumerate(panel.case_ids):
            baseline_metrics = _terminal_case_metrics(
                panel.baseline_eta[:, index],
                panel.baseline_xi[:, index],
                panel.baseline_gxi[:, index],
                panel.truth_eta[:, index],
                args.length,
            )
            oracle_metrics = _terminal_case_metrics(
                oracle["eta"][:, index],
                oracle["xi"][:, index],
                oracle["gxi"][:, index],
                panel.truth_eta[:, index],
                args.length,
            )
            cases[str(int(case_id))] = {
                "depth": float(panel.depths[index]),
                "baseline": baseline_metrics,
                "phase_oracle": oracle_metrics,
                "terminal_phase_component_fraction_of_dno_defect": float(
                    np.linalg.norm(correction_final[index])
                    / (np.linalg.norm(defect_final[index]) + 1e-30)
                ),
            }
        raw_errors = np.asarray(
            [
                cases[str(int(case_id))]["phase_oracle"]["raw_eta_error"]
                for case_id in panel.case_ids
            ],
            dtype=np.float64,
        )
        aligned_errors = np.asarray(
            [
                cases[str(int(case_id))]["phase_oracle"]["aligned_eta_error"]
                for case_id in panel.case_ids
            ],
            dtype=np.float64,
        )
        payload = {
            "mechanism": "same-state order-six defect, modal phase component only",
            **common_payload,
            "all_oracle_rollouts_finite": bool(
                all(np.isfinite(oracle[field]).all() for field in ("eta", "xi", "gxi"))
            ),
            "all_selected_raw_eta_below_0p25": bool(np.all(raw_errors < 0.25)),
            "oracle_raw_eta_max": float(np.max(raw_errors)),
            "oracle_aligned_eta_max": float(np.max(aligned_errors)),
            "cases": cases,
        }
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            args.run_dir / f"neutral_multiarm_case{int(panel.case_ids[0])}_{stamp}"
            if multi_arm
            else args.run_dir / f"phase_oracle_fixed6_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if multi_arm:
        np.savez_compressed(
            output_dir / "trajectories.npz",
            times=panel.times,
            depth=panel.depths[0],
            case_id=panel.case_ids[0],
            arm_names=np.asarray(MULTI_ARM_NAMES),
            truth_eta=panel.truth_eta[:, 0],
            truth_xi=panel.truth_xi[:, 0],
            truth_gxi=panel.truth_gxi[:, 0],
            baseline_eta=panel.baseline_eta[:, 0],
            baseline_xi=panel.baseline_xi[:, 0],
            baseline_gxi=panel.baseline_gxi[:, 0],
            arm_eta=oracle["eta"],
            arm_xi=oracle["xi"],
            arm_gxi=oracle["gxi"],
        )
    else:
        np.savez_compressed(
            output_dir / "trajectories.npz",
            times=panel.times,
            depths=panel.depths,
            case_ids=panel.case_ids,
            truth_eta=panel.truth_eta,
            truth_xi=panel.truth_xi,
            truth_gxi=panel.truth_gxi,
            baseline_eta=panel.baseline_eta,
            baseline_xi=panel.baseline_xi,
            baseline_gxi=panel.baseline_gxi,
            oracle_eta=oracle["eta"],
            oracle_xi=oracle["xi"],
            oracle_gxi=oracle["gxi"],
        )
    print(json.dumps(_json_safe(payload), indent=2, allow_nan=False), flush=True)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
