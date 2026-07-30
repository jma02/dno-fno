"""Focused one-step operator and translation-speed probe for Tanaka case 27."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import build_predict_gxi_with_depth, load_run


DEFAULT_TRUTH = Path(
    "outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/"
    "eval_suite_f64h_n32/tanaka_g0_trajs.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--case_id", type=int, default=27)
    return parser.parse_args()


def projected_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    k_max: int,
) -> float:
    pred_hat = np.fft.rfft(prediction)
    target_hat = np.fft.rfft(target)
    keep = np.arange(pred_hat.size) <= k_max
    return float(
        np.linalg.norm((pred_hat - target_hat)[keep])
        / (np.linalg.norm(target_hat[keep]) + 1e-30)
    )


def main() -> None:
    args = parse_args()
    with np.load(args.truth) as archive:
        case_ids = np.asarray(archive["case_ids"])
        matches = np.flatnonzero(case_ids == args.case_id)
        if matches.size != 1:
            raise ValueError(f"case {args.case_id} occurs {matches.size} times")
        case_index = int(matches[0])
        eta = np.asarray(archive["truth_eta"][0, case_index], dtype=np.float64)
        xi = np.asarray(archive["truth_xi"][0, case_index], dtype=np.float64)
        target = np.asarray(archive["truth_gxi"][0, case_index], dtype=np.float64)
        depth = float(np.asarray(archive["depths"])[case_index])

    nx = eta.size
    k = np.fft.fftfreq(nx, d=1.0 / nx)
    eta_x = np.fft.ifft(1j * k * np.fft.fft(eta)).real
    speed_truth = -float(np.vdot(target, eta_x).real / np.vdot(eta_x, eta_x).real)

    print(
        f"case={args.case_id} h={depth:.9g} "
        f"a/h={np.max(eta) / depth:.6g} c_truth={speed_truth:.9g}"
    )
    for run_dir in args.run_dirs:
        loaded = load_run(run_dir, checkpoint=args.checkpoint)
        predict = build_predict_gxi_with_depth(loaded)
        prediction = np.asarray(
            jax.device_get(
                predict(
                    jnp.asarray(eta),
                    jnp.asarray(xi),
                    jnp.asarray(np.log(depth)),
                )
            ),
            dtype=np.float64,
        )
        speed_pred = -float(
            np.vdot(prediction, eta_x).real / np.vdot(eta_x, eta_x).real
        )
        print(
            f"{run_dir} epoch={loaded.epoch} "
            f"rel_l2_k128={projected_relative_l2(prediction, target, 128):.8g} "
            f"c_pred={speed_pred:.9g} "
            f"dc={speed_pred - speed_truth:+.8g} "
            f"dc_rel={(speed_pred - speed_truth) / speed_truth:+.6%}"
        )


if __name__ == "__main__":
    main()
