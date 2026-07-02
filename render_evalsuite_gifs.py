"""Split an eval_suite trajs.npz into per-IC comparison npzs and render GIFs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajs_npz", required=True, help="eval_suite <regime>_trajs.npz")
    p.add_argument("--case_ids", type=int, nargs="+", required=True)
    p.add_argument("--length", type=float, default=2.0 * float(np.pi))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trajs_path = Path(args.trajs_npz).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else trajs_path.parent / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(trajs_path)
    nx = d["pred_eta"].shape[-1]
    x = np.linspace(0.0, args.length, nx, endpoint=False, dtype=np.float32)
    t = np.asarray(d["times"], dtype=np.float32)
    depths = np.asarray(d["depths"])

    render_script = Path(__file__).parent / "solver" / "evals" / "render_rollout_movie.py"

    for j in args.case_ids:
        h = float(depths[j])
        stem = f"case{j:04d}_h{h:.4f}"
        comp_npz = out_dir / f"{stem}_comparison.npz"
        np.savez_compressed(
            comp_npz,
            x=x, t=t,
            truth_eta=d["truth_eta"][:, j, :].astype(np.float32),
            pred_eta=d["pred_eta"][:, j, :].astype(np.float32),
            truth_xi=d["truth_xi"][:, j, :].astype(np.float32),
            pred_xi=d["pred_xi"][:, j, :].astype(np.float32),
            truth_gxi=d["truth_gxi"][:, j, :].astype(np.float32),
            pred_gxi=d["pred_gxi"][:, j, :].astype(np.float32),
            depth=np.float32(h),
        )

        pred_nan = bool(np.isnan(d["pred_eta"][:, j, :]).any())
        title = f"tanaka_g0 j={j} h={h:.3f}{' [NaN]' if pred_nan else ''}"
        gif = out_dir / f"{stem}.gif"
        cmd = [
            sys.executable, str(render_script),
            "--comparison_npz", str(comp_npz),
            "--output", str(gif),
            "--title", title,
            "--fps", str(args.fps),
            "--n_frames", str(args.n_frames),
            "--dpi", str(args.dpi),
        ]
        print(f"[j={j}] rendering -> {gif}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
