from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.solitary_loader_jax import list_soliton_files, load_soliton_file
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ground-truth soliton trajectories with the same rollout movie style used for Tanaka rollouts."
    )
    parser.add_argument("--output_dir", default="outputs/gt_h_rollouts")
    parser.add_argument("--prefix", default="coll.anim_h")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def build_truth_payload(data: dict[str, object]) -> dict[str, np.ndarray]:
    eta = np.asarray(data["eta"])
    xi = np.asarray(data["xi"])
    gxi = np.asarray(data["gxi"])
    return {
        "x": np.asarray(data["x"]),
        "t": np.asarray(data["t"]),
        "pred_eta": eta,
        "pred_xi": xi,
        "pred_gxi": gxi,
        "eta0": eta[0],
        "xi0": xi[0],
        "gxi0": gxi[0],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for path in list_soliton_files():
        if not path.name.startswith(args.prefix):
            continue

        data = load_soliton_file(path)
        payload = build_truth_payload(data)
        stem = path.name
        npz_path = output_dir / f"{stem}.npz"
        gif_path = output_dir / f"{stem}.gif"

        np.savez_compressed(npz_path, **payload)
        render_rollout_gif(
            payload,
            output_path=gif_path,
            title=f"{stem} | ground truth",
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )

        rows.append(
            {
                "name": stem,
                "npz_path": str(npz_path),
                "gif_path": str(gif_path),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
