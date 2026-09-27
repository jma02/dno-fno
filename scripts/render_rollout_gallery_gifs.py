"""Render median/worst examples from saved evaluation archives, using CPUs only."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rollout-gallery-matplotlib")

import numpy as np  # noqa: E402

# Import the CPU renderer without importing solver's JAX package initializer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solver" / "evals"))
from render_rollout_movie import render_rollout_gif  # noqa: E402


def render_archive(task: tuple[Path, Path]) -> list[Path]:
    archive, output = task
    family = archive.stem.removesuffix("_trajs")
    summary = json.loads(archive.with_name(f"{family}_summary.json").read_text())
    with np.load(archive) as stored:
        errors = stored["rel_l2_eta"][-1]
        valid = np.flatnonzero(stored["truth_valid"] & np.isfinite(errors))
        order = valid[np.argsort(errors[valid])]
        if not len(order):
            raise ValueError(f"No finite truth-valid examples in {archive}")
        picks = [("median", int(order[len(order) // 2])), ("worst", int(order[-1]))]
        indices = [index for _, index in picks]
        fields = {
            name: stored[name][:, indices, :]
            for name in (
                "truth_eta",
                "truth_xi",
                "truth_gxi",
                "pred_eta",
                "pred_xi",
                "pred_gxi",
            )
        }
        times = stored["times"]
        ids = stored["simulation_ids"][indices]
        depths = stored["depths"][indices]

    x = np.linspace(
        0.0, summary["length"], fields["truth_eta"].shape[-1], endpoint=False
    )
    written = []
    for column, (label, index) in enumerate(picks):
        path = output / f"{family}_{label}_sim{int(ids[column])}.gif"
        print(f"Rendering {path.name} from row {index}", flush=True)
        payload = {name: values[:, column, :] for name, values in fields.items()}
        payload.update(x=x, t=times)
        render_rollout_gif(
            payload,
            path,
            title=(
                f"{family.replace('_', ' ')} | {label} final eta error | "
                f"simulation {int(ids[column])} | h = {depths[column]:.4g}"
            ),
            fps=12,
            n_frames=len(times),
            dpi=64,
        )
        print(f"Written {path.name}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for paths in pool.map(
            render_archive, [(path, args.output) for path in args.trajs]
        ):
            for path in paths:
                print(path, flush=True)


if __name__ == "__main__":
    main()
