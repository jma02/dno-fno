"""Render truth-vs-surrogate animations from an eval_suite ``<regime>_trajs.npz``.

Usage:
    uv run python -m solver.evals.animate_trajs \\
        --trajs outputs/.../eval_suite/random_sea_deep_trajs.npz \\
        --select best,median,worst \\
        --out outputs/.../eval_suite/random_sea_deep_movies/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def _select_ic_indices(rel_l2_eta: np.ndarray, select: Sequence[str], n_ics: int) -> list[tuple[str, int]]:
    final = rel_l2_eta[-1].copy()
    valid = np.isfinite(final)
    if valid.any():
        order = np.argsort(np.where(valid, final, np.inf))
    else:
        order = np.arange(n_ics)
    out: list[tuple[str, int]] = []
    for tok in select:
        if tok == "best":
            out.append(("best", int(order[0])))
        elif tok == "worst":
            out.append(("worst", int(order[-1])))
        elif tok == "median":
            out.append(("median", int(order[len(order) // 2])))
        elif tok.startswith("idx="):
            out.append((tok, int(tok.split("=", 1)[1])))
        else:
            try:
                out.append((f"idx={int(tok)}", int(tok)))
            except ValueError:
                continue
    return out


def render_movie(
    out_path: Path,
    *,
    x: np.ndarray,
    times: np.ndarray,
    truth_eta: np.ndarray,        # (n_t, nx)
    pred_eta: np.ndarray,
    truth_xi: np.ndarray,
    pred_xi: np.ndarray,
    truth_gxi: np.ndarray,
    pred_gxi: np.ndarray,
    rel_l2_eta: np.ndarray,       # (n_t,)
    rel_l2_xi: np.ndarray,
    rel_l2_gxi: np.ndarray,
    title: str,
    fps: int = 24,
    stride: int = 1,
    dpi: int = 110,
) -> None:
    """3-row × 2-col figure: η, ξ, G(η)ξ; left=overlay, right=error vs time so far."""
    n_t = truth_eta.shape[0]
    frames = list(range(0, n_t, stride))

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.0), gridspec_kw={"width_ratios": [2.2, 1.0]})
    fig.suptitle(title, fontsize=12, fontweight="bold")

    rows = [
        (r"$\eta$", truth_eta, pred_eta, rel_l2_eta, "tab:blue"),
        (r"$\xi$", truth_xi, pred_xi, rel_l2_xi, "tab:green"),
        (r"$G(\eta)\xi$", truth_gxi, pred_gxi, rel_l2_gxi, "tab:orange"),
    ]

    lines_truth, lines_pred, err_lines, err_dots = [], [], [], []
    for r, (label, tr, _, e, color) in enumerate(rows):
        ax_l = axes[r, 0]
        ax_r = axes[r, 1]
        (lt,) = ax_l.plot(x, tr[0], color="black", lw=1.0, label="truth")
        (lp,) = ax_l.plot(x, tr[0], color="tab:red", lw=1.0, ls="--", label="surrogate")
        ax_l.set_ylabel(label, fontsize=12)
        ax_l.grid(alpha=0.3)
        if r == 0:
            ax_l.legend(loc="upper right", fontsize=9)
        ax_l.set_xlim(float(x.min()), float(x.max()))
        ymax = float(np.nanmax(np.abs(tr))) * 1.15
        if not np.isfinite(ymax) or ymax == 0.0:
            ymax = 1.0
        ax_l.set_ylim(-ymax, ymax)

        finite_e = e[np.isfinite(e)]
        floor = max(1e-8, float(np.nanmin(finite_e[finite_e > 0])) if finite_e.size and (finite_e > 0).any() else 1e-8)
        ax_r.semilogy(times, np.maximum(e, floor), color=color, lw=1.0)
        (dot,) = ax_r.semilogy([times[0]], [max(e[0], floor)], "o", color=color, markersize=5)
        ax_r.set_xlim(float(times[0]), float(times[-1]))
        emax = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
        ax_r.set_ylim(floor, max(emax * 1.5, floor * 10))
        ax_r.grid(alpha=0.3, which="both")
        ax_r.set_ylabel("rel L2", fontsize=10)
        if r == 2:
            ax_r.set_xlabel("t", fontsize=10)
            axes[2, 0].set_xlabel("x", fontsize=10)

        lines_truth.append(lt)
        lines_pred.append(lp)
        err_dots.append(dot)
        err_lines.append(ax_r)

    time_text = fig.text(0.5, 0.965, "", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))

    def update(frame: int):
        t = times[frame]
        time_text.set_text(f"t = {t:.2f}")
        for r, (_, tr, pr, e, _) in enumerate(rows):
            lines_truth[r].set_ydata(tr[frame])
            lines_pred[r].set_ydata(pr[frame])
            err_dots[r].set_data([t], [max(e[frame], 1e-12)])
        return [*lines_truth, *lines_pred, *err_dots, time_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp4":
        try:
            anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=2400))
        except (RuntimeError, FileNotFoundError):
            gif_path = out_path.with_suffix(".gif")
            print(f"  ffmpeg unavailable; falling back to {gif_path}")
            anim.save(str(gif_path), writer=PillowWriter(fps=fps))
    else:
        anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajs", required=True, help="Path to <regime>_trajs.npz")
    parser.add_argument("--out", default=None, help="Output dir. Defaults to <trajs_parent>/<stem>_movies/")
    parser.add_argument("--select", default="best,median,worst",
                        help="comma-separated: best,median,worst,idx=K,...")
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (1 = every saved frame).")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    args = parser.parse_args()

    trajs_path = Path(args.trajs).resolve()
    stem = trajs_path.stem.replace("_trajs", "")
    out_dir = Path(args.out).resolve() if args.out else trajs_path.parent / f"{stem}_movies"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(trajs_path) as d:
        times = np.asarray(d["times"])
        depths = np.asarray(d["depths"])
        case_ids = np.asarray(d["case_ids"]) if "case_ids" in d.files else np.arange(d["depths"].shape[0])
        truth_eta = np.asarray(d["truth_eta"])    # (n_t, NB, nx)
        truth_xi = np.asarray(d["truth_xi"])
        truth_gxi = np.asarray(d["truth_gxi"])
        pred_eta = np.asarray(d["pred_eta"])
        pred_xi = np.asarray(d["pred_xi"])
        pred_gxi = np.asarray(d["pred_gxi"])
        re = np.asarray(d["rel_l2_eta"])          # (n_t, NB)
        rx = np.asarray(d["rel_l2_xi"])
        rg = np.asarray(d["rel_l2_gxi"])

    n_t, NB, nx = truth_eta.shape
    x = np.linspace(0.0, args.length, nx, endpoint=False)

    select = [tok.strip() for tok in args.select.split(",") if tok.strip()]
    picks = _select_ic_indices(re, select, NB)
    print(f"Animating {len(picks)} cases from {trajs_path.name} (n_t={n_t}, NB={NB})")
    for label, j in picks:
        out_path = out_dir / f"{stem}_{label}_case{int(case_ids[j])}_h{float(depths[j]):.3f}.{args.format}"
        title = (
            f"{stem}  {label} (case={int(case_ids[j])}, h={float(depths[j]):.3f})  "
            f"final η rel-L2={float(re[-1, j]):.2e}"
        )
        print(f"  rendering {out_path.name}...")
        render_movie(
            out_path,
            x=x, times=times,
            truth_eta=truth_eta[:, j, :], pred_eta=pred_eta[:, j, :],
            truth_xi=truth_xi[:, j, :], pred_xi=pred_xi[:, j, :],
            truth_gxi=truth_gxi[:, j, :], pred_gxi=pred_gxi[:, j, :],
            rel_l2_eta=re[:, j], rel_l2_xi=rx[:, j], rel_l2_gxi=rg[:, j],
            title=title, fps=args.fps, stride=args.stride,
        )
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
