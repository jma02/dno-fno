from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "cm"

SKY_COLOR = "#0b1d35"
TRUE_LINE_COLOR = "#8ed6ff"
PRED_LINE_COLOR = "#ff5a5f"
TEXT_COLOR = "#c8e0f0"
AXIS_COLOR = "#aac8e0"
SPINE_COLOR = "#304a60"
TRUE_LINE_ALPHA = 1.0
PRED_LINE_ALPHA = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a rollout-vs-truth movie from a saved comparison NPZ.")
    parser.add_argument("--comparison_npz", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def padded_limits(*series: np.ndarray, frame_idx: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([values[frame_idx] for values in series], axis=0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.concatenate([values[np.isfinite(values)] for values in series], axis=0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    span = vmax - vmin
    pad = 0.08 * span if span > 0 else 1.0
    return vmin - pad, vmax + pad


def finite_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def render_rollout_gif(
    payload: dict[str, np.ndarray],
    output_path: str | Path,
    title: str | None = None,
    fps: int = 30,
    n_frames: int = 200,
    dpi: int = 120,
) -> Path:
    x = np.asarray(payload["x"])
    t = np.asarray(payload["t"])
    pred_eta = np.asarray(payload["pred_eta"])
    pred_xi = np.asarray(payload["pred_xi"])
    pred_gxi = np.asarray(payload["pred_gxi"])
    has_truth = all(key in payload for key in ("truth_eta", "truth_xi", "truth_gxi"))

    if has_truth:
        truth_eta = np.asarray(payload["truth_eta"])
        truth_xi = np.asarray(payload["truth_xi"])
        truth_gxi = np.asarray(payload["truth_gxi"])
    else:
        truth_eta = pred_eta
        truth_xi = pred_xi
        truth_gxi = pred_gxi

    n_steps = truth_eta.shape[0]
    idx = np.linspace(0, n_steps - 1, min(n_frames, n_steps), dtype=int)

    eta_all = np.concatenate((truth_eta[idx], pred_eta[idx]), axis=0)
    eta_finite = eta_all[np.isfinite(eta_all)]
    if eta_finite.size == 0:
        eta_finite = truth_eta[np.isfinite(truth_eta)]
    eta_min = float(np.min(eta_finite))
    eta_max = float(np.max(eta_finite))
    amp = eta_max - eta_min if eta_max != eta_min else 1.0

    water_bottom = eta_min - 0.12 * amp
    y_top = eta_max + 0.18 * amp
    xi_limits = padded_limits(truth_xi, pred_xi, frame_idx=idx)
    gxi_limits = padded_limits(truth_gxi, pred_gxi, frame_idx=idx)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        facecolor=SKY_COLOR,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.0]},
    )
    ax_eta, ax_xi, ax_gxi = axes
    for ax in axes:
        ax.set_facecolor(SKY_COLOR)

    truth0 = truth_eta[idx[0]]
    pred0 = pred_eta[idx[0]]
    truth_xi0 = truth_xi[idx[0]]
    pred_xi0 = pred_xi[idx[0]]
    truth_gxi0 = truth_gxi[idx[0]]
    pred_gxi0 = pred_gxi[idx[0]]

    truth_line = None
    if has_truth:
        (truth_line,) = ax_eta.plot(
            x,
            finite_curve(truth0),
            color=TRUE_LINE_COLOR,
            lw=1.3,
            alpha=TRUE_LINE_ALPHA,
            zorder=5,
            label="true",
        )
    (pred_line,) = ax_eta.plot(
        x,
        finite_curve(pred0),
        color=PRED_LINE_COLOR,
        lw=1.8,
        alpha=PRED_LINE_ALPHA,
        zorder=6,
        label="ours" if has_truth else "rollout",
    )

    truth_xi_line = None
    if has_truth:
        (truth_xi_line,) = ax_xi.plot(
            x,
            finite_curve(truth_xi0),
            color=TRUE_LINE_COLOR,
            lw=1.2,
            alpha=TRUE_LINE_ALPHA,
            zorder=5,
            label="true",
        )
    (pred_xi_line,) = ax_xi.plot(
        x,
        finite_curve(pred_xi0),
        color=PRED_LINE_COLOR,
        lw=1.8,
        alpha=PRED_LINE_ALPHA,
        zorder=6,
        label="ours" if has_truth else "rollout",
    )

    truth_gxi_line = None
    if has_truth:
        (truth_gxi_line,) = ax_gxi.plot(
            x,
            finite_curve(truth_gxi0),
            color=TRUE_LINE_COLOR,
            lw=1.2,
            alpha=TRUE_LINE_ALPHA,
            zorder=5,
            label="true",
        )
    (pred_gxi_line,) = ax_gxi.plot(
        x,
        finite_curve(pred_gxi0),
        color=PRED_LINE_COLOR,
        lw=1.8,
        alpha=PRED_LINE_ALPHA,
        zorder=6,
        label="ours" if has_truth else "rollout",
    )

    time_text = ax_eta.text(
        0.015,
        0.93,
        f"t = {t[idx[0]]:.2f}",
        transform=ax_eta.transAxes,
        color="white",
        fontsize=10,
        va="top",
        fontfamily="monospace",
    )

    display_title = title or "rollout comparison"
    ax_eta.set_xlim(x[0], x[-1])
    ax_eta.set_ylim(water_bottom, y_top)
    ax_eta.set_ylabel(r"$\eta$", color=AXIS_COLOR, fontsize=12, rotation=0, labelpad=14)
    ax_eta.set_title(display_title, color=TEXT_COLOR, fontsize=11, pad=8)
    ax_xi.set_ylim(*xi_limits)
    ax_xi.set_ylabel(r"$\xi$", color=AXIS_COLOR, fontsize=12, rotation=0, labelpad=14)
    ax_gxi.set_ylim(*gxi_limits)
    ax_gxi.set_ylabel(r"$G(\eta)\xi$", color=AXIS_COLOR, fontsize=12, rotation=0, labelpad=18)
    ax_gxi.set_xlabel("x", color=AXIS_COLOR, fontsize=10)

    for ax in axes:
        ax.tick_params(colors=AXIS_COLOR, labelsize=8)
        ax.grid(True, alpha=0.18, color="#6c8aa3")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(SPINE_COLOR)

    leg = ax_eta.legend(loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(TEXT_COLOR)

    plt.tight_layout(pad=0.8, h_pad=0.7)

    def update(frame_idx: int):
        eta_true = truth_eta[idx[frame_idx]]
        eta_pred = pred_eta[idx[frame_idx]]
        xi_true = truth_xi[idx[frame_idx]]
        xi_pred = pred_xi[idx[frame_idx]]
        gxi_true = truth_gxi[idx[frame_idx]]
        gxi_pred = pred_gxi[idx[frame_idx]]
        if truth_line is not None:
            truth_line.set_ydata(finite_curve(eta_true))
        pred_line.set_ydata(finite_curve(eta_pred))
        if truth_xi_line is not None:
            truth_xi_line.set_ydata(finite_curve(xi_true))
        pred_xi_line.set_ydata(finite_curve(xi_pred))
        if truth_gxi_line is not None:
            truth_gxi_line.set_ydata(finite_curve(gxi_true))
        pred_gxi_line.set_ydata(finite_curve(gxi_pred))
        time_text.set_text(f"t = {t[idx[frame_idx]]:.2f}")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(idx),
        interval=1000 / fps,
        blit=False,
    )

    output_path = Path(output_path).resolve()
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    npz_path = Path(args.comparison_npz).resolve()
    payload = np.load(npz_path)
    output_path = Path(args.output).resolve() if args.output else npz_path.with_suffix(".gif")
    written = render_rollout_gif(
        payload,
        output_path=output_path,
        title=args.title or npz_path.stem,
        fps=args.fps,
        n_frames=args.n_frames,
        dpi=args.dpi,
    )
    print(written)


if __name__ == "__main__":
    main()
