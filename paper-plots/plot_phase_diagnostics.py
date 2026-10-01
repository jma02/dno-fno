"""Plot phase and local-potential errors from saved evaluations, CPU only."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper-plots/figures"
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
TITLES = ("Stokes", "Tanaka", "Benjamin–Feir", "JONSWAP / TMA")
MODELS = (
    ("c27_w320_b4_h80_ema_grad09_20260927", "Original EMA", "#c45b22"),
    ("c27_phase_m1_tg10_ep120_gpu0_20260930", "Local tangent", "#007da6"),
    ("c27_phase_modal_tg25_ep120_gpu1_20260930", "Modal phase", "#009e73"),
)


def fitted_speed(eta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Least-squares speed in eta_t = q = -c eta_x on the 2pi domain."""
    k = np.arange(eta.shape[-1] // 2 + 1)
    dx = np.fft.irfft(1j * k * np.fft.rfft(eta, axis=-1), n=eta.shape[-1])
    return -np.sum(q * dx, axis=-1) / np.sum(dx**2, axis=-1)


def median_band(
    ax: Axes, times: np.ndarray, values: np.ndarray, label: str, color: str
) -> None:
    low, median, high = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    ax.fill_between(times, low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(times, median, label=label, color=color, linewidth=1.8)


def save_figure(fig: Figure, name: str, title: str, ylabel: str) -> None:
    fig.suptitle(title, x=0.08, ha="left", fontsize=19, y=0.99)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 0.945),
               ncol=len(labels), frameon=False, fontsize=11)
    fig.supylabel(ylabel, x=0.015, fontsize=12)
    for ax in fig.axes:
        ax.set_yscale("log")
        ax.grid(axis="y", color="#e8ebee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    fig.subplots_adjust(top=1 - 1.05 / fig.get_figheight())
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.edgecolor": "#bfc6cc", "xtick.color": "#626c76",
        "ytick.color": "#626c76", "axes.titlesize": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    speed_fig, speed_axes = plt.subplots(1, 2, figsize=(11, 4.9))
    phase_fig, phase_axes = plt.subplots(2, 2, figsize=(11, 8))
    for family_index, family in enumerate(FAMILIES):
        for run, label, color in MODELS:
            evaluation = ("eval_best_current_test_stratified_n32_fp32net_fp64solver"
                          if run == MODELS[0][0] else "eval_epoch120_test_stratified_n32")
            path = ROOT / "outputs" / run / f"{evaluation}_gpu{int(family_index >= 2)}" / f"{family}_trajs.npz"
            with np.load(path) as data:
                times = data["times"]
                truth = data["truth_eta"]
                pred = data["pred_eta"]
                if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
                    raise ValueError(f"Nonfinite saved trajectory: {path}")
                ref_hat = np.fft.rfft(truth, axis=-1)[..., 1:129]
                pred_hat = np.fft.rfft(pred, axis=-1)[..., 1:129]
                angles = np.angle(pred_hat * ref_hat.conj())
                power = np.abs(ref_hat)**2
                phase = np.rad2deg(np.sqrt(
                    np.sum(power * angles**2, axis=-1) / np.sum(power, axis=-1)
                ))
                ax = phase_axes.flat[family_index]
                median_band(ax, times[1:], phase[1:], label, color)
                ax.set_title(f"{TITLES[family_index]} · {truth.shape[1]} cases", loc="left")
                ax.set_xlabel("Time")
                ax.set_xlim(0, times[-1])
                print(f"{label}, {family}: final Fourier-phase median {np.median(phase[-1]):.6g} deg")
                if family_index < 2:
                    groups = json.loads(path.with_name(f"{family}_summary.json").read_text())[
                        "evaluation_source"]["selected_parameter_group_ids"]
                    selected = np.array([family == "stokes" or "_m1_" in group for group in groups])
                    c_ref = fitted_speed(truth[:, selected], data["truth_gxi"][:, selected])
                    c_pred = fitted_speed(pred[:, selected], data["pred_gxi"][:, selected])
                    speed_error = 100 * np.abs((c_pred - c_ref) / c_ref)
                    if not np.isfinite(speed_error).all():
                        raise ValueError(f"Undefined fitted speed: {path}")
                    ax = speed_axes[family_index]
                    median_band(ax, times, speed_error, label, color)
                    title = "Stokes" if family == "stokes" else "Isolated Tanaka"
                    ax.set_title(f"{title} · {selected.sum()} cases", loc="left")
                    ax.set_xlabel("Time")
                    ax.set_xlim(0, times[-1])
                    print(f"{label}, {family}: final fitted-speed median {np.median(speed_error[-1]):.6g}%")

    save_figure(speed_fig, "15-phase-speed-error", "Phase-speed error",
                "Fitted translation-speed error (%)")
    save_figure(phase_fig, "16-fourier-phase-error", "Fourier-phase error",
                "Power-weighted RMS phase error (degrees)")

    probe_fig, probe_axes = plt.subplots(2, 2, figsize=(11, 8))
    with np.load(ROOT / "outputs/operator_probe_a100_20261001/operator_probes.npz") as data:
        methods = data["method_names"].tolist()
        errors = data["relative_l2"]
        for family_index, title in enumerate(TITLES):
            ax = probe_axes.flat[family_index]
            selected = data["family_ids"] == family_index
            for index, (run, label, color) in enumerate((("M2", "M2", "#606970"), *MODELS)):
                per_case = errors[methods.index(run), selected]
                values = 100 * np.stack((per_case[:, 0], np.median(per_case[:, 5:9], axis=1)))
                low, median, high = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
                x = np.array([0., 1.]) + (index - 1.5) * 0.055
                ax.errorbar(x, median, yerr=np.stack((median - low, high - median)),
                            color=color, label=label, marker="o", markersize=5,
                            linewidth=1.6, elinewidth=1, capsize=3)
                print(f"{label}, {FAMILIES[family_index]}: DNO median {median[0]:.6g} -> {median[1]:.6g}%")
            ax.set_title(f"{title} · {selected.sum()} cases", loc="left")
            ax.set_xticks([0, 1], ["Original potential", "10% perturbation"])
            ax.set_xlim(-0.25, 1.25)
    save_figure(probe_fig, "17-local-perturbation-error", "Local potential perturbations",
                "Relative DNO error (%)")


if __name__ == "__main__":
    main()
