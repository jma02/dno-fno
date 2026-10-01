"""Plot phase and local-potential errors from saved evaluations, CPU only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_rollout_translation_decomposition import (  # noqa: E402
    aligned_fields,
    optimal_displacement,
)

FIGURES = ROOT / "paper-plots/figures"
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
TITLES = ("Stokes", "Tanaka", "Benjamin–Feir", "JONSWAP / TMA")
MODELS = (
    ("c27_w320_b4_h80_ema_grad09_20260927", "Neural network", "#c45b22"),
)


def median_band(
    ax: Axes, times: np.ndarray, values: np.ndarray, label: str, color: str
) -> None:
    low, median, high = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    ax.fill_between(times, low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(times, median, label=label, color=color, linewidth=1.8)


def save_figure(fig: Figure, name: str, title: str, ylabel: str) -> None:
    fig.suptitle(title, x=0.08, ha="left", fontsize=19, y=0.99)
    legend = {label: handle for ax in fig.axes
              for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(legend.values(), legend.keys(), loc="upper left", bbox_to_anchor=(0.075, 0.945),
               ncol=len(legend), frameon=False, fontsize=11)
    fig.supylabel(ylabel, x=0.015, fontsize=12)
    for ax in fig.axes:
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
            evaluation = "eval_best_current_test_stratified_n32_fp32net_fp64solver"
            path = ROOT / "outputs" / run / f"{evaluation}_gpu{int(family_index >= 2)}" / f"{family}_trajs.npz"
            with np.load(path) as data:
                times = data["times"]
                truth = data["truth_eta"]
                pred = data["pred_eta"]
                if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
                    raise ValueError(f"Nonfinite saved trajectory: {path}")
                ref_hat = np.fft.rfft(truth[1:], axis=-1)
                pred_hat = np.fft.rfft(pred[1:], axis=-1)
                weights = np.full(ref_hat.shape[-1], 2.)
                weights[[0, -1]] = 1
                total = np.sum(weights * np.abs(pred_hat - ref_hat)**2, axis=-1)
                amplitude = np.sum(weights * (np.abs(pred_hat) - np.abs(ref_hat))**2, axis=-1)
                phase = 100 * (1 - amplitude / total)
                if not (np.isfinite(phase).all() and np.all((phase >= -1e-6) & (phase <= 100 + 1e-6))):
                    raise ValueError(f"Undefined phase decomposition: {path}")
                phase = np.clip(phase, 0, 100)
                ax = phase_axes.flat[family_index]
                median_band(ax, times[1:], phase, label, color)
                ax.set_title(f"{TITLES[family_index]} · {truth.shape[1]} cases", loc="left")
                ax.set_xlabel("Time")
                ax.set_xlim(0, times[-1])
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
                print(f"{label}, {family}: final Fourier-phase share {np.median(phase[-1]):.3f}%", flush=True)
                if family_index < 2:
                    groups = json.loads(path.with_name(f"{family}_summary.json").read_text())[
                        "evaluation_source"]["selected_parameter_group_ids"]
                    selected = np.array([family == "stokes" or "_m1_" in group for group in groups])
                    reference = truth[5::5, selected].reshape(-1, truth.shape[-1])
                    prediction = pred[5::5, selected].reshape(reference.shape)
                    shifts = np.array([
                        optimal_displacement(p, r, 2 * np.pi)
                        for p, r in zip(prediction, reference)
                    ])
                    aligned = aligned_fields(prediction, shifts, 2 * np.pi)
                    raw_error = np.sum((prediction - reference)**2, axis=-1)
                    aligned_error = np.minimum(raw_error, np.sum((aligned - reference)**2, axis=-1))
                    translation = (100 * (1 - aligned_error / raw_error)).reshape(-1, selected.sum())
                    if not (np.isfinite(translation).all() and np.all(translation <= phase[4::5, selected] + 1e-5)):
                        raise ValueError(f"Inconsistent translation decomposition: {path}")
                    ax = speed_axes[family_index]
                    median_band(ax, times[5::5], translation, label, color)
                    title = "Stokes" if family == "stokes" else "Isolated Tanaka"
                    ax.set_title(f"{title} · {selected.sum()} cases", loc="left")
                    ax.set_xlabel("Time")
                    ax.set_xlim(0, times[-1])
                    ax.set_ylim(0, 100)
                    ax.set_yticks([0, 25, 50, 75, 100])
                    print(f"{label}, {family}: final translation-removable share {np.median(translation[-1]):.3f}%", flush=True)

    save_figure(speed_fig, "15-phase-speed-error", "Surface error removed by translation alignment",
                "Share of total squared surface error (%)")
    save_figure(phase_fig, "16-fourier-phase-error", "Fourier-phase share of surface error",
                "Share of total squared surface error (%)")

    probe_fig, ax = plt.subplots(figsize=(11, 5.8))
    with np.load(ROOT / "outputs/operator_probe_a100_20261001/operator_probes.npz") as data:
        methods = data["method_names"].tolist()
        errors = data["relative_l2"]
        for index, (run, label, color) in enumerate((("M2", "M2", "#606970"), *MODELS)):
            per_case = errors[methods.index(run)]
            ratios = np.median(per_case[:, 5:9], axis=1) / per_case[:, 0]
            low, median, high = np.array([
                np.quantile(ratios[data["family_ids"] == family_index], [0.25, 0.5, 0.75])
                for family_index in range(4)
            ]).T
            x = np.arange(4) + (index - 0.5) * 0.13
            ax.errorbar(x, median, yerr=np.stack((median - low, high - median)),
                        color=color, label=label, marker="o", markersize=6,
                        linestyle="none", elinewidth=1.5, capsize=4)
            print(f"{label}: paired perturbation multipliers {median.round(3)}", flush=True)
    ax.set_xticks(np.arange(4), TITLES)
    ax.set_xlim(-0.45, 3.45)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 25)
    ax.set_yticks([1, 2, 5, 10, 20], ["1×", "2×", "5×", "10×", "20×"])
    ax.axhline(1, color="#606970", linestyle="--", linewidth=1)
    ax.text(0.99, 1.07, "1× = unchanged error", ha="right", color="#606970",
            transform=ax.get_yaxis_transform(), fontsize=10)
    save_figure(probe_fig, "17-local-perturbation-error", "DNO error after a 10% potential perturbation",
                "Perturbed error / original error")

    metadata = json.loads((ROOT / "outputs/operator_probe_a100_20261001/operator_probes.json").read_text())
    with np.load(ROOT / metadata["cases"]) as cases:
        coefficients = np.fft.rfft(cases["xi"], axis=-1, norm="forward")
        coefficients[:, 0] = 0
        coefficients[:, 129:] = 0
        original = np.fft.irfft(coefficients, n=1024, axis=-1, norm="forward")
        phases = np.random.default_rng(metadata["seed"]).uniform(
            0, 2 * np.pi, size=(len(original), 4, 513))
        changes = 0.1 * np.fft.irfft(np.abs(coefficients)[:, None] * np.exp(1j * phases),
                                   n=1024, axis=-1, norm="forward")
        np.testing.assert_allclose(
            np.linalg.norm(changes, axis=-1) / np.linalg.norm(original, axis=-1)[:, None],
            0.1, rtol=1e-12)
        example_fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey="col")
        x = np.arange(1024) / 1024
        for family_index, title in enumerate(TITLES):
            case = np.flatnonzero(cases["family_ids"] == family_index)[0]
            scale = np.sqrt(np.mean(original[case]**2))
            base, change = original[case] / scale, changes[case, 0] / scale
            axes[0, family_index].plot(x, base, color="#606970", linewidth=1.6, label="Original potential")
            axes[0, family_index].plot(x, base + change, color="#c45b22", linewidth=1.6,
                                       linestyle="--", label="Perturbed potential")
            axes[0, family_index].set_title(title, loc="left")
            axes[1, family_index].plot(x, change, color="#007da6", linewidth=1.6,
                                       label="Added change (10% RMS)")
            axes[1, family_index].axhline(0, color="#bfc6cc", linewidth=0.7)
            axes[1, family_index].set_xlabel("x / L")
            axes[1, family_index].set_xticks([0, 0.5, 1])
            axes[1, family_index].set_xlim(0, 1)
            print(f"Perturbation example {title}: simulation {cases['simulation_ids'][case]}, probe 0", flush=True)
        axes[0, 0].set_ylabel("Potential")
        axes[1, 0].set_ylabel("Added change")
    save_figure(example_fig, "18-local-perturbation-examples", "Examples of the 10% potential perturbations",
                "Units of original potential RMS")


if __name__ == "__main__":
    main()
