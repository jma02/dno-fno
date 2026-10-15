"""Plot phase and local-potential errors from saved evaluations, CPU only."""

from __future__ import annotations

import argparse
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
from scipy.optimize import brentq

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
    ax: Axes, times: np.ndarray, values: np.ndarray, label: str, color: str,
    *, shade: bool = True, linestyle: str = "-",
) -> None:
    low, median, high = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    if shade:
        ax.fill_between(times, low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(times, median, label=label, color=color, linewidth=1.8, linestyle=linestyle)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hero", action="store_true", help="Plot saved hero-run epochs 1, 6 and 7 only.")
    parser.add_argument("--phase-example", action="store_true", help="Show actual phase drift in a representative epoch-7 Stokes wave.")
    parser.add_argument("--phase-training", action="store_true", help="Plot terminal phase error across saved EMA training epochs.")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.edgecolor": "#bfc6cc", "xtick.color": "#626c76",
        "ytick.color": "#626c76", "axes.titlesize": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if args.phase_training:
        epochs = np.arange(1, 11)
        evaluations = [
            ROOT / "outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/eval_fast_weight_ema_nestedfused_contextbatch",
            *(ROOT / f"outputs/c27_nested_b16_ema_low_lr_continue_20261014/eval_epoch{epoch}_weight_ema" for epoch in (2, 3)),
            *(ROOT / f"outputs/c27_nested_b16_ema_low_lr_ep60_20261014/eval_epoch{epoch}_weight_ema" for epoch in epochs[3:]),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for family, title, ax in zip(FAMILIES, TITLES, axes.flat, strict=True):
            medians = []
            for directory in evaluations:
                path = directory / f"{family}_trajs.npz"
                with np.load(path) as data:
                    reference = np.fft.rfft(data["truth_eta"][-1], axis=-1)
                    prediction = np.fft.rfft(data["pred_eta"][-1], axis=-1)
                    final_time = data["times"][-1]
                weights = np.full(reference.shape[-1], 2.)
                weights[[0, -1]] = 1
                total = np.sum(weights * np.abs(prediction - reference)**2, axis=-1)
                amplitude = np.sum(weights * (np.abs(prediction) - np.abs(reference))**2, axis=-1)
                norm = np.sum(weights * np.abs(reference)**2, axis=-1)
                phase_error = 100 * np.sqrt(np.maximum(total - amplitude, 0) / norm)
                if not np.isfinite(phase_error).all():
                    raise ValueError(f"Undefined phase decomposition: {path}")
                medians.append(float(np.median(phase_error)))
            ax.plot(epochs, medians, "o-", color="#c45b22", linewidth=1.8,
                    markersize=5, label="Median of 32 cases · weight EMA")
            ax.set_title(title, loc="left")
            ax.set_title(f"T = {final_time:g}", loc="right", fontsize=11, color="#808a94")
            ax.set(xlabel="Training epoch", xticks=epochs, xlim=(0.8, epochs[-1] + 0.2), ylim=(0, 1.25 * max(medians)))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.locator_params(axis="y", nbins=5)
            reduction = 100 * (1 - medians[-1] / medians[0])
            ax.text(0.97, 0.92, f"{reduction:.0f}% lower than epoch 1", transform=ax.transAxes,
                    ha="right", va="top", color="#c45b22", fontsize=11)
            print(f"{family}: phase-error medians (%)={medians}; reduction={reduction:.3f}%", flush=True)
        save_figure(fig, "22-phase-error-vs-training", "Phase-related surface error across training",
                    "Final relative surface error from phase mismatch (%)")
        return
    if args.phase_example:
        path = ROOT / "outputs/c27_nested_b16_ema_low_lr_ep60_20261014/eval_epoch7_weight_ema/stokes_trajs.npz"
        with np.load(path) as data:
            case = np.flatnonzero(data["simulation_ids"] == 1901833).item()
            reference, prediction = data["truth_eta"][-1, case], data["pred_eta"][-1, case]
            final_time = data["times"][-1]
        length, size = 2 * np.pi, reference.size
        k = np.fft.fftfreq(size, d=length / size) * 2 * np.pi
        reference_hat = np.fft.fft(reference)
        carrier = np.argmax(np.abs(reference_hat[1:size // 2])) + 1
        wavelength = length / carrier
        shift = optimal_displacement(prediction, reference, length)
        shift = (shift + wavelength / 2) % wavelength - wavelength / 2
        aligned = aligned_fields(prediction[None], np.array([shift]), length)[0]
        coefficients = np.fft.fft(np.stack((reference, prediction, aligned))) / size
        crossings = np.flatnonzero((reference <= 0) & (np.roll(reference, -1) > 0))
        left = crossings[np.argmin(np.abs(crossings - size / 2))] * length / size
        crossing = brentq(lambda x: (np.exp(1j * k * x) @ coefficients[0]).real,
                         left, left + length / size, xtol=1e-14)
        predicted_crossing = brentq(lambda x: (np.exp(1j * k * x) @ coefficients[1]).real,
                                   crossing - 5 * abs(shift), crossing + 5 * abs(shift), xtol=1e-14)
        positions = crossing + np.linspace(-wavelength / 2, wavelength / 2, 2048)
        zoom = crossing + np.linspace(-4 * abs(shift), 5 * abs(shift), 512)
        fields = (np.exp(1j * positions[:, None] * k) @ coefficients.T).real
        zoom_fields = (np.exp(1j * zoom[:, None] * k) @ coefficients.T).real
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for index, (label, color, style) in enumerate((
            ("Reference", "#687580", "-"), ("Prediction", "#c45b22", "--"),
            ("Aligned prediction", "#007da6", ":"),
        )):
            axes[0].plot(positions - crossing, fields[:, index], color=color, linestyle=style, label=label)
            axes[1].plot(zoom - crossing, zoom_fields[:, index], color=color, linestyle=style)
        axes[0].set(title="Actual wave profiles", xlabel="Position relative to reference crossing", ylabel="Surface elevation")
        axes[1].set(title="Zoom: the horizontal offset", xlabel="Position relative to reference crossing", ylabel="Surface elevation")
        axes[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True, useOffset=False)
        crossing_shift = predicted_crossing - crossing
        axes[1].annotate("", xy=(0, 0), xytext=(crossing_shift, 0),
                            arrowprops={"arrowstyle": "<->", "color": "#343a40"})
        axes[1].annotate("Horizontal shift", xy=(crossing_shift / 2, 0),
                            xytext=(0, -16), textcoords="offset points", ha="center", fontsize=10)
        raw_error, aligned_error = prediction - reference, aligned - reference
        removed = 100 * (1 - np.dot(aligned_error, aligned_error) / np.dot(raw_error, raw_error))
        for index, color in ((1, "#c45b22"), (2, "#007da6")):
            axes[2].plot(positions - crossing, fields[:, index] - fields[:, 0], color=color)
        axes[2].set(title=f"Alignment removes {removed:.1f}% of squared error",
                       xlabel="Position relative to reference crossing", ylabel="Prediction minus reference")
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True, useOffset=False)
        save_figure(fig, "21-phase-drift-example", f"Stokes phase drift · epoch 7 · time {final_time:g}", "")
        print(f"Stokes ID 1901833: fitted shift={shift:.9g}, crossing shift={crossing_shift:.9g}; alignment removes {removed:.6f}% of squared error", flush=True)
        return
    if args.hero:
        run = ROOT / "outputs/c27_nested_b16_ema_low_lr_ep60_20261014"
        evaluations = (
            (ROOT / "outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/eval_fast_weight_ema_nestedfused_contextbatch", "Epoch 1", "#687580"),
            (run / "eval_epoch6_weight_ema", "Epoch 6", "#007da6"),
            (run / "eval_epoch7_weight_ema", "Epoch 7", "#c45b22"),
        )
        drift_fig, drift_axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
        phase_fig, phase_axes = plt.subplots(2, 2, figsize=(11, 8))
        share_fig, share_axes = plt.subplots(2, 2, figsize=(11, 8))
        for family_index, family in enumerate(FAMILIES):
            for directory, label, color in evaluations:
                path = directory / f"{family}_trajs.npz"
                with np.load(path) as data:
                    times = data["times"]
                    truth, pred = data["truth_eta"], data["pred_eta"]
                if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
                    raise ValueError(f"Nonfinite saved trajectory: {path}")
                reference_hat = np.fft.rfft(truth[1:], axis=-1)
                prediction_hat = np.fft.rfft(pred[1:], axis=-1)
                weights = np.full(reference_hat.shape[-1], 2.)
                weights[[0, -1]] = 1
                total = np.sum(weights * np.abs(prediction_hat - reference_hat)**2, axis=-1)
                amplitude = np.sum(weights * (np.abs(prediction_hat) - np.abs(reference_hat))**2, axis=-1)
                phase = np.maximum(total - amplitude, 0)
                norm = np.sum(weights * np.abs(reference_hat)**2, axis=-1)
                phase_error = 100 * np.sqrt(phase / norm)
                phase_share = 100 * phase / total
                if not (np.isfinite(phase_error).all() and np.isfinite(phase_share).all()):
                    raise ValueError(f"Undefined phase decomposition: {path}")
                for axes, values in ((phase_axes, phase_error), (share_axes, phase_share)):
                    ax = axes.flat[family_index]
                    median_band(ax, times[1:], values, label, color,
                                shade=label == "Epoch 7", linestyle="--" if label == "Epoch 1" else "-")
                    ax.set_title(f"{TITLES[family_index]} · {truth.shape[1]} cases", loc="left")
                    ax.set(xlabel="Time", xlim=(0, times[-1]))
                phase_axes.flat[family_index].set_yscale("log")
                share_axes.flat[family_index].set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
                print(f"{label}, {family}: median terminal phase error={np.median(phase_error[-1]):.6g}%; phase share={np.median(phase_share[-1]):.3f}%", flush=True)
                if family_index >= 2:
                    continue
                groups = json.loads(path.with_name(f"{family}_summary.json").read_text())["evaluation_source"]["selected_parameter_group_ids"]
                selected = np.array([family == "stokes" or "_m1_" in group for group in groups])
                reference = truth[10::10, selected].reshape(-1, truth.shape[-1])
                prediction = pred[10::10, selected].reshape(reference.shape)
                shifts = np.array([optimal_displacement(p, r, 2 * np.pi)
                                   for p, r in zip(prediction, reference, strict=True)])
                aligned = aligned_fields(prediction, shifts, 2 * np.pi)
                raw_error = np.sum((prediction - reference)**2, axis=-1)
                aligned_error = np.minimum(raw_error, np.sum((aligned - reference)**2, axis=-1))
                removed = (100 * (1 - aligned_error / raw_error)).reshape(-1, selected.sum())
                shifts = shifts.reshape(removed.shape)
                carrier = (np.argmax(np.abs(np.fft.rfft(truth[0, selected]))[:, 1:], axis=-1) + 1
                           if family == "stokes" else np.ones(selected.sum()))
                period = 2 * np.pi / carrier
                shifts = (shifts + period / 2) % period - period / 2
                shifts = np.unwrap(shifts * carrier, axis=0) / carrier
                if not (np.isfinite(shifts).all() and np.isfinite(removed).all()
                        and np.all(removed <= phase_share[9::10, selected] + 1e-5)):
                    raise ValueError(f"Inconsistent translation decomposition: {path}")
                median_band(drift_axes[0, family_index], times[10::10], 100 * np.abs(shifts) / (2 * np.pi), label, color)
                median_band(drift_axes[1, family_index], times[10::10], removed, label, color)
                title = "Stokes" if family == "stokes" else "Isolated Tanaka"
                drift_axes[0, family_index].set_title(f"{title} · {selected.sum()} cases", loc="left")
                drift_axes[0, family_index].set_yscale("log")
                drift_axes[1, family_index].set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], xlabel="Time", xlim=(0, times[-1]))
                print(f"{label}, {family}: median terminal absolute shift={np.median(100 * np.abs(shifts[-1]) / (2 * np.pi)):.6g}% of domain; alignment removes={np.median(removed[-1]):.3f}% of squared error", flush=True)
        drift_axes[0, 0].set_ylabel("Absolute translation drift\n(% of domain length)")
        drift_axes[1, 0].set_ylabel("Squared error removed\nby translation alignment (%)")
        save_figure(drift_fig, "19-hero-translation-drift", "Translation drift during the hero run", "")
        save_figure(phase_fig, "20-hero-fourier-phase", "Surface error from phase mismatch",
                    "Relative surface error from phase mismatch (%)")
        save_figure(share_fig, "20b-hero-fourier-phase-share", "How much surface error comes from phase?",
                    "Phase share of squared surface error (%)")
        return
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
