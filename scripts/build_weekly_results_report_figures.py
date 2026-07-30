#!/usr/bin/env python3
"""Build CPU-only figures and statistics for the 2026-07-16 weekly report."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analyze_neutral_multiarm import (
    optimal_displacement,
    periodic_shift,
    relative_l2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / (
    "outputs/c21_tangent_w100_from_c20_20260714_171442/"
    "eval_final_heldout_unguarded_n256_20260715_035324"
)
ORACLE_ROOT = REPO_ROOT / "outputs/c21_tangent_w100_from_c20_20260714_171442"
FAMILIES = (
    "tanaka_g0",
    "tanaka_g1",
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "linear",
    "random_sea_deep",
    "random_sea_finite",
    "stokes_deep",
    "stokes_finite",
)
FAMILY_LABELS = {
    "tanaka_g0": "Tanaka $g_0$",
    "tanaka_g1": "Tanaka $g_1$",
    "bf_g0": "Benjamin--Feir $g_0$",
    "bf_g1": "Benjamin--Feir $g_1$",
    "bf_modal": "Benjamin--Feir modal",
    "linear": "Linear registry",
    "random_sea_deep": "Random sea, deep",
    "random_sea_finite": "Random sea, finite",
    "stokes_deep": "Stokes, deep",
    "stokes_finite": "Stokes, finite",
}
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
THRESHOLDS = (0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class FamilyData:
    name: str
    attempted: int
    valid: int
    truth_invalid: int
    nonfinite_valid: int
    errors: np.ndarray
    aligned_errors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "notes/figures/weekly_results_20260716",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=REPO_ROOT / "notes/weekly_results_stats_20260716.json",
    )
    return parser.parse_args()


def load_family(name: str) -> FamilyData:
    path = EVAL_ROOT / name / f"{name}_trajs.npz"
    with np.load(path, allow_pickle=False) as archive:
        truth_valid = np.asarray(archive["truth_valid"], dtype=bool)
        model_nonfinite = np.asarray(archive["model_nonfinite_any"], dtype=bool)
        final_error = np.asarray(archive["rel_l2_eta"][-1], dtype=np.float64)
        truth_eta = np.asarray(archive["truth_eta"][-1], dtype=np.float64)
        pred_eta = np.asarray(archive["pred_eta"][-1], dtype=np.float64)
    scored = truth_valid & ~model_nonfinite & np.isfinite(final_error)
    aligned_errors = np.asarray(
        tuple(
            relative_l2(
                periodic_shift(
                    prediction,
                    -optimal_displacement(prediction, truth, 2.0 * math.pi),
                    2.0 * math.pi,
                ),
                truth,
            )
            for prediction, truth in zip(pred_eta[scored], truth_eta[scored], strict=True)
        ),
        dtype=np.float64,
    )
    if np.any(aligned_errors > final_error[scored] + 2e-7):
        raise RuntimeError(f"continuous alignment increased raw error for {name}")
    return FamilyData(
        name=name,
        attempted=int(truth_valid.size),
        valid=int(np.sum(truth_valid)),
        truth_invalid=int(np.sum(~truth_valid)),
        nonfinite_valid=int(np.sum(truth_valid & model_nonfinite)),
        errors=final_error[scored],
        aligned_errors=aligned_errors,
    )


def exact_zero_event_upper_bound(n: int, alpha: float = 0.05) -> float:
    return float("nan") if n <= 0 else 1.0 - alpha ** (1.0 / n)


def summarize_errors(errors: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(errors, QUANTILES)
    counts = {str(threshold): int(np.sum(errors > threshold)) for threshold in THRESHOLDS}
    return {
        "n": int(errors.size),
        "mean": float(np.mean(errors)),
        "quantiles": {
            f"q{int(round(100 * q)):02d}": float(value)
            for q, value in zip(QUANTILES, quantiles, strict=True)
        },
        "maximum": float(np.max(errors)),
        "threshold_counts": counts,
        "threshold_rates": {
            threshold: count / int(errors.size) for threshold, count in counts.items()
        },
    }


def write_statistics(data: Sequence[FamilyData], path: Path) -> dict[str, Any]:
    pooled = np.concatenate(tuple(item.errors for item in data))
    pooled_aligned = np.concatenate(tuple(item.aligned_errors for item in data))
    attempted = sum(item.attempted for item in data)
    valid = sum(item.valid for item in data)
    nonfinite_valid = sum(item.nonfinite_valid for item in data)
    macro = json.loads((EVAL_ROOT / "macro_summary.json").read_text())
    result: dict[str, Any] = {
        "protocol": {
            "checkpoint": "C21 final",
            "evaluation_root": str(EVAL_ROOT.relative_to(REPO_ROOT)),
            "panels": len(data),
            "attempted": attempted,
            "truth_valid": valid,
            "truth_invalid": attempted - valid,
            "model_nonfinite_on_truth_valid": nonfinite_valid,
            "zero_nonfinite_one_sided_95pct_upper_rate": (
                exact_zero_event_upper_bound(valid)
            ),
            "primary_metric": "terminal relative L2 elevation error, unaligned",
        },
        "pooled_truth_valid": summarize_errors(pooled),
        "pooled_truth_valid_continuously_aligned": summarize_errors(pooled_aligned),
        "equal_panel_macro": {
            key: value
            for key, value in macro.items()
            if key.startswith("rel_l2_eta_")
            or key.startswith("terminal_eta_failure_rate_")
            or key.startswith("model_nonfinite_any_rate_")
        },
        "families": {
            item.name: {
                "attempted": item.attempted,
                "truth_valid": item.valid,
                "truth_invalid": item.truth_invalid,
                "model_nonfinite_on_truth_valid": item.nonfinite_valid,
                "zero_nonfinite_one_sided_95pct_upper_rate": (
                    exact_zero_event_upper_bound(item.valid)
                ),
                **summarize_errors(item.errors),
                "continuously_aligned": summarize_errors(item.aligned_errors),
            }
            for item in data
        },
    }
    expected_counts = {"0.25": 83, "0.5": 23, "0.75": 11, "1.0": 6}
    if result["pooled_truth_valid"]["threshold_counts"] != expected_counts:
        raise RuntimeError(
            "fresh-suite threshold counts disagree with the audited macro summary: "
            f"{result['pooled_truth_valid']['threshold_counts']}"
        )
    if (attempted, valid, nonfinite_valid) != (2560, 2508, 0):
        raise RuntimeError(
            "fresh-suite accounting mismatch: "
            f"attempted={attempted}, valid={valid}, nonfinite={nonfinite_valid}"
        )
    expected_aligned_counts = {"0.25": 17, "0.5": 5, "0.75": 1, "1.0": 0}
    aligned_counts = result["pooled_truth_valid_continuously_aligned"][
        "threshold_counts"
    ]
    if aligned_counts != expected_aligned_counts:
        raise RuntimeError(
            "aligned fresh-suite threshold counts disagree with the audited result: "
            f"{aligned_counts}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_family_quantiles(data: Sequence[FamilyData], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    positions = np.arange(len(data))[::-1]
    for position, item in zip(positions, data, strict=True):
        q25, q50, q75, q90, q95, q99 = np.quantile(item.errors, QUANTILES)
        axis.plot([q90, q99], [position, position], color="#4477AA", lw=1.4)
        axis.plot([q25, q75], [position, position], color="#4477AA", lw=6.0)
        axis.scatter(q50, position, color="#111111", marker="o", s=26, zorder=3)
        axis.scatter(q95, position, color="#CC6677", marker="D", s=26, zorder=3)
        aligned_q95 = np.quantile(item.aligned_errors, 0.95)
        axis.scatter(
            aligned_q95,
            position,
            facecolor="white",
            edgecolor="#228833",
            marker="s",
            s=30,
            zorder=3,
        )
    axis.axvline(0.25, color="#DDCC77", ls="--", lw=1.4, label="error 0.25")
    axis.axvline(1.0, color="#AA3377", ls=":", lw=1.7, label="error 1")
    axis.set_xscale("log")
    axis.set_yticks(positions, tuple(FAMILY_LABELS[item.name] for item in data))
    axis.set_xlabel(r"terminal relative elevation error $e_\eta(T)$")
    axis.set_title("C21 fresh held-out suite: familywise error quantiles")
    axis.grid(axis="x", which="both", alpha=0.22)
    axis.legend(
        handles=(
            plt.Line2D([], [], color="#4477AA", lw=6, label="interquartile range"),
            plt.Line2D([], [], color="#111111", marker="o", lw=0, label="median"),
            plt.Line2D([], [], color="#CC6677", marker="D", lw=0, label="95th percentile"),
            plt.Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="white",
                markeredgecolor="#228833",
                lw=0,
                label="aligned 95th percentile",
            ),
            plt.Line2D([], [], color="#4477AA", lw=1.4, label="90th--99th percentile"),
            plt.Line2D([], [], color="#DDCC77", ls="--", label="error 0.25"),
            plt.Line2D([], [], color="#AA3377", ls=":", label="error 1"),
        ),
        fontsize=8,
        ncol=2,
        loc="lower right",
    )
    save_figure(fig, output_dir, "c21_heldout_family_quantiles")


def empirical_exceedance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(errors)
    probability = (values.size - np.arange(values.size)) / values.size
    return values, probability


def plot_exceedance(data: Sequence[FamilyData], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    pooled = np.concatenate(tuple(item.errors for item in data))
    pooled_aligned = np.concatenate(tuple(item.aligned_errors for item in data))
    selected = {
        "All truth-valid": pooled,
        "All, one-shift aligned": pooled_aligned,
        "Tanaka $g_0$": next(item.errors for item in data if item.name == "tanaka_g0"),
        "Tanaka $g_1$": next(item.errors for item in data if item.name == "tanaka_g1"),
        "BF modal": next(item.errors for item in data if item.name == "bf_modal"),
    }
    colors = ("#111111", "#777777", "#4477AA", "#EE6677", "#228833")
    for (label, errors), color in zip(selected.items(), colors, strict=True):
        values, probability = empirical_exceedance(errors)
        positive = values > 0.0
        line_style = "--" if label == "All, one-shift aligned" else "-"
        axis.step(
            values[positive],
            probability[positive],
            where="post",
            label=label,
            color=color,
            ls=line_style,
        )
    for threshold in THRESHOLDS:
        axis.axvline(threshold, color="#888888", lw=0.8, ls="--", alpha=0.55)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(2e-3, 1.05)
    axis.set_xlabel(r"threshold $\tau$ for $e_\eta(T)$")
    axis.set_ylabel(r"empirical exceedance $\Pr[e_\eta(T)>\tau]$")
    axis.set_title("Finite-error tails on fresh trajectory-held-out initial conditions")
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    save_figure(fig, output_dir, "c21_heldout_exceedance")


def tail_label(path: Path) -> str:
    match = re.search(r"(tanaka_g[01])_idx(\d+)", path.stem)
    if match is None:
        return "BF modal, index 22"
    family, index = match.groups()
    return f"{FAMILY_LABELS[family]}, index {index}"


def plot_tail_trajectories(output_dir: Path) -> None:
    series_root = EVAL_ROOT / "gxi_phase_audit_tau1"
    paths = sorted(series_root.glob("tanaka_*.npz"))
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.0), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(paths)))
    dx = 2.0 * math.pi / 1024.0
    for path, color in zip(paths, colors, strict=True):
        with np.load(path, allow_pickle=False) as archive:
            times = np.asarray(archive["times"])
            raw = np.asarray(archive["raw_eta_error"])
            aligned = np.asarray(archive["aligned_eta_error"])
            displacement = np.asarray(archive["displacement"]) / dx
        label = tail_label(path)
        axes[0].plot(times, raw, color=color, lw=1.8, label=label)
        axes[0].plot(times, aligned, color=color, lw=1.1, ls="--")
        axes[1].plot(times, displacement, color=color, lw=1.8, label=label)
    axes[0].axhline(0.25, color="#777777", ls=":", lw=1.2)
    axes[0].axhline(1.0, color="#333333", ls="--", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"relative elevation error")
    axes[0].set_title("Five finite C21 Tanaka tails: raw (solid) and aligned (dashed)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(0.0, color="#777777", lw=0.8)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("best-fit displacement (grid points)")
    axes[1].set_title("The dominant error grows as a smooth secular translation")
    for axis in axes:
        axis.grid(alpha=0.22)
    save_figure(fig, output_dir, "c21_worst_tanaka_error_growth")

    bf_path = series_root / "bf_modal_idx22_gxi_phase_series.npz"
    with np.load(bf_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times"])
        raw = np.asarray(archive["raw_eta_error"])
        aligned = np.asarray(archive["aligned_eta_error"])
    fraction_removed = 1.0 - aligned**2 / (raw**2 + 1e-30)
    fraction_removed = np.where(raw > 1e-6, fraction_removed, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    axes[0].plot(times, raw, label="raw", color="#4477AA", lw=1.8)
    axes[0].plot(times, aligned, label="one-shift aligned", color="#CC6677", lw=1.8)
    axes[0].axhline(1.0, color="#333333", ls="--", lw=1.0)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"relative elevation error")
    axes[0].set_title("BF-modal index 22")
    axes[0].legend()
    axes[1].plot(times, fraction_removed, color="#228833", lw=1.8)
    axes[1].axhline(1.0, color="#777777", lw=0.8)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("squared error removed by one shift")
    axes[1].set_title("A single shift does not repair modal transfer")
    for axis in axes:
        axis.grid(alpha=0.22)
    save_figure(fig, output_dir, "c21_worst_bf_modal_error_growth")


def load_oracle(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def plot_oracle_discriminator(output_dir: Path) -> None:
    case24 = load_oracle(
        ORACLE_ROOT / "neutral_multiarm_case1000024_20260716_013336/summary.json"
    )
    case27 = load_oracle(
        ORACLE_ROOT / "neutral_multiarm_case27_20260716_015100/summary.json"
    )
    arms = (
        ("baseline", "C21 baseline"),
        ("modal", "modal phase"),
        ("localized_sigma1", r"local $h$"),
        ("localized_sigma4", r"local $4h$"),
        ("radial_complement", "radial complement"),
        ("full_defect", "full defect"),
    )

    def metric(summary: dict[str, Any], arm: str, name: str) -> float:
        if arm == "baseline":
            return float(summary["baseline"][name])
        return float(summary["arms"][arm]["terminal_metrics"][name])

    locations = np.arange(len(arms))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    raw24 = [metric(case24, arm, "raw_eta_error") for arm, _ in arms]
    raw27 = [metric(case27, arm, "raw_eta_error") for arm, _ in arms]
    aligned24 = [metric(case24, arm, "aligned_eta_error") for arm, _ in arms]
    aligned27 = [metric(case27, arm, "aligned_eta_error") for arm, _ in arms]
    axis.bar(locations - width / 2, raw24, width, color="#4477AA", label="case 24 raw")
    axis.bar(locations + width / 2, raw27, width, color="#EE6677", label="case 27 raw")
    axis.scatter(locations - width / 2, aligned24, marker="o", color="#002B55", s=26, label="case 24 aligned")
    axis.scatter(locations + width / 2, aligned27, marker="D", color="#881133", s=24, label="case 27 aligned")
    axis.axhline(0.25, color="#777777", ls="--", lw=1.0)
    axis.set_yscale("log")
    axis.set_ylim(1e-9, 2.2)
    axis.set_xticks(locations, tuple(label for _, label in arms), rotation=18, ha="right")
    axis.set_ylabel(r"terminal relative elevation error")
    axis.set_title("Stagewise defect oracle: reduced corrections do not transfer")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(fontsize=8, ncol=2, loc="lower left")
    save_figure(fig, output_dir, "c21_oracle_cross_case")


def extract_final_gif_frames(output_dir: Path) -> list[Path]:
    gif_root = EVAL_ROOT / "worst_case_gifs_tau1"
    frame_dir = output_dir / "failure_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for gif_path in sorted(gif_root.glob("*.gif")):
        with Image.open(gif_path) as image:
            image.seek(image.n_frames - 1)
            frame = image.convert("RGB")
            output = frame_dir / f"{gif_path.stem}_final.png"
            frame.save(output, optimize=True)
            outputs.append(output)
    if len(outputs) != 6:
        raise RuntimeError(f"expected six worst-case GIFs, extracted {len(outputs)}")
    return outputs


def main() -> None:
    args = parse_args()
    data = tuple(map(load_family, FAMILIES))
    stats = write_statistics(data, args.stats_json)
    plot_family_quantiles(data, args.output_dir)
    plot_exceedance(data, args.output_dir)
    plot_tail_trajectories(args.output_dir)
    plot_oracle_discriminator(args.output_dir)
    frames = extract_final_gif_frames(args.output_dir)
    print(
        json.dumps(
            {
                "stats": str(args.stats_json),
                "figures": str(args.output_dir),
                "truth_valid": stats["protocol"]["truth_valid"],
                "threshold_counts": stats["pooled_truth_valid"]["threshold_counts"],
                "extracted_frames": [str(path) for path in frames],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
