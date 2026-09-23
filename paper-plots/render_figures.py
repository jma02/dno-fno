"""Render the nine manuscript placeholders from epoch-40 hard-P128 diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
from pypdf import PdfWriter  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
FIGURES.mkdir(exist_ok=True)
C27, CS = "#146e9e", "#272b30"
COLORS = ("#4263a5", "#21918c", "#ce8732", "#9764a5")
FAMILIES = ("Stokes", "Tanaka", "JONSWAP/TMA", "Benjamin–Feir")
LENGTH = 2 * np.pi
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#bcc2c9", "axes.linewidth": 0.7,
    "grid.color": "#e7ebef", "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "savefig.dpi": 200, "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans", "figure.facecolor": "white",
})
NAMES: list[str] = []
CAPTIONS: dict[str, str] = {}


def finish(fig: Figure, name: str, caption: str, footer: str) -> None:
    fig.text(0.01, 0.008, footer, fontsize=7, color="#65717f", va="bottom")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    NAMES.append(name)
    CAPTIONS[name] = caption
    print(f"rendered {name}", flush=True)


def median_band(ax: Axes, x: np.ndarray, values: np.ndarray, color: str, label: str) -> None:
    low, median, high = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, median, color=color, label=label)


if __name__ == "__main__":
    with np.load(OUT / "plot-data.npz") as archive:
        roll = {key.removeprefix("rollout_"): archive[key] for key in archive.files if key.startswith("rollout_")}
        snapshot = {key.removeprefix("snapshot_"): archive[key] for key in archive.files if key.startswith("snapshot_")}
        benchmark = json.loads(str(archive["benchmark_metadata"]))
    x = np.arange(1024) / 1024
    panel_note = "Epoch-40 C27 · 128 held-out simulations · no adaptive damping · N = 1024 · g = 1"

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.7), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.045, 1, 0.955))
    for family, ax in enumerate(axes.flat):
        eta = snapshot[f"example_{family}_eta"]
        h = float(snapshot[f"example_{family}_h"])
        t = float(snapshot[f"example_{family}_time"])
        peak = 1 + np.argmax(abs(np.fft.rfft(eta))[1:])
        amplitude = 0.5 * np.ptp(eta)
        ax.plot(x, eta, color=COLORS[family], lw=1.6)
        ax.fill_between(x, 0, eta, color=COLORS[family], alpha=0.08)
        ax.set(title=f"({chr(97 + family)}) {FAMILIES[family]}", xlabel="$x/L$", ylabel=r"$\eta(x)$", xlim=(0, 1))
        ax.text(0.98, 0.94, f"$k_p h={peak * h:.2g}$\n$a/h={amplitude / h:.2g}$ · $t={t:.2g}$",
                ha="right", va="top", transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.grid(alpha=0.6)
    finish(fig, "01-wave-regimes",
           "Four training snapshots from the equal-family hard-P128 dataset used by the epoch-40 C27 checkpoint, chosen at the median sampled DNO error within each family. The peak wavenumber is the strongest nonzero surface Fourier mode; a is half the surface range. Snapshot row IDs are in plot-data.npz.",
           "Equal-family hard-P128 training snapshots · kₚ from the surface spectrum · a = (max η − min η)/2")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.06, 1, 0.94))
    for split, offset, color, label in ((0, -0.10, "#9aa6b4", "Training rows"), (2, 0.10, C27, "Held-out rows")):
        for family in range(4):
            values = snapshot["error"][(snapshot["family"] == family) & (snapshot["split"] == split)]
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            axes[0].errorbar(family + offset, median, yerr=[[median - q1], [q3 - median]],
                            color=color, fmt="o", capsize=3, label=label if family == 0 else None)
    bins = np.digitize(snapshot["kh"], [np.pi / 10, np.pi])
    for depth_bin, (color, marker, label) in enumerate(zip(
        ("#21918c", "#d5973c", "#6964a5"), ("o", "s", "^"),
        (r"$k_ph<\pi/10$", r"$\pi/10\leq k_ph<\pi$", r"$k_ph\geq\pi$"), strict=True
    )):
        for family in range(4):
            values = snapshot["error"][(snapshot["family"] == family) & (snapshot["split"] == 2) & (bins == depth_bin)]
            if len(values) == 0:
                continue
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            axes[1].errorbar(family + 0.20 * (depth_bin - 1), median,
                            yerr=[[median - q1], [q3 - median]], color=color, fmt=marker, capsize=3)
        axes[1].plot([], [], color=color, marker=marker, ls="none", label=label)
    for ax, title in zip(axes, ("(a) Simulation-split accuracy", "(b) Test rows by depth"), strict=True):
        ax.set(title=title, yscale="log", ylabel=r"Relative DNO error $e_G$", xticks=range(4), xticklabels=FAMILIES, xlim=(-0.5, 3.5))
        ax.grid(axis="y", which="major")
        ax.legend(loc="upper left", fontsize=7)
    finish(fig, "02-dno-class-errors",
           "Measured epoch-40 C27 relative L2 error against the stored hard-P128 DNO targets: 512 uniformly sampled rows per family from each of the saved training and test splits. Complete simulations belong to only one split. Points and bars show medians and interquartile ranges. Right panel uses test rows only, grouped by the stated peak-wavenumber depth bins; missing groups contain no sampled rows. These are sampled errors, not exhaustive dataset statistics.",
           "512 rows / family / split · median + interquartile range · simulation-disjoint train/test split")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.65), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.075, 1, 0.87))
    for ax, field, title in zip(axes, ("eta", "xi"), (r"(a) Surface elevation $e_\eta(t)$", r"(b) Potential $e_\xi(t)$"), strict=True):
        for family in range(4):
            selected = roll[f"{field}_error"][1:, roll["family"] == family]
            ax.plot(roll["tau"][1:], np.median(selected, axis=1), color=COLORS[family], lw=1, alpha=0.85, label=FAMILIES[family])
        median_band(ax, roll["tau"][1:], roll[f"{field}_error"][1:], CS, "Pooled C27")
        ax.set(title=title, xlabel="Normalized time $t/T$", ylabel="Relative $L^2$ error", yscale="log", xlim=(0, 1))
        ax.grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=5, frameon=False)
    finish(fig, "03-trajectory-errors",
           "Epoch-40 C27 trajectory errors on 32 parameter-stratified simulations per family from the saved test split (128 total). The black curve and shading are the pooled median and interquartile range; thin colored curves are family medians. Times are normalized separately by T=20 for Stokes/JONSWAP-TMA and T=200 for Tanaka/Benjamin–Feir. All 128 reference solves are valid and saved predictions are finite with positive fluid depth. Learned inference uses FP32 with FP64 physics and integration, without adaptive damping. This is a selected test panel, not the complete test set. No matched vanilla-FNO baseline is included.",
           panel_note + " · T = 20 or 200; normalized time")

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 7.1), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.04, 1, 0.94))
    for col, name in enumerate(("stokes", "tanaka")):
        times = roll[f"{name}_times"]
        truth = roll[f"{name}_truth_eta"][-1]
        prediction = roll[f"{name}_pred_eta"][-1]
        aligned = roll[f"{name}_aligned"][-1]
        raw_error = np.linalg.norm(prediction - truth) / np.linalg.norm(truth)
        aligned_error = float(roll[f"{name}_aligned_error"][-1])
        axes[0, col].plot(times, roll[f"{name}_shift"] / LENGTH, color=C27)
        axes[0, col].set(title=f"{'Stokes' if name == 'stokes' else 'Isolated Tanaka'} · case {int(roll[f'{name}_case_id'])}", xlabel="$t$", ylabel=r"Translation $\Delta x/L$")
        axes[0, col].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        for row, values, error, title in ((1, prediction, raw_error, "Unaligned"), (2, aligned, aligned_error, "Translation-aligned")):
            axes[row, col].plot(x, truth, color=CS, lw=1.7, label="CS reference")
            axes[row, col].plot(x, values, color=C27, ls="--", label="C27")
            axes[row, col].set(title=f"{title} · $t={times[-1]:g}$ · error {100 * error:.3g}%", xlabel="$x/L$", ylabel=r"$\eta(x,t)$", xlim=(0, 1))
            axes[row, col].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        for ax in axes[:, col]:
            ax.grid(alpha=0.7)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", ncols=2, frameon=False)
    finish(fig, "04-traveling-wave-accuracy",
           "Phase drift and final unaligned/aligned surface profiles for median-terminal-error Stokes and isolated Tanaka cases in the current test panel. Isolated Tanaka cases are identified from their single-wave parameter groups. Translation minimizes the periodic surface L2 discrepancy by Fourier interpolation. For Stokes the shift is unwrapped modulo one carrier wavelength, resolving the equivalent periodic minimizers. Alignment is diagnostic only; the raw errors are shown as well.",
           "Top: continuous Fourier alignment · middle: raw prediction · bottom: aligned prediction · current TEST panel")

    # First strong crest-amplification event, with a fixed coordinate recentering.
    truth = roll["collision_truth_eta"]
    pred = roll["collision_pred_eta"]
    times = roll["collision_times"]
    crest = truth.max(axis=-1)
    events, _ = find_peaks(crest, prominence=0.10 * crest[0])
    events = events[(events >= 4) & (events < len(times) - 4)]
    event = int(events[0])
    frames = [event - 4, event, event + 4]
    center = int(np.argmax(truth[event]))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25), sharex=True, sharey=True, layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.09, 1, 0.86))
    for ax, index, title in zip(axes, frames, ("Before", "Interaction", "After"), strict=True):
        actual = np.roll(truth[index], 512 - center)
        predicted = np.roll(pred[index], 512 - center)
        error = np.linalg.norm(actual - predicted) / np.linalg.norm(actual)
        ax.plot(x - 0.5, actual, color=CS, label="CS reference", lw=1.7)
        ax.plot(x - 0.5, predicted, color=C27, ls="--", label="C27")
        ax.set(title=f"{title} · $t={times[index]:.1f}$", xlabel=r"$(x-x_c)/L$", xlim=(-0.5, 0.5))
        ax.text(0.97, 0.93, rf"$e_\eta$ = {error:.2%}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.grid(alpha=0.6)
    axes[0].set_ylabel(r"$\eta(x,t)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=2, frameon=False)
    finish(fig, "05-tanaka-collision",
           f"Representative two-wave Tanaka interaction from the current test panel, simulation {int(roll['collision_case_id'])}, h={float(roll['collision_depth']):.6g}. The displayed event is an early local maximum of crest amplification, not a tracked-crest estimate of collision time. All panels use identical axes and the same fixed spatial recentering around the reference event crest; the learned profiles are not individually aligned. This supplies the draft's interaction illustration without claiming an unverified collision-time statistic.",
           f"Tanaka case {int(roll['collision_case_id'])} · fixed recentering of both methods · event selected from reference crest amplification")

    fig, axes = plt.subplots(3, 2, figsize=(8.8, 7.3), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.04, 1, 0.96))
    for col, name in enumerate(("bf", "sea")):
        depth, times = float(roll[f"{name}_depth"]), roll[f"{name}_times"]
        k = np.arange(513)
        multiplicity = np.full(len(k), 2.0)
        multiplicity[[0, -1]] = 1
        energies = [0.5 * multiplicity * (
            abs(np.fft.rfft(roll[f"{name}_{prefix}_eta"], axis=-1) / 1024) ** 2
            + k * np.tanh(depth * k) * abs(np.fft.rfft(roll[f"{name}_{prefix}_xi"], axis=-1) / 1024) ** 2
        ) for prefix in ("truth", "pred")]
        k0 = int(1 + np.argmax(energies[0][0, 1:]))
        limit = min(128, max(24, 4 * k0))
        scale = max(energies[0][:, 1:limit + 1].max(), energies[1][:, 1:limit + 1].max())
        for row, (energy, label) in enumerate(zip(energies, ("CS reference", "C27"), strict=True)):
            mesh = axes[row, col].pcolormesh(times, np.arange(1, limit + 1),
                np.log10(np.maximum(energy[:, 1:limit + 1].T / scale, 1e-8)),
                vmin=-7, vmax=0, cmap="magma", shading="auto", rasterized=True)
            axes[row, col].set(title=f"{FAMILIES[3 if col == 0 else 2]} · {label}", xlabel="$t$", ylabel="Fourier mode $n$")
            if row == 1:
                fig.colorbar(mesh, ax=axes[:2, col], shrink=0.8, label=r"$\log_{10}(\mathcal{E}_k / E_*)$")
        if name == "bf":
            initial = energies[0][0]
            # Identify the symmetric seeded pair, not higher harmonics.
            candidates = np.arange(1, k0)
            q = int(candidates[np.argmax(np.minimum(initial[k0 - candidates], initial[k0 + candidates]))])
            for energy, color, label, style in zip(energies, (CS, C27), ("CS reference", "C27"), ("-", "--"), strict=True):
                ratio = (energy[:, k0 - q] + energy[:, k0 + q]) / energy[:, k0]
                axes[2, col].plot(times, ratio, color=color, label=label, ls=style)
            axes[2, col].set(title=f"Seeded sidebands: $n_0={k0}$, $q={q}$", xlabel="$t$", ylabel=r"$R_{\mathrm{sb}}(t)$")
            axes[2, col].legend(frameon=False)
        else:
            modes = np.arange(energies[0].shape[-1])
            masks = ((modes > 0) & (modes < 0.5 * k0), (modes >= 0.5 * k0) & (modes <= 1.5 * k0), modes > 1.5 * k0)
            for band, mask in enumerate(masks):
                for energy, style in zip(energies, ("-", "--"), strict=True):
                    fraction = energy[:, mask].sum(axis=1) / energy[:, 1:].sum(axis=1)
                    axes[2, col].plot(times, fraction, color=COLORS[band], ls=style, label=("Low", "Peak", "High")[band] if style == "-" else None)
            axes[2, col].set(title=f"Bands relative to initial peak $n_p={k0}$", xlabel="$t$", ylabel="Energy fraction")
            axes[2, col].legend(frameon=False, ncols=3)
        axes[2, col].grid(alpha=0.6)
    finish(fig, "06-spectral-transfer",
           "Quadratic modal-energy evolution in representative Benjamin–Feir and JONSWAP/TMA test cases. Each column shares one color scale across the two methods, normalized by its joint maximum E*. One-sided Fourier energies include conjugate-mode multiplicities. The Benjamin–Feir carrier and strongest symmetric seeded sideband pair are identified from the initial energy spectrum. Sea bands are fixed at n<0.5np, 0.5np<=n<=1.5np and n>1.5np, excluding the zero mode. Solid/dashed band curves denote CS/C27.",
           "Matched color scales within each column · sea bands: solid CS / dashed C27 · g = 1")

    fig = plt.figure(figsize=(9.3, 4.7), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.06, 1, 0.94))
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[:, :2])
    median_band(ax, roll["ham_tau"], roll["ham_pred"], C27, "C27 states")
    median_band(ax, roll["ham_tau"], roll["ham_truth"], CS, "CS states")
    ax.set(title="(a) Physical Hamiltonian · pooled", xlabel="Normalized time $t/T$", ylabel=r"Signed drift $\delta_H(t)$", yscale="symlog", xlim=(0, 1))
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    for index, name in enumerate(("stokes", "tanaka", "sea", "bf")):
        ax = fig.add_subplot(grid[index // 2, 2 + index % 2])
        time_values = roll[f"{name}_times"][::10]
        ax.plot(time_values, roll[f"{name}_ham_pred"], color=C27)
        ax.plot(time_values, roll[f"{name}_ham_truth"], color=CS, ls=":")
        ax.set(title=f"({chr(98 + index)}) {FAMILIES[index]}", xlabel="$t$", ylabel=r"$\delta_H$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.grid(alpha=0.5)
    finish(fig, "07-hamiltonian-drift",
           "Independent physical Hamiltonian check: order-6 Craig–Sulem with padding factor 8 is freshly evaluated on both the reference and C27 predicted states at 26 saved times per case. It is not the archived learned-DNO energy diagnostic. The pooled panel shows signed median and interquartile range on a symmetric-log scale with linear threshold 1e-8; the four examples use linear signed axes. Saved states and Hamiltonian evaluation use float64. Energy drift is not assumed to equal the CS numerical floor; the underlying arrays are included in plot-data.npz.",
           "H evaluated with CS-6 at predicted states · 26 times × 128 cases · shaded bands = IQR")

    fig, ax = plt.subplots(figsize=(6.8, 4.15), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.07, 1, 0.93))
    resolutions = np.asarray(sorted({row["nx"] for row in benchmark["latency"]}))
    for i, method in enumerate(("C27", "CS-2", "CS-4", "CS-6", "CS-8")):
        records = [row for row in benchmark["latency"] if row["method"] == method]
        milliseconds = 1000 * np.array([row["seconds"] for row in records])
        slope = np.polyfit(np.log(resolutions), np.log(milliseconds), 1)[0]
        color = C27 if i == 0 else plt.colormaps["copper"](0.15 + 0.19 * i)
        ax.loglog(resolutions, milliseconds, "o-", color=color, lw=2 if i == 0 else 1.2, label=f"{method}  ($p={slope:.2f}$)")
    reference = resolutions * np.log2(resolutions)
    ax.loglog(resolutions, reference / reference[2] * 0.6, "--", color="#9ca6af", label=r"$N_x\log N_x$ (scaled)")
    ax.set(title="Single DNO evaluation · warmed CPU execution", xlabel="Spatial resolution $N_x$", ylabel="Median latency (ms)", xticks=resolutions)
    ax.set_xticklabels([str(int(n)) for n in resolutions])
    ax.grid(which="major")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    finish(fig, "08-dno-complexity-scaling",
           "Fresh CPU-only, batch-one DNO latency on an AMD EPYC 9124 host with an eight-logical-CPU affinity. Learned inference uses FP32; CS and physics use FP64. Seven synchronized measurements follow JIT warm-up at each resolution and order. One current-test Stokes initial condition is Fourier-resampled. CS padding is 8. Fitted log-log exponents over N=128..2048 are empirical finite-range summaries, not asymptotic proofs. C27 has no M input. These CPU measurements do not establish GPU performance.",
           "AMD EPYC 9124 CPU · 8 logical CPUs · FP32 net / FP64 physics · batch 1 · 7 repeats · JIT excluded")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, 0.08, 1, 0.92))
    cs_records = sorted([row for row in benchmark["rollouts"] if row["order"]], key=lambda row: row["order"])
    neural = next(row for row in benchmark["rollouts"] if row["method"] == "C27")
    axes[0].plot([row["order"] for row in cs_records], [row["seconds"] for row in cs_records], "o-", color=CS, label="Craig–Sulem")
    axes[0].axhline(neural["seconds"], color=C27, lw=2, label="C27 (independent of $M$)")
    axes[0].set(title="(a) Fixed-horizon runtime", xlabel="Craig–Sulem order $M$", ylabel="Trajectory time (s)", xticks=[2, 4, 6, 8, 10], ylim=(0, neural["seconds"] * 1.17))
    axes[0].legend(frameon=False, loc="upper left")
    for row in benchmark["rollouts"]:
        if row["method"] == "CS-10":
            continue
        color = C27 if row["method"] == "C27" else CS
        axes[1].loglog(row["seconds"], row["eta_trajectory_error"], "o", color=color, ms=7)
        axes[1].annotate(row["method"], (row["seconds"], row["eta_trajectory_error"]), xytext=(6, 5), textcoords="offset points", fontsize=8, color=color)
    axes[1].set(title="(b) Accuracy–runtime comparison", xlabel="Trajectory time (s)", ylabel=r"$E_{\eta,\mathrm{traj}}$ vs. CS-10")
    axes[1].margins(x=0.28, y=0.2)
    for ax in axes:
        ax.grid(alpha=0.6)
    finish(fig, "09-runtime-tradeoff",
           "Fresh short-horizon CPU benchmark on one current-test Stokes initial condition, Fourier-resampled to N=256: T=1, dt=0.01, GL2 with four fixed-point iterations, hard cutoff at mode 64, g=1 and padding factor 8. All methods use the same surrogate-callback integration path, including CS actions, and save the same 51 states. Learned inference uses FP32 with FP64 CS and integration. Timings are medians of three warmed synchronized runs. Errors compare surface trajectories with CS-10; the CS-10 zero self-error is omitted from the log error plot. This illustration is distinct from the long-horizon evaluation panel.",
           "CPU · same GL2 integration path · N = 256 · T = 1 · Δt = 0.01 · 3 repeats · compile time excluded")

    readme = (OUT / "README.md").read_text().split("\n## Figure captions")[0]
    (OUT / "README.md").write_text(readme.rstrip() + "\n\n## Figure captions\n\n" +
        "\n\n".join(f"### {name}\n\n{text}" for name, text in CAPTIONS.items()) + "\n")
    with PdfWriter() as writer:
        for name in NAMES:
            writer.append(FIGURES / f"{name}.pdf")
        writer.write(OUT / "preview.pdf")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for ax, name in zip(axes.flat, NAMES, strict=True):
        ax.imshow(plt.imread(FIGURES / f"{name}.png"))
        ax.set_axis_off()
        ax.set_title(name[3:].replace("-", " ").capitalize(), fontsize=13, pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.08, hspace=0.18)
    fig.savefig(OUT / "overview.png", dpi=150, facecolor="#f3f5f7")
    plt.close(fig)
    with zipfile.ZipFile(OUT / "c27-hard128-plot-pack.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix in (".pdf", ".png", ".md", ".npz", ".py", ".tex"):
                bundle.write(path, arcname=Path("c27-hard128-plot-pack") / path.relative_to(OUT))
    with zipfile.ZipFile(OUT / "c27-hard128-latex.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [OUT / "draft.tex", *(FIGURES / f"{name}.pdf" for name in NAMES)]:
            bundle.write(path, arcname=path.relative_to(OUT))
    print("Wrote preview.pdf, overview.png and both hard128 ZIP bundles", flush=True)
