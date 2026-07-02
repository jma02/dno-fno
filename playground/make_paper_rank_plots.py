"""Paper-quality figures corroborating the low-rank DNO-residual theorems.

Consumes the three summary.json files produced by:
  - playground/dno_residual_rank_probe.py     (restricted operator rank, M-sweep, eta-quantile sweep)
  - playground/dno_snapshot_rank_probe.py     (rollout snapshot rank — the quantity the proofs are about)
  - playground/dno_operator_rank_probe.py     (global operator rank, N-sweep)

Outputs into a single directory: figures + a one-page caption-style README block
in the form of a short JSON manifest.

Figures produced:
  fig1_snapshot_vs_operator.pdf   — the headline two-quantity gap
  fig2_operator_rank_vs_N.pdf     — Prop. 3 verification
  fig3_snapshot_spectra.pdf       — per-regime singular spectra over a rollout
  fig4_series_convergence.pdf     — Craig-Sulem convergence vs eta magnitude

All figures are emitted as both PDF (vector, for paper) and PNG (300 dpi, for inspection).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def configure_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


REGIME_DISPLAY = {
    "linear":            r"Linear",
    "stokes_finite":     r"Stokes (finite)",
    "stokes_deep":       r"Stokes (deep)",
    "tanaka_g0":         r"Tanaka $g_0$",
    "tanaka_g1":         r"Tanaka $g_1$",
    "bf_g0":             r"Benjamin--Feir $g_0$",
    "bf_g1":             r"Benjamin--Feir $g_1$",
    "bf_modal":          r"Benjamin--Feir modal",
    "random_sea_finite": r"Random sea (finite)",
    "random_sea_deep":   r"Random sea (deep)",
}

REGIME_COLOR = {
    "linear":            "#4c72b0",
    "stokes_finite":     "#55a868",
    "stokes_deep":       "#2e7d32",
    "tanaka_g0":         "#c44e52",
    "tanaka_g1":         "#8c2d2d",
    "bf_g0":             "#dd8452",
    "bf_g1":             "#a05a30",
    "bf_modal":          "#7f3f1d",
    "random_sea_finite": "#8172b3",
    "random_sea_deep":   "#5a4a82",
}


def load_summaries(snap_path: Path, op_path: Path, residual_path: Path):
    snap = json.loads(snap_path.read_text())
    op = json.loads(op_path.read_text())
    res = json.loads(residual_path.read_text())
    return snap, op, res


def rank_at_tol(sigma: np.ndarray, tol: float) -> int:
    if sigma.size == 0:
        return 0
    return int(np.sum(sigma > tol * max(float(sigma[0]), 1e-30)))


def fig1_snapshot_vs_operator(snap: dict, op: dict, out_dir: Path) -> None:
    tol_label = r"$\sigma_i > 10^{-4} \sigma_1$"
    snap_by_regime = {r["regime"]: r for r in snap["results"]}
    op_by_regime = {r["regime"]: r for r in op["results"]}

    regimes = [r for r in op_by_regime.keys() if r in snap_by_regime]
    # order by snapshot rank ascending for visual clarity
    def snap_rank(regime: str) -> int:
        return int(snap_by_regime[regime]["ranks"]["rel_tol_1e-04"])
    regimes.sort(key=snap_rank)

    snap_ranks = [snap_rank(r) for r in regimes]
    n_used = op["n_sweep"][-1]
    op_ranks = []
    for r in regimes:
        entries = op_by_regime[r]["by_N"]
        last = max(entries, key=lambda e: e["N"])
        op_ranks.append(int(last["ranks"]["rel_tol_1e-04"]))

    labels = [REGIME_DISPLAY.get(r, r) for r in regimes]
    y = np.arange(len(regimes))
    height = 0.36

    fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(regimes) + 1.4))
    bars_snap = ax.barh(y - height / 2, snap_ranks, height,
                        label="Snapshot rank along rollout (the quantity the proofs bound)",
                        color="#4c72b0", edgecolor="none")
    bars_op = ax.barh(y + height / 2, op_ranks, height,
                      label=fr"Global operator rank, $N={n_used}$",
                      color="#c44e52", edgecolor="none")

    for bar, val in zip(bars_snap, snap_ranks):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=7, color="#2a3f60")
    for bar, val in zip(bars_op, op_ranks):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=7, color="#7a2a30")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Numerical rank ({tol_label})")
    ax.set_xlim(0, max(op_ranks) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=1, framealpha=0.9, edgecolor="0.6")
    ax.set_title(r"Snapshot rank $\ll$ operator rank for the truncated DNO residual ($M=6$)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_snapshot_vs_operator.pdf")
    fig.savefig(out_dir / "fig1_snapshot_vs_operator.png")
    plt.close(fig)


def fig2_operator_rank_vs_N(op: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ns_all = sorted(op["n_sweep"])
    ax.plot(ns_all, ns_all, "k--", lw=0.9, alpha=0.6, label=r"$\mathrm{rank}=N$")
    for r in op["results"]:
        ns = [int(e["N"]) for e in r["by_N"]]
        rk = [int(e["ranks"]["rel_tol_1e-04"]) for e in r["by_N"]]
        regime = r["regime"]
        ax.plot(ns, rk, marker="o",
                color=REGIME_COLOR.get(regime, None),
                label=REGIME_DISPLAY.get(regime, regime))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"Grid size $N$")
    ax.set_ylabel(r"Operator rank ($\sigma_i > 10^{-4}\sigma_1$)")
    ax.set_title(r"$R_M(\eta)$ is not low rank in $M$ alone (Prop.~3)", loc="left")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.6")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_operator_rank_vs_N.pdf")
    fig.savefig(out_dir / "fig2_operator_rank_vs_N.png")
    plt.close(fig)


def fig3_snapshot_spectra(snap: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    results = sorted(snap["results"], key=lambda r: int(r["ranks"]["rel_tol_1e-04"]))
    for r in results:
        s = np.asarray(r["singular_values"], dtype=np.float64)
        rel = s / max(float(s[0]), 1e-30)
        regime = r["regime"]
        rk = int(r["ranks"]["rel_tol_1e-04"])
        ax.semilogy(np.arange(1, s.shape[0] + 1), np.maximum(rel, 1e-12),
                    color=REGIME_COLOR.get(regime, None), alpha=0.92,
                    label=f"{REGIME_DISPLAY.get(regime, regime)} (rk={rk})")
    ax.axhline(1e-4, color="0.4", lw=0.7, ls=":", label=r"$10^{-4}$ tolerance")
    ax.set_xlabel(r"Singular index $i$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(r"Snapshot rank of $R_M(\eta(t),\xi(t))$ over a rollout (truth, $M=6$)",
                 loc="left")
    ax.set_xlim(0, 251)
    ax.set_ylim(1e-7, 2.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, edgecolor="0.6", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_snapshot_spectra.pdf")
    fig.savefig(out_dir / "fig3_snapshot_spectra.png")
    plt.close(fig)


def fig4_series_convergence(residual: dict, out_dir: Path) -> None:
    quantiles = sorted({float(p["eta_quantile"]) for r in residual["results"] for p in r["eta_probes"]})
    regimes_to_show = ["tanaka", "bf", "stokes_deep"]
    available = [r for r in residual["results"] if r["dataset"] in regimes_to_show]
    if not available:
        available = residual["results"][:3]

    fig, axes = plt.subplots(len(available), len(quantiles),
                             figsize=(2.5 * len(quantiles) + 0.8, 2.2 * len(available) + 0.6),
                             squeeze=False, sharex=True, sharey="row")
    M_cmap = plt.get_cmap("viridis")

    for row, result in enumerate(available):
        probes_by_q = {float(p["eta_quantile"]): p for p in result["eta_probes"]}
        for col, q in enumerate(quantiles):
            ax = axes[row, col]
            if q not in probes_by_q:
                ax.axis("off")
                continue
            probe = probes_by_q[q]
            entries = probe["by_order"]
            n_M = len(entries)
            for i, entry in enumerate(entries):
                M = int(entry["order"])
                s = np.asarray(entry["singular_values"], dtype=np.float64)
                rel = s / max(float(s[0]), 1e-30)
                color = M_cmap(0.1 + 0.8 * i / max(n_M - 1, 1))
                ax.semilogy(np.arange(1, s.shape[0] + 1), np.maximum(rel, 1e-10),
                            color=color, lw=1.0, label=f"$M={M}$")
            if row == 0:
                ax.set_title(fr"$\eta$-quantile $q={q:g}$")
            if col == 0:
                regime_name = REGIME_DISPLAY.get(result["dataset"], result["dataset"])
                ax.set_ylabel(f"{regime_name}\n" + r"$\sigma_i/\sigma_1$")
            if row == len(available) - 1:
                ax.set_xlabel(r"Singular index $i$ (xi-PCA restricted)")
            if row == 0 and col == len(quantiles) - 1:
                ax.legend(loc="lower left", fontsize=7, framealpha=0.9, edgecolor="0.6")

    fig.suptitle(r"Craig--Sulem convergence: $R_M$ spectrum vs.\ truncation order $M$",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "fig4_series_convergence.pdf")
    fig.savefig(out_dir / "fig4_series_convergence.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap_summary",
                        default="playground/runs/dno_snapshot_rank/summary.json")
    parser.add_argument("--op_summary",
                        default="playground/runs/dno_operator_rank/summary.json")
    parser.add_argument("--residual_summary",
                        default="playground/runs/dno_residual_rank_probe/summary.json")
    parser.add_argument("--output_dir",
                        default="playground/runs/paper_rank_plots")
    args = parser.parse_args()

    snap_path = (REPO_ROOT / args.snap_summary).resolve()
    op_path = (REPO_ROOT / args.op_summary).resolve()
    res_path = (REPO_ROOT / args.residual_summary).resolve()

    for p in (snap_path, op_path, res_path):
        if not p.exists():
            raise FileNotFoundError(f"missing summary: {p}")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_paper_style()
    snap, op, residual = load_summaries(snap_path, op_path, res_path)

    fig1_snapshot_vs_operator(snap, op, output_dir)
    fig2_operator_rank_vs_N(op, output_dir)
    fig3_snapshot_spectra(snap, output_dir)
    fig4_series_convergence(residual, output_dir)

    manifest = {
        "fig1_snapshot_vs_operator": {
            "what": "Snapshot rank (proof quantity) vs global operator rank, same eta and M.",
            "supports": "Theorems' core distinction; visual headline.",
        },
        "fig2_operator_rank_vs_N": {
            "what": "Rank @ 1e-4 tolerance grows with N along the rank=N line for all regimes.",
            "supports": "Prop. 3 — fixed M does not bound operator rank.",
        },
        "fig3_snapshot_spectra": {
            "what": "Per-regime singular spectra of R_M(eta(t), xi(t)) along the truth rollout.",
            "supports": "Prop. 6 (finite-Stokes), Prop. 13/14 (BF finite sidebands), Prop. 15 (cascade).",
        },
        "fig4_series_convergence": {
            "what": "Singular spectra of restricted R_M as M varies, at three eta magnitudes.",
            "supports": "Craig--Sulem series convergence for small eta; growth for large eta.",
        },
        "source_summaries": {
            "snapshot_rank": str(snap_path),
            "operator_rank": str(op_path),
            "residual_rank": str(res_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for fname in ("fig1_snapshot_vs_operator", "fig2_operator_rank_vs_N",
                  "fig3_snapshot_spectra", "fig4_series_convergence"):
        print(f"wrote {output_dir / (fname + '.pdf')}")
        print(f"wrote {output_dir / (fname + '.png')}")
    print(f"wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
