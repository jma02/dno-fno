"""Render completed-epoch losses without touching the running trainer."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-fno-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--reference", type=Path)
parser.add_argument("--leon-result", type=Path)
args = parser.parse_args()
rows = list(map(json.loads, (args.run / "train_log.jsonl").read_text().splitlines()))
best = min(rows, key=lambda row: row["val_loss"])
reference = None
if args.reference:
    reference = min(map(json.loads, (args.reference / "train_log.jsonl").read_text().splitlines()),
                    key=lambda row: row["val_loss"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bfc5cc", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.70, wspace=0.28)
epochs = [row["epoch"] for row in rows]
for ax, key, scale, title in zip(axes, ("val_l2_loss", "val_loss"), (100, 1),
                                ("Relative L² loss (%)", "Validation objective"), strict=True):
    if key == "val_l2_loss":
        ax.plot(epochs, [100 * row["train_l2_loss"] for row in rows],
                color="#087eaa", lw=1.8, label="Training")
    ax.plot(epochs, [scale * row[key] for row in rows],
            color="#c45b22", lw=1.8, label="Validation")
    if reference:
        ax.axhline(scale * reference[key], color="#777f89", ls="--", lw=1.2,
                   label="Old full C27 · best checkpoint")
    if key == "val_l2_loss" and args.leon_result:
        leon_loss = json.loads(args.leon_result.read_text())["weight_ema_full_validation"]["relative_l2"]
        ax.axhline(scale * leon_loss, color="#8b5ea8", ls=":", lw=1.8,
                   label="Leon · epoch 4 weight EMA")
    ax.scatter(best["epoch"], scale * best[key], color="#c45b22", s=32, zorder=3)
    ax.set(xlabel="Epoch", title=title, xlim=(1, epochs[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(axis="y", color="#e9edf1", lw=0.7)
fig.suptitle("Gradient-EMA training", x=0.08, y=0.98, ha="left", fontsize=19)
fig.text(0.98, 0.955, f"{epochs[-1]} completed epochs", ha="right", color="#777f89")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
           bbox_to_anchor=(0.07, 0.91), ncol=2, frameon=False, fontsize=10)
for suffix in ("png", "pdf"):
    fig.savefig(args.run / f"loss-curve.{suffix}", dpi=180)
plt.close(fig)
print(f"Best validation: {best['val_loss']:.6g} at epoch {best['epoch']}")
