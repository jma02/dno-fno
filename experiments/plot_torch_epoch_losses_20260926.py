"""Plot recorded epoch losses, separating projection and weight-EMA effects."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = ("torch_spectral_width256_full_epoch_20260926",
             "torch_spectral_width256_epoch2_20260926",
             "torch_spectral_width256_p128_epoch3_20260926",
             "torch_spectral_p128_epoch4_weightema_20260926")
    records = [json.loads((ROOT / "experiments" / f"{name}.json").read_text()) for name in names]
    epochs = [record["completed_epochs"] for record in records]
    train = [record["train_relative_l2"] for record in records]
    validation = [record["full_validation"]["relative_l2"] for record in records]
    projected = records[2]["initial_full_validation"]["relative_l2"]
    ema = records[3]["weight_ema_full_validation"]["relative_l2"]
    assert epochs == [1, 2, 3, 4]
    assert all(record["full_validation"]["examples"] == 1474440 for record in records)
    output = ROOT / "outputs/torch_epoch_loss_trajectory_20260926"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.3))
    fig.subplots_adjust(left=.12, right=.96, top=.85, bottom=.23)
    fig.suptitle("Training and validation over four epochs", x=.12, ha="left", fontsize=18)
    ax.set_title("Current spectral model · full validation: 1,474,440 examples", loc="left", fontsize=11, pad=14)
    ax.plot(epochs, train, "o-", color="#d47719", linewidth=2.3, label="Training: mean over the epoch")
    ax.plot(epochs[:2], validation[:2], "o-", color="#2374b5", linewidth=2.3,
            label="Validation: end-of-epoch weights")
    ax.plot([2, 3, 4], [projected, *validation[2:]], "-", color="#2374b5", linewidth=2.3)
    ax.scatter([3, 4], validation[2:], color="#2374b5", zorder=4)
    ax.plot([2, 2], [validation[1], projected], ":", color="#2374b5", linewidth=2)
    ax.scatter([2], [projected], marker="D", facecolors="white", edgecolors="#2374b5", zorder=5)
    ax.annotate(f"128-mode projection alone\n{projected:.3%}, before epoch 3",
                xy=(2, projected), xytext=(1.05, .000205), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#555555"}, color="#333333")
    ax.plot([4, 4], [validation[-1], ema], ":", color="#16834a", linewidth=2)
    ax.scatter([4], [ema], marker="*", s=175, color="#16834a", zorder=5,
                label="Validation: epoch-4 weight EMA")
    ax.annotate(f"Weight EMA: {ema:.5%}", xy=(4, ema), xytext=(2.85, .000030),
                color="#16834a", fontsize=11,
                arrowprops={"arrowstyle": "->", "color": "#16834a"})
    ax.set(xlabel="Completed training epochs", ylabel="Relative L2 error (lower is better)",
           xticks=epochs, xlim=(.9, 4.15), ylim=(0, .0006))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.text(.12, .105, "Dotted drops are changes without another epoch: output projection at epoch 2, weight averaging at epoch 4.", fontsize=9)
    fig.text(.12, .067, "Epochs 3–4 use the 128-mode projection. Weight EMA runs during the final 20% of epoch 4 (decay 0.999).", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"loss_over_epochs.{extension}", dpi=180)
    plt.close(fig)
    report = {"source_files": [f"experiments/{name}.json" for name in names],
              "epochs": epochs, "training_epoch_mean_relative_l2": train,
              "validation_endpoint_relative_l2": validation,
              "epoch2_projected_before_epoch3_relative_l2": projected,
              "epoch4_weight_ema_validation_relative_l2": ema,
              "validation_examples": 1474440,
              "scope": "Recorded four-epoch history; epoch3 adds projection, epoch4 adds shadow weight EMA; no extrapolated epochs"}
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "loss_over_epochs.png")


if __name__ == "__main__":
    main()
