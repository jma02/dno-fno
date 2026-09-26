"""Plot measured fixed-set error and held-out error from the overfit diagnostic."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    name = "torch_fixed_set_overfit_20260926"
    report = json.loads((ROOT / "experiments" / f"{name}.json").read_text())
    output = ROOT / "outputs" / name
    output.mkdir(parents=True, exist_ok=True)
    trajectory = [{"step": 0, **report["initial"]}, *report["progress"]]
    steps = [row["step"] for row in trajectory]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for split, label in (("train", "Fixed training set (256)"), ("validation", "Held-out set (256)")):
        axes[0].semilogy(steps, [row[split]["mean"] for row in trajectory], marker=".", label=label)
    axes[0].axhline(report["target"], color="black", linestyle="--", label="10× target")
    for family, label in enumerate(("Stokes", "Tanaka", "Benjamin–Feir", "JONSWAP–TMA"), start=1):
        axes[1].semilogy(steps, [row["train"]["per_family"][str(family)] for row in trajectory],
                         marker=".", label=label)
    for ax, title in zip(axes, ("Mean per-example operator error", "Fixed-training error by wave family"), strict=True):
        ax.set(title=title, xlabel="Additional updates on the same batch", ylabel="Relative L2")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Epoch-2 checkpoint → fixed-set overfit; LR 1e-5, gradient EMA 0.8")
    fig.savefig(output / "trajectory.png", dpi=160)
    plt.close(fig)
    initial = np.array(report["initial"]["train"]["per_example"])
    final = np.array(report["progress"][-1]["train"]["per_example"])
    print(json.dumps({"initial_train": initial.mean(), "final_train": final.mean(),
                      "improved_training_cases": int((final < initial).sum()),
                      "tenfold_improved_training_cases": int((final <= initial / 10).sum()),
                      "best_train": report["best_train_relative_l2"], "best_step": report["best_step"],
                      "final_validation": report["progress"][-1]["validation"]["mean"]}, indent=2))


if __name__ == "__main__":
    main()
