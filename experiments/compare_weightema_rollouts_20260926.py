"""Paired H100 rollout comparison: epoch3, epoch4 ordinary, epoch4 weight EMA."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/torch_p128_epoch4_ema_worst64_rollouts_20260926"


def main() -> None:
    with np.load(ROOT / "outputs/torch_rollout_screen64_20260926/reference.npz") as source:
        reference = dict(source)
    manifest = json.loads((ROOT / "outputs/torch_rollout_screen64_20260926/manifest.json").read_text())
    folders = {"epoch3": "torch_p128_epoch3_h100_rollout_timing_20260926",
               "epoch4_raw": "torch_p128_epoch4_raw_rollout64_20260926",
               "epoch4_ema": OUTPUT.name}
    report = {"scope": "Same64 validation cases/16 per family, t=4, original order6/pad8 reference; all model rollouts on H100",
              "models": {}}
    errors = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, folder in folders.items():
        with np.load(ROOT / "outputs" / folder / "predictions.npz") as source:
            prediction = dict(source)
        for key in ("times", "depths", "simulation_ids"):
            np.testing.assert_array_equal(prediction[key], reference[key])
        metadata = json.loads(str(prediction["rollout_metadata_json"]))
        assert metadata["model_max_mode"] == 128 and metadata["nonfinite_trajectories"] == 0
        assert "H100" in metadata["gpu"]
        errors[label] = {}
        for field in ("eta", "xi", "gxi"):
            pred, truth = prediction[f"pred_{field}"], reference[f"truth_{field}"]
            assert np.isfinite(pred).all()
            errors[label][field] = np.linalg.norm(pred - truth, axis=-1) / (np.linalg.norm(truth, axis=-1) + 1e-12)
        final = errors[label]["eta"][-1]
        report["models"][label] = {"metadata": metadata,
            "final_surface_relative_l2": {"mean": float(final.mean()), "median": float(np.median(final)),
                                           "p90": float(np.quantile(final, .9)), "max": float(final.max())},
            "family_mean_surface_l2": {family: float(final[np.array(manifest["families"]) == family].mean())
                                       for family in sorted(set(manifest["families"]))}}
        axes[0].semilogy(reference["times"][1:], errors[label]["eta"][1:].mean(axis=1), label=label)
    report["paired_ema_comparisons"] = {}
    for label in ("epoch3", "epoch4_raw"):
        old, new = errors[label]["eta"][-1], errors["epoch4_ema"]["eta"][-1]
        report["paired_ema_comparisons"][label] = {
            "cases_improved": int((new < old).sum()),
            "mean_reduction_percent": float(100 * (1 - new.mean() / old.mean())),
            "worst_reduction_percent": float(100 * (1 - new.max() / old.max()))}
    worst = np.argsort(errors["epoch4_ema"]["eta"][-1])[-6:][::-1]
    report["ema_worst_six"] = [{"simulation_id": int(reference["simulation_ids"][i]),
                                "family": manifest["families"][i],
                                **{label: float(errors[label]["eta"][-1, i]) for label in folders}}
                               for i in worst]
    x = np.arange(6)
    for offset, label in zip((-.25, 0, .25), folders, strict=True):
        axes[1].bar(x + offset, errors[label]["eta"][-1, worst], width=.25, label=label)
    axes[0].set(xlabel="Simulated time", ylabel="Mean surface relative L2", title="All64 cases")
    axes[1].set(yscale="log", ylabel="Final surface relative L2", title="Six worst EMA rollouts",
                xticks=x, xticklabels=[str(reference["simulation_ids"][i]) for i in worst])
    axes[1].tick_params(axis="x", labelrotation=30)
    for ax in axes:
        ax.legend()
    fig.savefig(OUTPUT / "checkpoint_comparison.png", dpi=160)
    plt.close(fig)
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
