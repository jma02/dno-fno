"""Isolate projecting the order-6 DNO before the nonlinear rollout product."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.solvers import time_integrator as ti  # noqa: E402

SOURCE = ROOT / "outputs/torch_rollout_screen64_20260926"
OUTPUT = ROOT / "outputs/rollout_projection_floor_20260926"
ORIGINAL_DNO = ti._dno_series_hat


def projected_dno(eta_hat: jax.Array, xi_hat: jax.Array, k: jax.Array,
                  g0: jax.Array, *, nx: int, order: int,
                  pad_factor: int = 8) -> jax.Array:
    spectrum = ORIGINAL_DNO(eta_hat, xi_hat, k, g0, nx=nx, order=order,
                            pad_factor=pad_factor)
    return spectrum.at[..., 129:].set(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("full", "projected", "compare"))
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.variant != "compare":
        jax.config.update("jax_enable_x64", True)
        with np.load(SOURCE / "initial_conditions.npz") as source:
            data = dict(source)
        params = ti.make_solver_params(data["eta"].shape[-1], float(data["length"]),
                                       jnp.asarray(data["depths"])[:, None],
                                       dno_order=6, pad_factor=8, filter_fraction=.25)
        # Only the DNO returned to the RHS changes. All recurrence terms,
        # integrator settings and saved-output filtering remain identical.
        predictor = projected_dno if args.variant == "projected" else ORIGINAL_DNO
        started = perf_counter()
        print(f"Starting {args.variant}: 64 cases, t=4, FP64 CPU", flush=True)
        with patch.object(ti, "_dno_series_hat", predictor):
            result = ti.rollout(ti.State(jnp.asarray(data["eta"], dtype=jnp.float64),
                                        jnp.asarray(data["xi"], dtype=jnp.float64)),
                                jnp.asarray(data["times"]), params, save_gxi=True,
                                substeps_per_interval=8, method="gl2_if",
                                implicit_iterations=4, zero_mean_xi=True)
            arrays = {key: np.asarray(result[key]) for key in ("eta", "xi", "gxi")}
        seconds = perf_counter() - started
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(OUTPUT / f"{args.variant}.npz", **arrays,
                            times=data["times"], simulation_ids=data["simulation_ids"])
        (OUTPUT / f"{args.variant}_timing.json").write_text(json.dumps({
            "seconds": seconds, "backend": jax.default_backend(), "all_finite": True,
        }, indent=2) + "\n")
        print(f"Completed {args.variant} in {seconds:.3f}s", flush=True)
        return

    with np.load(OUTPUT / "full.npz") as source:
        full = dict(source)
    with np.load(OUTPUT / "projected.npz") as source:
        projected = dict(source)
    with np.load(SOURCE / "reference.npz") as source:
        reference = dict(source)
    with np.load(ROOT / "outputs/torch_p128_epoch3_worst64_rollouts_20260926/predictions.npz") as source:
        model = dict(source)
    for other in (projected, reference, model):
        for key in ("times", "simulation_ids"):
            np.testing.assert_array_equal(full[key], other[key])
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    rows = []
    parity = {}
    for field in ("eta", "xi", "gxi"):
        parity[field] = float(np.max(np.abs(full[field] - reference[f"truth_{field}"])))
        np.testing.assert_allclose(full[field], reference[f"truth_{field}"], rtol=1e-10, atol=1e-13)
    for i, simulation_id in enumerate(full["simulation_ids"]):
        row = {"simulation_id": int(simulation_id), "family": manifest["families"][i]}
        for field in ("eta", "xi", "gxi"):
            truth = full[field][-1, i]
            denominator = np.linalg.norm(truth)
            floor = projected[field][-1, i] - truth
            model_error = model[f"pred_{field}"][-1, i] - truth
            row[field] = {
                "projection_relative_l2": float(np.linalg.norm(floor) / denominator),
                "model_relative_l2": float(np.linalg.norm(model_error) / denominator),
                "model_vs_projected_relative_l2": float(np.linalg.norm(model_error - floor)
                                                       / np.linalg.norm(projected[field][-1, i])),
                "projection_to_model_error_norm_ratio": float(np.linalg.norm(floor) / np.linalg.norm(model_error)),
            }
        rows.append(row)
    report = {"scope": "64 held-out rollouts; order6/pad8 FP64, t=4, GL2/8 substeps/4 iterations; only RHS DNO output projection changes",
              "full_reference_max_absolute_difference": parity,
              "timing": {variant: json.loads((OUTPUT / f"{variant}_timing.json").read_text())
                         for variant in ("full", "projected")},
              "summary": {field: {key: {"mean": float(np.mean([row[field][key] for row in rows])),
                                        "max": float(np.max([row[field][key] for row in rows]))}
                                  for key in rows[0][field]} for field in ("eta", "xi", "gxi")},
              "family_mean_eta": {family: {key: float(np.mean([row["eta"][key] for row in rows
                                                               if row["family"] == family]))
                                           for key in rows[0]["eta"]}
                                  for family in sorted(set(manifest["families"]))},
              "worst_six_by_model_eta": sorted(rows, key=lambda row: row["eta"]["model_relative_l2"], reverse=True)[:6],
              "cases": rows}
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    denominator = np.linalg.norm(full["eta"], axis=-1)
    for label, prediction in (("Projection alone (exact operator)", projected["eta"]),
                              ("Current learned model", model["pred_eta"])):
        curve = (np.linalg.norm(prediction - full["eta"], axis=-1) / denominator).mean(axis=-1)
        axes[0].semilogy(full["times"][1:], curve[1:], label=label)
    axes[0].set(xlabel="Simulated time", ylabel="Mean surface relative L2",
                title="Same 64 cases; original reference")
    axes[0].legend()
    worst = report["worst_six_by_model_eta"]
    positions = np.arange(len(worst))
    for offset, key, label in ((-.18, "projection_relative_l2", "Projection alone"),
                               (.18, "model_relative_l2", "Current model")):
        axes[1].bar(positions + offset, [row["eta"][key] for row in worst],
                    width=.36, label=label)
    axes[1].set(yscale="log", ylabel="Final surface relative L2", title="Six worst model rollouts",
                xticks=positions, xticklabels=[str(row["simulation_id"]) for row in worst])
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].legend()
    fig.savefig(OUTPUT / "projection_error.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
