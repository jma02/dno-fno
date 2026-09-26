"""Screen held-out snapshots, compute references, and render worst model rollouts."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/torch_epoch1_rollouts_20260926"
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reference", "render"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference-output", type=Path, default=OUTPUT)
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--worst", type=int, default=0)
    args = parser.parse_args()
    output = args.output
    sys.path.insert(0, str(ROOT))
    if args.stage == "prepare":
        output.mkdir(parents=True, exist_ok=False)
        source = ROOT.parent / "local-data/paper_equal_subset_20260924"
        arrays = {k: np.load(source / f"{k}.npy") for k in
                  ("eta", "xi", "depth", "family_id", "dataset_split", "simulation_id", "source_indices", "x")}
        rows = []
        for family in range(1, 5):
            candidates = np.flatnonzero((arrays["dataset_split"] == "validation") & (arrays["family_id"] == family))
            _, positions = np.unique(arrays["simulation_id"][candidates], return_index=True)
            if not 0 < args.per_family <= len(positions):
                raise ValueError("Insufficient distinct validation simulations")
            rows.extend(candidates[np.sort(positions)[:args.per_family]].tolist())
        length = float((arrays["x"][1] - arrays["x"][0]) * len(arrays["x"]))
        np.savez(output / "initial_conditions.npz", eta=arrays["eta"][rows], xi=arrays["xi"][rows],
                 depths=arrays["depth"][rows], simulation_ids=arrays["simulation_id"][rows],
                 times=np.linspace(0., 4., 51), length=length)
        manifest = {"source": str(source), "selection": f"First {args.per_family} distinct validation simulations per family in the local random subset; ranked after rollout",
                    "initial_state": "Held-out snapshots, not necessarily frame zero; reset autonomous dynamics clock to zero",
                    "families": [FAMILIES[int(arrays["family_id"][i]) - 1] for i in rows], "local_rows": rows,
                    "source_rows": arrays["source_indices"][rows].tolist(),
                    "simulation_ids": arrays["simulation_id"][rows].tolist(),
                    "depths": arrays["depth"][rows].tolist(), "length": length,
                    "tmax": 4., "saved_dt": .08, "substeps": 8, "internal_dt": .01,
                    "implicit_iterations": 4, "filter_fraction": .25,
                    "checkpoint": "torch_spectral_width256_full_epoch_20260926.pt"}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    elif args.stage == "reference":
        import jax
        import jax.numpy as jnp
        from solver.solvers import time_integrator as ti

        jax.config.update("jax_enable_x64", True)
        with np.load(output / "initial_conditions.npz") as source:
            data = dict(source)
        params = ti.make_solver_params(data["eta"].shape[-1], float(data["length"]),
                                       jnp.asarray(data["depths"])[:, None], dno_order=6,
                                       pad_factor=8, filter_fraction=.25)
        started = perf_counter()
        print(f"Computing float64 order-6 reference, {len(data['eta'])} trajectories", flush=True)
        result = ti.rollout(ti.State(jnp.asarray(data["eta"], dtype=jnp.float64),
                                    jnp.asarray(data["xi"], dtype=jnp.float64)),
                            jnp.asarray(data["times"]), params, save_gxi=True,
                            substeps_per_interval=8, method="gl2_if", implicit_iterations=4,
                            zero_mean_xi=True)
        truth = {k: np.asarray(result[k]) for k in ("eta", "xi", "gxi")}
        seconds = perf_counter() - started
        np.savez_compressed(output / "reference.npz", **{f"truth_{k}": v for k, v in truth.items()},
                            times=data["times"], depths=data["depths"], simulation_ids=data["simulation_ids"])
        (output / "reference_timing.json").write_text(json.dumps({"seconds": seconds, "backend": jax.default_backend(),
                                                                 "dno_order": 6, "pad_factor": 8}, indent=2) + "\n")
        print(f"Reference complete in {seconds:.3f}s", flush=True)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from solver.evals.eval_suite import compute_metrics
        from solver.evals.render_rollout_movie import render_rollout_gif

        manifest = json.loads((args.reference_output / "manifest.json").read_text())
        with np.load(output / "predictions.npz") as source:
            predicted = dict(source)
        model_metadata = json.loads(str(predicted["rollout_metadata_json"]))
        epoch = model_metadata["epoch"]
        manifest["checkpoint"] = model_metadata["checkpoint"]
        with np.load(args.reference_output / "reference.npz") as source:
            reference = dict(source)
        times = predicted["times"]
        np.testing.assert_array_equal(times, reference["times"])
        np.testing.assert_array_equal(predicted["simulation_ids"], reference["simulation_ids"])
        truth = {k: reference[f"truth_{k}"] for k in ("eta", "xi", "gxi")}
        pred = {k: predicted[f"pred_{k}"] for k in ("eta", "xi", "gxi")}
        metrics = compute_metrics(truth, pred, times, manifest["length"])
        arrays = metrics.pop("_arrays")
        np.savez_compressed(output / "comparison_trajs.npz", **predicted, **{f"truth_{k}": v for k, v in truth.items()},
                            **{k: arrays[k] for k in ("rel_l2_eta", "rel_l2_xi", "rel_l2_gxi", "truth_valid")})
        x = np.arange(pred["eta"].shape[-1]) * manifest["length"] / pred["eta"].shape[-1]
        rows = []
        for i, family in enumerate(manifest["families"]):
            finite = all(np.isfinite(pred[k][:, i]).all() for k in pred)
            row = {"family": family, "simulation_id": manifest["simulation_ids"][i], "depth": manifest["depths"][i],
                   "model_finite": finite, "reference_valid": bool(arrays["truth_valid"][i]),
                   "final_relative_l2": {k: float(arrays[f"rel_l2_{k}"][-1, i]) for k in pred},
                   "max_reference_energy_drift": float(np.nanmax(np.abs(arrays["energy_drift_truth"][:, i])))}
            rows.append(row)
        valid_indices = np.flatnonzero(arrays["truth_valid"]).tolist()
        ranked = sorted(valid_indices, key=lambda i: rows[i]["final_relative_l2"]["eta"]
                        if rows[i]["model_finite"] else float("inf"), reverse=True)
        selected = ranked[:args.worst] if args.worst else list(range(len(rows)))
        if not selected:
            raise ValueError("No valid references to rank")
        fig, axes = plt.subplots(len(selected), 3, figsize=(16, 2.8 * len(selected)),
                                 squeeze=False, constrained_layout=True)
        for rank, i in enumerate(selected):
            row = rows[i]
            family = row["family"]
            name = f"worst_{rank + 1:02d}_{family}_sim{row['simulation_id']}" if args.worst else family
            row["plot"] = name + ".gif"
            payload = {"x": x, "t": times, **{f"pred_{k}": v[:, i] for k, v in pred.items()},
                       **{f"truth_{k}": v[:, i] for k, v in truth.items()}}
            np.savez_compressed(output / f"{name}_comparison.npz", **payload)
            render_rollout_gif(payload, output / f"{name}.gif",
                               title=f"{family} | epoch {epoch} | simulation {row['simulation_id']} | h={row['depth']:.4g}",
                               fps=12, n_frames=51, dpi=75)
            ax, residual, error = axes[rank]
            ax.plot(x, truth["eta"][-1, i], label="Order-6 reference", color="black", linewidth=1.3)
            ax.plot(x, pred["eta"][-1, i], label=f"Epoch-{epoch} model", linestyle="--", linewidth=1.)
            ax.set_title(f"#{rank + 1} {family}, sim {row['simulation_id']}: t=4, L2={row['final_relative_l2']['eta']:.3%}")
            ax.set_ylabel("eta")
            ax.grid(alpha=.2)
            scale = np.sqrt(np.mean(truth["eta"][-1, i] ** 2)) + 1e-12
            residual.plot(x, 100 * (pred["eta"][-1, i] - truth["eta"][-1, i]) / scale, color="tab:red")
            residual.axhline(0, color="black", linewidth=.6)
            residual.set_title("Surface difference at t=4")
            residual.set_ylabel("Error / reference RMS (%)")
            residual.grid(alpha=.2)
            for k in ("eta", "xi", "gxi"):
                error.semilogy(times[1:], np.maximum(arrays[f"rel_l2_{k}"][1:, i], 1e-12), label=k)
            error.set_title("Relative L2 (t>0)")
            error.set_xlabel("Simulated time")
            error.grid(alpha=.2)
            if rank == 0:
                ax.legend(fontsize=8)
                error.legend(fontsize=8)
            print(json.dumps(row), flush=True)
        fig.savefig(output / "overview.png", dpi=140)
        plt.close(fig)
        report = {"manifest": manifest, "model_rollout": model_metadata,
                  "reference": json.loads((args.reference_output / "reference_timing.json").read_text()),
                  "trajectories": rows, "evaluation": metrics,
                  "ranked_indices": ranked, "plotted_indices": selected,
                  "ranking_metric": "Final surface relative L2 at t=4; nonfinite models first; invalid references excluded",
                  "scope": f"Worst within {len(rows)} validation snapshots, short horizon; not the full production test suite"}
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (ROOT / "experiments" / f"{output.name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
