"""Evaluate a checkpoint on the training run's test split (10% holdout of the
combined training dataset, never seen during training) and save per-source
representative plots + metrics. Mirrors eval_on_dno_dataset.py.

Usage:
    uv run python train-jax-10m/eval_on_train_holdout.py \\
        --run_dir /home/johnma/dno-fno/outputs/<run_name> \\
        --dataset combined_dataset.npz --batch_size 64 --allow_cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from dno_net_v2 import CraigSulemDNO
from losses import count_params
from util import (
    NormStats,
    build_split_indices,
    compute_log_depth,
    denormalize_targets,
    load_dataset_arrays,
    normalize_features,
    require_jax_devices,
)
from jax_training_util import plot_labeled_samples, summarize_errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the training run's 10% holdout test split.",
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--dataset", default=None,
                   help="Override dataset path. Defaults to the dataset recorded in config.json.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override split seed. Defaults to seed recorded in config.json (or 0).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--keep_xi_mean", action="store_true",
                   help="Skip the eval-time xi zero-mean projection. Note that load_dataset_arrays already projects.")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Optional cap on test examples (for quick CPU evals).")
    return p.parse_args()


def _load_checkpoint(run_dir: Path, checkpoint_name: str):
    ckpt_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    metadata_path = ckpt_dir / "metadata.json"
    if not ckpt_dir.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or metadata at {ckpt_dir}")
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=None, prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return ckpt_dir, params, metadata


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)
    run_dir = Path(args.run_dir).resolve()

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_dir, params, metadata = _load_checkpoint(run_dir, args.checkpoint)
    stats = metadata["stats"]
    norm_mode = config.get("norm", "minmax")
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))

    dataset_name = args.dataset or config["dataset"]
    dataset_path = (REPO_ROOT / "data" / dataset_name).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    dataset = load_dataset_arrays(dataset_path)

    # Recover test split using the same seed (and 80/10/10 split) the trainer used.
    n_total = int(dataset["eta"].shape[0])
    _, _, test_indices = build_split_indices(n_total, seed)
    if args.max_examples is not None and args.max_examples < test_indices.shape[0]:
        test_indices = np.sort(np.random.default_rng(seed).choice(
            test_indices, size=int(args.max_examples), replace=False,
        ))
    eta = dataset["eta"][test_indices]
    xi = dataset["xi"][test_indices]
    gxi = dataset["gxi"][test_indices]
    depth = dataset["depth"][test_indices]
    sources = dataset["source"][test_indices]
    x = dataset["x"]
    legend = {int(k): v for k, v in json.loads(
        dataset_path.with_suffix(".meta.json").read_text(encoding="utf-8")
    ).get("source_legend", {}).items()}
    source_labels = np.asarray([legend.get(int(s), str(int(s))) for s in sources], dtype=object)

    if not args.keep_xi_mean:
        # load_dataset_arrays already projects; this is a defensive no-op for clarity.
        xi = xi - np.mean(xi, axis=1, keepdims=True)

    ns = NormStats.from_dict(stats, mode=norm_mode)

    if config.get("model", "fno") == "cs_dno":
        feature_absmax = np.asarray(ns.feature_absmax).reshape(-1)
        model = CraigSulemDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            n_polys=int(config.get("cs_n_polys", 3)),
            use_first_deriv=bool(config.get("cs_use_first_deriv", True)),
            use_second_deriv=bool(config.get("cs_use_second_deriv", True)),
            use_half_deriv=bool(config.get("cs_use_half_deriv", True)),
            use_hilbert=bool(config.get("cs_use_hilbert", True)),
            mult_hidden=int(config.get("cs_mult_hidden", 32)),
            use_g1_baseline=bool(config.get("cs_use_g1_baseline", False)),
            g1_k_cut=int(config.get("cs_g1_k_cut", 128)),
            tie_xi_out_mult=bool(config.get("cs_tie_xi_out_mult", False)),
            phi_bias_free=bool(config.get("cs_phi_bias_free", False)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", feature_absmax[1])),
            eta_scale=float(config.get("eta_scale", feature_absmax[0])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )
    elif config.get("model", "fno") == "spectral_dno":
        model = SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(ns.feature_absmax).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )
    else:
        model = FNO1d(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            domain_length=float(stats.get("domain_length", 2.0 * np.pi)),
            xi_scale=float(np.asarray(ns.feature_absmax).reshape(-1)[1]),
            target_scale=float(ns.target_absmax),
        )

    @jax.jit
    def predict_batch(p, batch_inputs, batch_depth):
        return model.apply({"params": p}, batch_inputs, batch_depth)

    chunks: list[np.ndarray] = []
    for start in range(0, eta.shape[0], args.batch_size):
        end = start + args.batch_size
        batch_inputs = normalize_features(eta[start:end], xi[start:end], ns)
        batch_depth = compute_log_depth(depth[start:end])
        batch_inputs_d = jax.device_put(batch_inputs, devices[0])
        batch_depth_d = jax.device_put(batch_depth, devices[0])
        pred_norm = np.asarray(jax.device_get(predict_batch(params, batch_inputs_d, batch_depth_d)))
        chunks.append(denormalize_targets(pred_norm, ns).astype(np.float32))
    predictions = np.concatenate(chunks, axis=0)

    flat_pred = predictions.reshape((predictions.shape[0], -1))
    flat_target = gxi[..., None].reshape((gxi.shape[0], -1))
    rel_l2 = np.linalg.norm(flat_pred - flat_target, axis=1) / (np.linalg.norm(flat_target, axis=1) + 1e-12)
    rel_l1 = np.sum(np.abs(flat_pred - flat_target), axis=1) / (np.sum(np.abs(flat_target), axis=1) + 1e-12)
    pred_std = np.std(flat_pred, axis=1)
    tgt_std = np.std(flat_target, axis=1)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else run_dir / ("eval_on_train_holdout_keep_xi_mean" if args.keep_xi_mean else "eval_on_train_holdout")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(ckpt_dir),
        "checkpoint_epoch": int(metadata["epoch"]),
        "trained_on_dataset": config["dataset"],
        "model": config.get("model", "fno"),
        "norm": norm_mode,
        "evaluated_on_dataset": dataset_path.name,
        "split": "test",
        "split_seed": seed,
        "zero_mean_xi": not args.keep_xi_mean,
        "num_test_examples": int(eta.shape[0]),
        "param_count": count_params(params),
        "backend": backend,
        "overall": summarize_errors(rel_l2, rel_l1),
        "prediction_std": {
            "mean": float(np.mean(pred_std)),
            "median": float(np.median(pred_std)),
            "min": float(np.min(pred_std)),
            "max": float(np.max(pred_std)),
        },
        "per_source": {},
    }

    rng = np.random.default_rng(seed)
    selected_sources = sorted({legend.get(int(s), str(int(s))) for s in sources})
    for src in selected_sources:
        mask = source_labels == src
        if not mask.any():
            continue
        src_l2 = rel_l2[mask]
        src_l1 = rel_l1[mask]
        order = np.argsort(src_l2)
        chosen = np.asarray(
            [int(order[0]), int(order[len(order) // 2]), int(order[-1]), int(rng.integers(0, len(order)))],
            dtype=np.int32,
        )
        labels = ("best", "median", "worst", "random")
        plot_labeled_samples(
            output_dir / f"representative_samples_{src}.png",
            f"{src.title()}: {dataset_path.stem} train-holdout representatives",
            x,
            eta[mask][chosen], xi[mask][chosen],
            gxi[mask][chosen, :, None], predictions[mask][chosen],
            src_l2[chosen], src_l1[chosen],
            labels,
            depth_values=depth[mask][chosen],
        )
        summary["per_source"][src] = {
            "num_examples": int(mask.sum()),
            "rel_l2": {
                "mean": float(src_l2.mean()), "median": float(np.median(src_l2)),
                "p95": float(np.quantile(src_l2, 0.95)), "max": float(src_l2.max()),
            },
            "rel_l1": {
                "mean": float(src_l1.mean()), "median": float(np.median(src_l1)),
            },
            "prediction_std_mean": float(pred_std[mask].mean()),
            "target_std_mean": float(tgt_std[mask].mean()),
            "chosen_rel_l2": src_l2[chosen].tolist(),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"overall": summary["overall"], "per_source": {k: v["rel_l2"] for k, v in summary["per_source"].items()}}, indent=2))
    print(f"saved -> {output_dir}")


if __name__ == "__main__":
    main()
