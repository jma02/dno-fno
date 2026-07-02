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
MODEL_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, MODEL_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from losses import count_params
from util import (
    NormStats,
    denormalize_targets,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    build_split_indices,
    normalize_features,
    require_jax_devices,
)
from jax_training_util import plot_representative_samples, summarize_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a JAX FNO checkpoint on the held-out Tanaka test split."
    )
    parser.add_argument("--dataset", default="tanaka_1_clean.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--carbs_root", default=None)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def choose_carbs_root(raw_carbs_root: str | None) -> Path:
    if raw_carbs_root is not None:
        return (REPO_ROOT / raw_carbs_root).resolve()

    outputs_dir = REPO_ROOT / "outputs"
    candidates = sorted(
        (
            path
            for path in outputs_dir.glob("carbs_jax_10m*")
            if path.is_dir() and ((path / "best_result.json").exists() or (path / "carbs_history.json").exists())
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find a 10M JAX CARBS output directory under outputs/")
    return candidates[0]


def resolve_run_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir is not None:
        run_dir = (REPO_ROOT / args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, None

    carbs_root = choose_carbs_root(args.carbs_root)
    best_result_path = carbs_root / "best_result.json"
    if best_result_path.exists():
        with open(best_result_path, "r", encoding="utf-8") as handle:
            best_result = json.load(handle)
        return Path(best_result["result"]["run_dir"]).resolve(), carbs_root

    history_path = carbs_root / "carbs_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Could not find best_result.json or carbs_history.json in {carbs_root}")

    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    successes = [record for record in history if record["result"].get("success")]
    if not successes:
        raise ValueError(f"No successful CARBS trials found in {history_path}")
    best_record = min(successes, key=lambda record: float(record["result"]["best_val_loss"]))
    return Path(best_record["result"]["run_dir"]).resolve(), carbs_root


def load_checkpoint(run_dir: Path, checkpoint_name: str):
    checkpoint_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    metadata_path = checkpoint_dir / "metadata.json"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return checkpoint_dir, params, payload


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)

    run_dir, carbs_root = resolve_run_dir(args)
    with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint_path, params, checkpoint_payload = load_checkpoint(run_dir, args.checkpoint)

    dataset_path = REPO_ROOT / "data" / args.dataset
    dataset = load_dataset_arrays(dataset_path)
    stats = checkpoint_payload.get("stats") or load_or_compute_stats(dataset_path, dataset=dataset)
    if config.get("model", "fno") == "fno" and bool(
        config.get("linear_baseline", config.get("linear_hotpath", config.get("predict_residual", False)))
    ):
        raise ValueError("Checkpoint uses removed FNO linear-baseline/residual path.")
    norm_mode = config.get("norm", "minmax")
    ns = NormStats.from_dict(stats, mode=norm_mode)
    x = dataset["x"]
    train_indices, val_indices, test_indices = build_split_indices(
        int(dataset["eta"].shape[0]),
        args.seed,
    )
    eval_indices = val_indices if args.split == "val" else test_indices
    if args.subset_size is not None and args.subset_size < eval_indices.shape[0]:
        subset_rng = np.random.default_rng(args.seed)
        eval_indices = np.sort(subset_rng.choice(eval_indices, size=args.subset_size, replace=False))

    if config.get("model", "fno") == "spectral_dno":
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
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(ns.feature_absmax).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )

    @jax.jit
    def predict_batch(current_params, batch_inputs, batch_depth):
        return model.apply({"params": current_params}, batch_inputs, batch_depth)

    rel_l2_all: list[np.ndarray] = []
    rel_l1_all: list[np.ndarray] = []
    example_idx_all: list[np.ndarray] = []
    case_id_all: list[np.ndarray] = []
    time_all: list[np.ndarray] = []
    prediction_norm_cache: list[np.ndarray] = []

    for eta_b, xi_b, _, depth_b, batch_indices in get_batches(
        dataset["eta"],
        dataset["xi"],
        dataset["gxi"],
        dataset["depth"],
        eval_indices,
        args.batch_size,
        None,
        shuffle=False,
        drop_last=False,
    ):
        batch_inputs = normalize_features(eta_b, xi_b, ns)
        batch_inputs_device = jax.device_put(batch_inputs, devices[0])
        batch_depth_device = jax.device_put(depth_b, devices[0])
        predicted_norm = np.asarray(jax.device_get(predict_batch(params, batch_inputs_device, batch_depth_device)))
        predicted_raw = denormalize_targets(predicted_norm, ns)
        target_raw = dataset["gxi"][batch_indices, :, None]

        flat_pred = predicted_raw.reshape((predicted_raw.shape[0], -1))
        flat_target = target_raw.reshape((target_raw.shape[0], -1))
        rel_l2 = np.linalg.norm(flat_pred - flat_target, axis=1) / (
            np.linalg.norm(flat_target, axis=1) + 1e-12
        )
        rel_l1 = np.sum(np.abs(flat_pred - flat_target), axis=1) / (
            np.sum(np.abs(flat_target), axis=1) + 1e-12
        )

        rel_l2_all.append(rel_l2.astype(np.float32))
        rel_l1_all.append(rel_l1.astype(np.float32))
        example_idx_all.append(batch_indices.astype(np.int64))
        case_id_all.append(dataset["case_id"][batch_indices])
        time_all.append(dataset["time"][batch_indices])
        prediction_norm_cache.append(predicted_norm.astype(np.float32))

    rel_l2 = np.concatenate(rel_l2_all, axis=0)
    rel_l1 = np.concatenate(rel_l1_all, axis=0)
    example_idx_vec = np.concatenate(example_idx_all, axis=0)
    case_id_vec = np.concatenate(case_id_all, axis=0)
    time_vec = np.concatenate(time_all, axis=0)
    prediction_norm_vec = np.concatenate(prediction_norm_cache, axis=0)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_jax_10m"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics = summarize_errors(rel_l2, rel_l1)

    sorted_indices = np.argsort(rel_l2)
    representative_indices = np.asarray(
        [
            int(sorted_indices[0]),
            int(sorted_indices[len(sorted_indices) // 2]),
            int(sorted_indices[-1]),
            int(np.random.default_rng(args.seed).integers(0, len(rel_l2))),
        ],
        dtype=np.int32,
    )

    if not args.skip_plots:
        rep_rows = example_idx_vec[representative_indices]
        rep_eta = dataset["eta"][rep_rows]
        rep_xi = dataset["xi"][rep_rows]
        rep_targets = dataset["gxi"][rep_rows, :, None]
        rep_predictions = denormalize_targets(prediction_norm_vec[representative_indices], ns)
        plot_representative_samples(
            output_dir / "representative_samples.png",
            f"Tanaka 10M: Representative {args.split.title()} Samples",
            np.asarray(x),
            rep_eta,
            rep_xi,
            rep_targets,
            rep_predictions,
            rel_l2[representative_indices],
            rel_l1[representative_indices],
            args.seed,
        )

    summary = {
        "backend": backend,
        "dataset": args.dataset,
        "seed": args.seed,
        "split": args.split,
        "subset_size": int(eval_indices.shape[0]),
        "batch_size": args.batch_size,
        "run_dir": str(run_dir),
        "carbs_root": str(carbs_root) if carbs_root is not None else None,
        "checkpoint": str(checkpoint_path),
        "num_examples": int(rel_l2.shape[0]),
        "num_train_examples": int(train_indices.shape[0]),
        "num_val_examples": int(val_indices.shape[0]),
        "num_test_examples": int(test_indices.shape[0]),
        "param_count": count_params(params),
        **overall_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        output_dir / "per_sample_metrics.npz",
        rel_l2=rel_l2,
        rel_l1=rel_l1,
        example_idx=example_idx_vec,
        case_id=case_id_vec,
        time=time_vec,
        representative_indices=representative_indices,
        x=np.asarray(x),
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
