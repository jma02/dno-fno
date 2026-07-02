"""L-BFGS fine-tune a trained FNO checkpoint on the full training split.

Single-GPU full-batch L-BFGS via `optax.lbfgs` with zoom line search. The
training set is staged on-device once in fp16 (cast to fp32 inside the loss),
which keeps a 4M-sample, 1024-grid dataset under ~25 GB. Loss is computed
chunk-by-chunk via `lax.scan` with `jax.checkpoint` to bound activation
memory during backward. No collectives, so line-search while-loops won't
deadlock the way they do under shard_map + pmean.

Usage:
    uv run python train-jax-10m/lbfgs_finetune.py \\
        --run_dir outputs/<existing_run> \\
        --checkpoint best \\
        --chunk 64 --max_iters 100 --output_name w128b6_lbfgs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
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
from losses import build_loss, count_params
from util import (
    NormStats,
    build_split_indices,
    compute_log_depth,
    load_dataset_arrays,
    make_normalizers,
    require_jax_devices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True,
                   help="Run directory whose checkpoint provides the warm-start params.")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--dataset", default=None,
                   help="Override training dataset (defaults to config.json's value).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override split seed (defaults to config.json's value).")
    p.add_argument("--chunk", type=int, default=64,
                   help="Per-device chunk size processed per scan step.")
    p.add_argument("--max_iters", type=int, default=100)
    p.add_argument("--memory_size", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Cap step size. Default None lets the line search decide.")
    p.add_argument("--n_val", type=int, default=20_000,
                   help="Val subsample for monitoring. 0 disables.")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--fp16_data", action="store_true", default=True,
                   help="Store dataset on device in fp16 (cast to fp32 in the loss).")
    p.add_argument("--no_fp16_data", action="store_false", dest="fp16_data",
                   help="Store dataset on device in fp32 (uses ~2x memory).")
    p.add_argument("--output_name", default=None,
                   help="Run-dir name under outputs/. Default: <orig>_lbfgs_<ts>.")
    return p.parse_args()


def _load_params(run_dir: Path, which: str):
    ckpt_dir = run_dir / ("best_val_ckpt" if which == "best" else "final_ckpt")
    if not (ckpt_dir / "metadata.json").exists():
        raise FileNotFoundError(f"missing metadata at {ckpt_dir}")
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=None, prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return ckpt_dir, params, metadata


def _build_model(config: dict[str, object], stats: dict[str, object]):
    ns = NormStats.from_dict(stats, mode=config.get("norm", "minmax"))
    if config.get("model", "fno") == "spectral_dno":
        return SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(ns.feature_absmax).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )
    return FNO1d(
        modes=int(config["modes"]),
        width=int(config["width"]),
        n_blocks=int(config.get("n_blocks", 4)),
        domain_length=float(stats.get("domain_length", 2.0 * np.pi)),
        xi_scale=float(np.asarray(ns.feature_absmax).reshape(-1)[1]),
        target_scale=float(ns.target_absmax),
    )


def _save_ckpt(out_dir: Path, params, metadata: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"params": jax.tree_util.tree_map(lambda v: np.asarray(v), jax.device_get(params))}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    checkpoints.save_checkpoint(
        ckpt_dir=out_dir, target=payload, step=int(metadata.get("iter", 0)),
        prefix="ckpt_", keep=1, overwrite=True,
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )


def _stage_chunked(
    arr: np.ndarray,
    indices: np.ndarray,
    chunk: int,
    device: jax.Device,
    dtype: np.dtype,
) -> tuple[jax.Array, int]:
    """Gather arr[indices], reshape to (n_chunks, chunk, ...), and place on device.

    Drops trailing samples so that n_total = n_chunks * chunk.
    """
    sel = arr[indices]
    n_keep = (sel.shape[0] // chunk) * chunk
    sel = sel[:n_keep].astype(dtype, copy=False)
    n_chunks = n_keep // chunk
    sel = sel.reshape((n_chunks, chunk) + sel.shape[1:])
    return jax.device_put(sel, device), n_chunks


def main() -> None:
    args = parse_args()
    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)
    # Single-device run: pin everything to devices[0]. CUDA_VISIBLE_DEVICES can
    # narrow this externally if multiple GPUs are present.
    device = devices[0]
    data_dtype = np.float16 if args.fp16_data else np.float32

    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_dir, params, metadata = _load_params(run_dir, args.checkpoint)
    stats = metadata["stats"]
    norm_mode = config.get("norm", "minmax")
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))

    dataset_name = args.dataset or config["dataset"]
    dataset_path = (REPO_ROOT / "data" / dataset_name).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    dataset = load_dataset_arrays(dataset_path)
    n_total = int(dataset["eta"].shape[0])
    train_idx, val_idx, _ = build_split_indices(n_total, seed)
    print(f"backend={backend} device={device.platform}:{device.id} "
          f"data_dtype={data_dtype.__name__} train={train_idx.size} val={val_idx.size}")

    chunk = int(args.chunk)
    print(f"staging full training split to {device} as {data_dtype.__name__} ...")
    t0 = perf_counter()
    eta_d, n_chunks = _stage_chunked(dataset["eta"], train_idx, chunk, device, data_dtype)
    xi_d, _ = _stage_chunked(dataset["xi"], train_idx, chunk, device, data_dtype)
    gxi_d, _ = _stage_chunked(dataset["gxi"], train_idx, chunk, device, data_dtype)
    depth_arr = compute_log_depth(dataset["depth"]).reshape(-1, 1)
    depth_d, _ = _stage_chunked(depth_arr, train_idx, chunk, device, np.float32)
    n_used = n_chunks * chunk
    print(f"  staged {n_used:,} samples ({n_chunks} chunks, chunk={chunk}) in {perf_counter()-t0:.1f}s; "
          f"dropped {train_idx.size - n_used} trailing samples")

    if args.n_val > 0:
        rng = np.random.default_rng(seed + 1)
        n_val_target = int(min(args.n_val, val_idx.size))
        val_pick = np.sort(rng.choice(val_idx, size=n_val_target, replace=False))
        eta_v, n_val_chunks = _stage_chunked(dataset["eta"], val_pick, chunk, device, data_dtype)
        xi_v, _ = _stage_chunked(dataset["xi"], val_pick, chunk, device, data_dtype)
        gxi_v, _ = _stage_chunked(dataset["gxi"], val_pick, chunk, device, data_dtype)
        depth_v, _ = _stage_chunked(depth_arr, val_pick, chunk, device, np.float32)
        n_val_used = n_val_chunks * chunk
        print(f"  val subset: {n_val_used:,} samples ({n_val_chunks} chunks)")
    else:
        eta_v = xi_v = gxi_v = depth_v = None
        n_val_used = 0

    model = _build_model(config, stats)
    ns = NormStats.from_dict(stats, mode=norm_mode)
    norm_inputs_jax, norm_targets_jax, _ = make_normalizers(ns)
    base_loss = build_loss(sobolev_k=int(config.get("sobolev_k", 1)))

    def per_chunk_loss(p, eta_b, xi_b, gxi_b, depth_b):
        # Cast on the fly in case data is fp16 on device.
        eta_b = eta_b.astype(jnp.float32)
        xi_b = xi_b.astype(jnp.float32)
        gxi_b = gxi_b.astype(jnp.float32)
        inputs = norm_inputs_jax(eta_b, xi_b)
        targets = norm_targets_jax(gxi_b)
        preds = model.apply({"params": p}, inputs, depth_b)
        return base_loss(preds, targets)

    per_chunk_ck = jax.checkpoint(per_chunk_loss)

    @jax.jit
    def total_loss(p, eta, xi, gxi, depth):
        # eta etc shape (n_chunks, chunk, ...). Inner mean already applied by base_loss,
        # so summing per-chunk losses and dividing by n_chunks gives the global mean.
        def step(carry, inputs):
            e, x, g, d = inputs
            return carry + per_chunk_ck(p, e, x, g, d), None
        total, _ = jax.lax.scan(step, jnp.float32(0.0), (eta, xi, gxi, depth))
        return total / eta.shape[0]

    params = jax.device_put(params, device)

    print(f"params: {count_params(params):,}; compiling initial loss ...")
    t0 = perf_counter()
    init_value = float(jax.device_get(total_loss(params, eta_d, xi_d, gxi_d, depth_d)))
    print(f"  initial train loss = {init_value:.6f}  ({perf_counter() - t0:.1f}s)")
    init_val = float("nan")
    if eta_v is not None:
        init_val = float(jax.device_get(total_loss(params, eta_v, xi_v, gxi_v, depth_v)))
        print(f"  initial val loss   = {init_val:.6f}")

    solver = optax.lbfgs(
        learning_rate=args.learning_rate,
        memory_size=int(args.memory_size),
    )
    opt_state = solver.init(params)

    @jax.jit
    def step_fn(p, state, eta, xi, gxi, depth):
        # Build the per-step loss closure inside jit so the dataset arrays remain
        # *traced* arguments, not closed-over constants. Closing over them would
        # bake the entire (~25 GB fp16 / ~50 GB fp32) dataset into the compiled
        # HLO module as a constant.
        def loss_for_lbfgs(pp):
            def step(carry, inputs):
                e, x, g, d = inputs
                return carry + per_chunk_ck(pp, e, x, g, d), None
            total, _ = jax.lax.scan(step, jnp.float32(0.0), (eta, xi, gxi, depth))
            return total / eta.shape[0]

        value, grad = optax.value_and_grad_from_state(loss_for_lbfgs)(p, state=state)
        updates, state = solver.update(
            grad, state, p, value=value, grad=grad, value_fn=loss_for_lbfgs,
        )
        new_p = optax.apply_updates(p, updates)
        return new_p, state, value

    output_name = args.output_name or f"{run_dir.name}_lbfgs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_run = REPO_ROOT / "outputs" / output_name
    out_run.mkdir(parents=True, exist_ok=True)
    log_path = out_run / "lbfgs_log.jsonl"
    config_out = dict(config)
    config_out.update({
        "lbfgs": {
            "source_run": str(run_dir),
            "source_ckpt": str(ckpt_dir),
            "n_train_used": int(n_used),
            "n_val_used": int(n_val_used),
            "chunk": chunk,
            "max_iters": int(args.max_iters),
            "memory_size": int(args.memory_size),
            "learning_rate": args.learning_rate,
            "seed": int(seed),
            "init_loss": init_value,
            "init_val_loss": init_val,
            "data_dtype": data_dtype.__name__,
        },
    })
    (out_run / "config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    best_val = init_val if not (init_val != init_val) else float("inf")  # NaN-safe
    best_iter = 0
    best_params = params

    log_path.write_text("")
    print(f"running L-BFGS for {args.max_iters} iters; logs -> {log_path}")
    t_start = perf_counter()
    for it in range(1, int(args.max_iters) + 1):
        params, opt_state, value = step_fn(params, opt_state, eta_d, xi_d, gxi_d, depth_d)
        v = float(jax.device_get(value))
        record: dict[str, object] = {"iter": it, "loss": v, "elapsed_s": perf_counter() - t_start}
        if eta_v is not None and (it == 1 or it % int(args.val_every) == 0 or it == args.max_iters):
            vv = float(jax.device_get(total_loss(params, eta_v, xi_v, gxi_v, depth_v)))
            record["val_loss"] = vv
            if vv < best_val:
                best_val = vv
                best_iter = it
                best_params = params
        with log_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(record) + "\n")
        if it == 1 or it % int(args.val_every) == 0 or it == args.max_iters:
            extra = f"  val={record.get('val_loss', float('nan')):.6f}" if "val_loss" in record else ""
            print(f"  iter {it:4d}  loss={v:.6f}{extra}  ({record['elapsed_s']:.0f}s)")

    final_meta = {
        "iter": int(args.max_iters),
        "stats": stats,
        "source_run": str(run_dir),
        "source_ckpt": str(ckpt_dir),
        "init_loss": init_value,
        "final_loss": float(jax.device_get(total_loss(params, eta_d, xi_d, gxi_d, depth_d))),
        "init_val_loss": init_val,
        "final_val_loss": (
            float(jax.device_get(total_loss(params, eta_v, xi_v, gxi_v, depth_v)))
            if eta_v is not None else float("nan")
        ),
        "best_val_loss": best_val,
        "best_iter": best_iter,
        "param_count": count_params(params),
    }
    _save_ckpt(out_run / "final_ckpt", params, final_meta)
    if eta_v is not None and best_iter > 0:
        _save_ckpt(out_run / "best_val_ckpt", best_params, {**final_meta, "iter": best_iter})
    print(json.dumps({k: v for k, v in final_meta.items() if k != "stats"}, indent=2))
    print(f"saved -> {out_run}")


if __name__ == "__main__":
    main()
