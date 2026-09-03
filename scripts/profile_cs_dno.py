"""Profile a single training step of the cs_dno model.

Mirrors the canonical JAX training engine as closely as
possible: same shard_map(batch) over both GPUs, same relative-L2 loss, same
AdamW schedule, same batch=256 → 128/GPU, nx=1024.

Reports median of many measurements for:
  - Forward-only
  - Forward + backward (value_and_grad)
  - Full step (forward + backward + adamw + all_gather param sync)
  - Single CraigSulemBlock forward
  - Batched rfft round-trip on (128, 1024) and (128*8, 1024)

Uses ``time.perf_counter`` around ``jax.block_until_ready``.
"""

from __future__ import annotations

# XLA flags MUST be set before importing jax.
import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true",
)
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import statistics
import sys
import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

REPO_ROOT = Path(__file__).resolve().parent.parent
DNO_DIR = REPO_ROOT / "models" / "dno-net"
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
TRAIN_DIR = REPO_ROOT / "train-jax-10m"
for _d in (REPO_ROOT, DNO_DIR, FNO_DIR, TRAIN_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from util import replicate_pytree_from_host  # noqa: E402

from dno_net_v2 import CraigSulemDNO, CraigSulemBlock  # noqa: E402
from losses import count_params, relative_l2_loss  # noqa: E402


# ----- match the training CLI ------------------------------------------------
BATCH_SIZE = 256
NX = 1024
WIDTH = 512
N_BLOCKS = 8
LATENT = 256
N_POLYS = 3
MULT_HIDDEN = 128
LR = 2e-4
WEIGHT_DECAY = 1e-4
DOMAIN_LENGTH = 2.0 * float(np.pi)

# --norm=scale + defaults for aux losses: pushforward/hamiltonian/jac/psd all OFF.
# Under norm=scale the model sees inputs/targets divided by feature/target abs-max;
# realistic values are O(1) after normalization. Random O(1) inputs are fine for
# profiling — actual values only shift constant folds, not FFT/matmul time.


def median_ms(
    fn, *args, warmup: int = 5, iters: int = 20
) -> tuple[float, float, float]:
    """Median / p90 / min of ``fn(*args)`` in milliseconds."""
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    med = statistics.median(samples)
    p90 = samples[int(0.9 * (len(samples) - 1))]
    return med, p90, samples[0]


if __name__ == "__main__":
    devices = jax.devices()
    n_devices = len(devices)
    print(f"jax backend: {jax.default_backend()}  devices: {n_devices}")
    for d in devices:
        print(f"  {d}")
    assert n_devices >= 2, "This script expects both RTX 6000 Ada visible."
    if BATCH_SIZE % n_devices != 0:
        raise ValueError(
            f"batch {BATCH_SIZE} not divisible by device count {n_devices}"
        )

    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())

    compute_dtype = jnp.float32
    model = CraigSulemDNO(
        width=WIDTH,
        n_blocks=N_BLOCKS,
        latent=LATENT,
        n_polys=N_POLYS,
        use_first_deriv=True,
        use_second_deriv=True,
        use_half_deriv=True,
        use_hilbert=True,
        mult_hidden=MULT_HIDDEN,
        domain_length=DOMAIN_LENGTH,
        xi_scale=1.0,
        eta_scale=1.0,
        target_scale=1.0,
        h_clip_max=5.0,
    )

    # Init on device 0 (matches trainer).
    with jax.default_device(devices[0]):
        params = model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1, NX, 2), dtype=compute_dtype),
            jnp.zeros((1, 1), dtype=compute_dtype),
        )["params"]
    n_params = count_params(params)
    print(f"param count: {n_params:,}")

    # Realistic-ish random batch on host.
    rng = np.random.default_rng(0)
    inputs_host = rng.standard_normal((BATCH_SIZE, NX, 2)).astype(np.float32)
    depth_host = np.log(rng.uniform(0.5, 3.0, (BATCH_SIZE, 1))).astype(np.float32)
    targets_host = rng.standard_normal((BATCH_SIZE, NX, 1)).astype(np.float32)

    inputs = jax.device_put(inputs_host, data_sharding)
    depth = jax.device_put(depth_host, data_sharding)
    targets = jax.device_put(targets_host, data_sharding)

    # AdamW opt matching trainer.
    lr_sched = optax.cosine_decay_schedule(init_value=LR, decay_steps=100_000)
    opt = optax.adamw(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)
    ts = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    ts = replicate_pytree_from_host(ts, replicated)

    loss_fn = relative_l2_loss

    # ------------------------------------------------------------------
    # Case 1: full training step (forward + backward + AdamW + all_gather)
    # ------------------------------------------------------------------
    def _step_body(state, x, d_, y):
        def _loss(p):
            pred = state.apply_fn({"params": p}, x, d_)
            return loss_fn(pred, y)

        loss_v, grads = jax.value_and_grad(_loss)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_v = jax.lax.pmean(loss_v, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)
        # match trainer: all_gather[0] to keep replicas in sync
        new_state = new_state.replace(
            params=jax.tree.map(
                lambda p: jax.lax.all_gather(p, axis_name="batch", tiled=False)[0],
                new_state.params,
            )
        )
        return new_state, loss_v

    train_step = jax.jit(
        shard_map(
            _step_body,
            mesh=mesh,
            in_specs=(P(), P("batch"), P("batch"), P("batch")),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )

    # ------------------------------------------------------------------
    # Case 2: forward + backward only (no optimizer, no all_gather)
    # ------------------------------------------------------------------
    def _fwdbwd_body(params_, x, d_, y):
        def _loss(p):
            pred = cast(jax.Array, model.apply({"params": p}, x, d_))
            return loss_fn(pred, y)

        loss_v, grads = jax.value_and_grad(_loss)(params_)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_v = jax.lax.pmean(loss_v, axis_name="batch")
        return loss_v, grads

    fwdbwd_step = jax.jit(
        shard_map(
            _fwdbwd_body,
            mesh=mesh,
            in_specs=(P(), P("batch"), P("batch"), P("batch")),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )

    # ------------------------------------------------------------------
    # Case 3: forward only (no loss, no backward)
    # ------------------------------------------------------------------
    def _fwd_body(params_, x, d_):
        return model.apply({"params": params_}, x, d_)

    fwd_step = jax.jit(
        shard_map(
            _fwd_body,
            mesh=mesh,
            in_specs=(P(), P("batch"), P("batch")),
            out_specs=P("batch"),
            check_rep=False,
        )
    )

    # ------------------------------------------------------------------
    # Case 4: single CraigSulemBlock forward, on one GPU
    # ------------------------------------------------------------------
    block = CraigSulemBlock(
        n_branches=LATENT,
        domain_length=DOMAIN_LENGTH,
        h_clip_max=5.0,
        mult_hidden=MULT_HIDDEN,
    )
    B_local = BATCH_SIZE // n_devices  # 128
    eta_feats_shape = (B_local, NX, WIDTH // 2)  # 128 x 1024 x 256
    xi_phys_shape = (B_local, NX)
    depth_shape = (B_local, 1)
    with jax.default_device(devices[0]):
        eta_feats0 = jax.random.normal(
            jax.random.PRNGKey(1), eta_feats_shape, dtype=compute_dtype
        )
        xi_phys0 = jax.random.normal(
            jax.random.PRNGKey(2), xi_phys_shape, dtype=compute_dtype
        )
        depth0 = jnp.log(
            0.5
            + jax.random.uniform(
                jax.random.PRNGKey(3), depth_shape, dtype=compute_dtype
            )
            * 2.5
        )
        block_params = block.init(jax.random.PRNGKey(4), eta_feats0, xi_phys0, depth0)[
            "params"
        ]

    @jax.jit
    def block_fwd(bp, ef, xp, dp):
        return block.apply({"params": bp}, ef, xp, dp)

    # Warm block on device 0
    _ = block_fwd(block_params, eta_feats0, xi_phys0, depth0)
    jax.block_until_ready(_)

    # ------------------------------------------------------------------
    # Case 5: raw rfft round-trip at several shapes.
    #   - (128, 1024)              : ξ trunk FFT
    #   - (128*8, 1024)            : concatenated 8-block ξ FFT
    #   - (128, 1024, 256)         : per-branch shape used inside a block
    #                                (prods_hat = rfft(prods) : 128 x 1024 x 256)
    # These bracket the actual FFT cost in a step.
    # ------------------------------------------------------------------
    with jax.default_device(devices[0]):
        rfft_small = jax.random.normal(
            jax.random.PRNGKey(5), (B_local, NX), dtype=compute_dtype
        )
        rfft_big = jax.random.normal(
            jax.random.PRNGKey(6), (B_local * N_BLOCKS, NX), dtype=compute_dtype
        )
        rfft_branched = jax.random.normal(
            jax.random.PRNGKey(7),
            (B_local, NX, LATENT),
            dtype=compute_dtype,
        )

    @jax.jit
    def rfft_roundtrip(x):
        xh = jnp.fft.rfft(x, axis=-1, norm="forward")
        return jnp.fft.irfft(xh, n=NX, axis=-1, norm="forward")

    @jax.jit
    def rfft_roundtrip_branched(x):
        # x shape (B, N, L) — FFT along the spatial axis, matching CraigSulemBlock's
        # rfft(prods, axis=1) → irfft(..., axis=-1) usage on (B, N, L)-shaped fields.
        xh = jnp.fft.rfft(x, axis=1, norm="forward")
        return jnp.fft.irfft(xh, n=NX, axis=1, norm="forward")

    # ------------------------------------------------------------------
    # Compile + warm each case, then time.
    # ------------------------------------------------------------------
    print("\ncompiling / warming ...")
    # full step
    ts2, _ = train_step(ts, inputs, depth, targets)
    jax.block_until_ready(ts2)
    # fwd+bwd
    lv, gr = fwdbwd_step(ts.params, inputs, depth, targets)
    jax.block_until_ready((lv, gr))
    # fwd only
    out = fwd_step(ts.params, inputs, depth)
    jax.block_until_ready(out)
    # rfft small/big/branched
    jax.block_until_ready(rfft_roundtrip(rfft_small))
    jax.block_until_ready(rfft_roundtrip(rfft_big))
    jax.block_until_ready(rfft_roundtrip_branched(rfft_branched))

    print("timing ...")
    iters = 40
    results: dict[str, tuple[float, float, float]] = {}

    results["full_step (fwd+bwd+adamw+all_gather)"] = median_ms(
        lambda state: train_step(state, inputs, depth, targets),
        ts,
        iters=iters,
    )

    results["fwd+bwd (loss+grad, no opt)"] = median_ms(
        lambda params_: fwdbwd_step(params_, inputs, depth, targets),
        ts.params,
        iters=iters,
    )

    results["fwd only"] = median_ms(
        lambda params_: fwd_step(params_, inputs, depth),
        ts.params,
        iters=iters,
    )

    results["single CraigSulemBlock fwd (128x1024, dev0)"] = median_ms(
        lambda bp: block_fwd(bp, eta_feats0, xi_phys0, depth0),
        block_params,
        iters=iters,
    )

    results["rfft roundtrip (128, 1024) dev0"] = median_ms(
        rfft_roundtrip,
        rfft_small,
        iters=iters,
    )

    results[f"rfft roundtrip ({B_local * N_BLOCKS}, 1024) dev0"] = median_ms(
        rfft_roundtrip,
        rfft_big,
        iters=iters,
    )

    results[f"rfft branched ({B_local}, 1024, {LATENT}) dev0"] = median_ms(
        rfft_roundtrip_branched,
        rfft_branched,
        iters=iters,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'component':<50s}  {'median (ms)':>12s}  {'p90':>8s}  {'min':>8s}")
    print("-" * 78)
    for name, (med, p90, mn) in results.items():
        print(f"{name:<50s}  {med:>12.3f}  {p90:>8.3f}  {mn:>8.3f}")
    print("=" * 78)

    med_full = results["full_step (fwd+bwd+adamw+all_gather)"][0]
    med_fb = results["fwd+bwd (loss+grad, no opt)"][0]
    med_fw = results["fwd only"][0]
    print("\nDerived:")
    print(f"  bwd + comm + adamw ≈ full_step - fwd = {med_full - med_fw:.3f} ms")
    print(f"  bwd (from fwdbwd - fwd) ≈ {med_fb - med_fw:.3f} ms")
    print(f"  opt + all_gather ≈ full_step - fwdbwd = {med_full - med_fb:.3f} ms")
    print(f"  fwd fraction ≈ {100.0 * med_fw / med_full:.1f} %")
    print(f"  observed it/s (steady-state) ≈ {1000.0 / med_full:.2f}")
