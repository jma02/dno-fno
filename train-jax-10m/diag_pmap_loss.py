"""Definitive diagnostic: test value_and_grad with sharded vs single-device data."""
from __future__ import annotations

import os
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "models", "fno-jax")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fno1d import FNO1d
from losses import build_loss

devices = jax.devices()
n = len(devices)
mesh = Mesh(np.array(devices), axis_names=("batch",))
data_shard = NamedSharding(mesh, P("batch"))
replicated = NamedSharding(mesh, P())
print(f"JAX {jax.__version__}, {n} devices")

# ── setup ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
B = 128  # total batch (64 per device with 2 GPUs)
nx = 512
model = FNO1d(modes=64, width=64, n_blocks=5)

with jax.default_device(devices[0]):
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, nx, 2), dtype=jnp.float32))["params"]
params_rep = jax.device_put(params, replicated)

inp = rng.standard_normal((B, nx, 2)).astype(np.float32)
tgt = rng.standard_normal((B, nx, 1)).astype(np.float32)

for loss_name in ("relative_l2", "mse"):
    loss_fn = build_loss(loss_name)
    print(f"\n=== {loss_name} loss ===")

    # A) single-device eval (ground truth)
    @jax.jit
    def eval_loss(p, x, y):
        return loss_fn(model.apply({"params": p}, x), y)

    p0 = jax.device_put(params, devices[0])
    x0 = jax.device_put(inp, devices[0])
    y0 = jax.device_put(tgt, devices[0])
    gt = float(jax.device_get(eval_loss(p0, x0, y0)))
    print(f"  A) single-device eval     = {gt:.6f}")

    # B) sharded forward only (no grad)
    @jax.jit
    def sharded_eval(p, x, y):
        return loss_fn(model.apply({"params": p}, x), y)

    xs = jax.device_put(inp, data_shard)
    ys = jax.device_put(tgt, data_shard)
    b = float(jax.device_get(sharded_eval(params_rep, xs, ys)))
    print(f"  B) sharded eval (no grad) = {b:.6f}  ratio={b/gt:.4f}")

    # C) sharded value_and_grad (the actual training scenario)
    @jax.jit
    def sharded_vg(p, x, y):
        def f(pp):
            return loss_fn(model.apply({"params": pp}, x), y)
        val, _ = jax.value_and_grad(f)(p)
        return val

    c = float(jax.device_get(sharded_vg(params_rep, xs, ys)))
    print(f"  C) sharded value_and_grad = {c:.6f}  ratio={c/gt:.4f}")

    # D) single-device value_and_grad
    @jax.jit
    def single_vg(p, x, y):
        def f(pp):
            return loss_fn(model.apply({"params": pp}, x), y)
        val, _ = jax.value_and_grad(f)(p)
        return val

    d = float(jax.device_get(single_vg(p0, x0, y0)))
    print(f"  D) single value_and_grad  = {d:.6f}  ratio={d/gt:.4f}")

print("\nDone.")
