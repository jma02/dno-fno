from __future__ import annotations

import argparse
import os
from functools import partial
from time import perf_counter

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from sklearn.datasets import load_digits
from tqdm import tqdm


class MLP(nn.Module):
    hidden_dim: int = 256
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_classes)(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal multi-GPU JAX pmap smoke test on sklearn digits.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def require_multi_gpu() -> list[jax.Device]:
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, found {backend!r}")
    devices = list(jax.local_devices())
    if len(devices) < 2:
        raise RuntimeError(f"Expected at least 2 local GPUs, found {len(devices)}")
    return devices


def load_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]

    split = int(0.8 * x.shape[0])
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    return x_train, y_train, x_val, y_val


def reshape_for_pmap(x: np.ndarray, y: np.ndarray, n_devices: int) -> tuple[np.ndarray, np.ndarray]:
    per_device_batch = x.shape[0] // n_devices
    x = x.reshape((n_devices, per_device_batch, x.shape[-1]))
    y = y.reshape((n_devices, per_device_batch))
    return x, y


def iterate_train_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    n_devices: int,
    rng: np.random.Generator,
):
    indices = rng.permutation(x.shape[0])
    usable = (indices.shape[0] // batch_size) * batch_size
    indices = indices[:usable]
    for start in range(0, usable, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield reshape_for_pmap(x[batch_idx], y[batch_idx], n_devices)


def iterate_eval_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
):
    for start in range(0, x.shape[0], batch_size):
        yield x[start : start + batch_size], y[start : start + batch_size]


def main() -> None:
    args = parse_args()
    devices = require_multi_gpu()
    n_devices = len(devices)
    if args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    x_train, y_train, x_val, y_val = load_dataset(args.seed)
    print(f"devices: {[str(d) for d in devices]}", flush=True)
    print(
        f"train={x_train.shape[0]} val={x_val.shape[0]} batch_size={args.batch_size} per_device={args.batch_size // n_devices}",
        flush=True,
    )

    model = MLP(hidden_dim=args.hidden_dim, num_classes=10)
    dummy_x = jnp.zeros((1, x_train.shape[1]), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(args.seed), dummy_x)["params"]

    tx = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    replicated_state = jax_utils.replicate(state, devices=devices)

    @partial(jax.pmap, axis_name="devices")
    def train_step(current_state, batch_x, batch_y):
        def loss_fn(current_params):
            logits = current_state.apply_fn({"params": current_params}, batch_x)
            one_hot = jax.nn.one_hot(batch_y, 10)
            return optax.softmax_cross_entropy(logits, one_hot).mean()

        loss_value, grads = jax.value_and_grad(loss_fn)(current_state.params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss_value = jax.lax.pmean(loss_value, axis_name="devices")
        next_state = current_state.apply_gradients(grads=grads)
        return next_state, loss_value

    @jax.jit
    def eval_step(current_params, batch_x):
        return model.apply({"params": current_params}, batch_x)

    def unreplicate_params(replicated_params):
        return jax.device_put(jax_utils.unreplicate(replicated_params), devices[0])

    warmup_x = np.zeros((args.batch_size, x_train.shape[1]), dtype=np.float32)
    warmup_y = np.zeros((args.batch_size,), dtype=np.int32)
    warmup_x_p, warmup_y_p = reshape_for_pmap(warmup_x, warmup_y, n_devices)

    started = perf_counter()
    replicated_state, warmup_loss = train_step(replicated_state, warmup_x_p, warmup_y_p)
    jax.block_until_ready(warmup_loss)
    print(f"warmup train_step compile+execute: {perf_counter() - started:.2f}s", flush=True)

    started = perf_counter()
    warmup_logits = eval_step(unreplicate_params(replicated_state.params), jax.device_put(warmup_x, devices[0]))
    jax.block_until_ready(warmup_logits)
    print(f"warmup eval_step compile+execute: {perf_counter() - started:.2f}s", flush=True)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        rng = np.random.default_rng(args.seed + epoch)
        losses: list[float] = []
        for batch_x, batch_y in iterate_train_batches(x_train, y_train, args.batch_size, n_devices, rng):
            replicated_state, loss_value = train_step(replicated_state, batch_x, batch_y)
            losses.append(float(np.asarray(jax.device_get(loss_value))[0]))

        params_single = unreplicate_params(replicated_state.params)
        correct = 0
        total = 0
        for batch_x, batch_y in iterate_eval_batches(x_val, y_val, args.batch_size):
            logits = np.asarray(jax.device_get(eval_step(params_single, jax.device_put(batch_x, devices[0]))))
            pred = np.argmax(logits, axis=-1)
            correct += int(np.sum(pred == batch_y))
            total += int(batch_y.shape[0])

        train_loss = float(np.mean(losses)) if losses else float("inf")
        val_acc = correct / max(1, total)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}")
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_acc={val_acc:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
