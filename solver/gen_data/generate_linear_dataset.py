"""Generate (eta, xi) -> G(eta) xi data from single-mode linear-theory wavetrains.

No time integration. For each random (a0, n0, h, phase, direction) draw, evaluate
the linear-theory single-mode wave and compute G(eta) xi via the Craig-Sulem
Taylor series at order M.

Linear theory at z=0:
    eta = a0 * cos(k0 x + phi)
    xi  = a0 * (omega/G0) * sin(k0 x + phi)   for right-going  (+omega branch)
    xi  = -a0 * (omega/G0) * sin(k0 x + phi)  for left-going   (-omega branch)
    omega = sqrt(g * k0 * tanh(k0 h)),  G0 = k0 * tanh(k0 h).

Direction is randomized 50/50 to give bidirectional coverage. One snapshot per
case (random phi) gives independent samples.

Output layout matches the Stokes generator (per-batch zip entries, meta.json,
state sidecar).
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format
from tqdm.auto import tqdm

from ..solvers.dno_series_jax import build_grid, dno_series_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-mode linear (eta, xi, G eta xi) dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_samples", type=int, default=250_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng_stream_id", type=int, default=0)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--n0_min", type=int, default=1)
    parser.add_argument("--n0_max", type=int, default=20)
    parser.add_argument("--a0_min", type=float, default=0.001,
                        help="lower amplitude bound; matches MATLAB dno_series_gen_data.")
    parser.add_argument("--a0_max", type=float, default=0.3,
                        help="upper amplitude bound; matches MATLAB dno_series_gen_data.")
    parser.add_argument("--steepness_min", type=float, default=5e-3)
    parser.add_argument("--steepness_max", type=float, default=0.15)
    parser.add_argument("--rejection_attempts", type=int, default=1000,
                        help="MATLAB-style rejection attempts on a0 before falling back to "
                             "uniform draw on the feasible interval per case.")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=1.5)
    parser.add_argument("--kh_min", type=float, default=0.5,
                        help="lower kh bound; below this G0 ~ k0^2 h is tiny and xi blows up")
    parser.add_argument("--kh_max", type=float, default=20.0,
                        help="upper kh bound; above this tanh saturates (deep)")
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_npy_entry(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


def load_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def make_batch_rng(seed: int, batch_idx: int, rng_stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, rng_stream_id, batch_idx]))


def _sample_a0_matlab_style(
    rng: np.random.Generator, k0: np.ndarray, *,
    a0_min: float, a0_max: float, steepness_min: float, steepness_max: float,
    max_attempts: int,
) -> np.ndarray:
    """Mirror MATLAB sample_amplitude: draw a0 ~ Uniform[a0_min, a0_max] and accept
    if steepness_min <= k0*a0 <= steepness_max. For samples that fail after
    max_attempts retries, fall back to a direct uniform draw on the per-case
    feasible interval [max(a0_min, steepness_min/k0), min(a0_max, steepness_max/k0)].
    Raises ValueError if any case has an empty feasible interval.
    """
    n = k0.shape[0]
    a0 = rng.uniform(a0_min, a0_max, size=n).astype(np.float64)
    for _ in range(max_attempts):
        steepness = a0 * k0
        bad = (steepness < steepness_min) | (steepness > steepness_max)
        if not bad.any():
            return a0
        a0[bad] = rng.uniform(a0_min, a0_max, size=int(bad.sum()))

    steepness = a0 * k0
    bad = (steepness < steepness_min) | (steepness > steepness_max)
    if bad.any():
        a0_lower = np.maximum(a0_min, steepness_min / np.maximum(k0, np.finfo(np.float64).eps))
        a0_upper = np.minimum(a0_max, steepness_max / np.maximum(k0, np.finfo(np.float64).eps))
        infeasible = a0_lower > a0_upper
        if infeasible.any():
            raise ValueError(
                "No feasible amplitude range for some k0 (raise a0_max or relax steepness bounds): "
                f"first infeasible k0={float(k0[infeasible][0]):.6g}."
            )
        u = rng.uniform(size=int(bad.sum()))
        a0[bad] = a0_lower[bad] + (a0_upper[bad] - a0_lower[bad]) * u
    return a0


def sample_linear_case_params(
    rng: np.random.Generator, *, batch_size: int, length: float,
    n0_min: int, n0_max: int, a0_min: float, a0_max: float,
    steepness_min: float, steepness_max: float,
    depth_min: float, depth_max: float, kh_min: float, kh_max: float,
    rejection_attempts: int,
) -> dict[str, np.ndarray]:
    n0 = rng.integers(n0_min, n0_max + 1, size=batch_size).astype(np.int32)
    k0 = n0.astype(np.float64) * (2.0 * np.pi / length)

    a0 = _sample_a0_matlab_style(
        rng, k0, a0_min=a0_min, a0_max=a0_max,
        steepness_min=steepness_min, steepness_max=steepness_max,
        max_attempts=rejection_attempts,
    )

    lo = np.maximum(depth_min, kh_min / k0)
    hi = np.minimum(depth_max, kh_max / k0)
    if np.any(lo > hi):
        raise ValueError(
            f"kh constraint produces empty depth range; lo={lo.min():.3f}, hi={hi.max():.3f}"
        )
    if np.allclose(lo, hi):
        depth = lo
    else:
        u = rng.uniform(0.0, 1.0, size=batch_size)
        depth = np.exp(np.log(lo) + u * (np.log(np.maximum(hi, lo)) - np.log(lo))).astype(np.float64)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=batch_size).astype(np.float64)
    direction = (rng.integers(0, 2, size=batch_size) * 2 - 1).astype(np.int32)  # +1 or -1
    return {"n0": n0, "a0": a0, "depth": depth, "phase": phase, "direction": direction}


def serialize_specs(params: dict[str, np.ndarray]) -> list[dict[str, float]]:
    n = int(params["a0"].shape[0])
    return [
        {
            "n0": int(params["n0"][i]),
            "a0": float(params["a0"][i]),
            "depth": float(params["depth"][i]),
            "phase": float(params["phase"][i]),
            "direction": int(params["direction"][i]),
        }
        for i in range(n)
    ]


def build_linear_batch(
    *, x: jnp.ndarray, k: jnp.ndarray, length: float, gravity: float,
    n0: jnp.ndarray, a0: jnp.ndarray, depth: jnp.ndarray, phase: jnp.ndarray,
    direction: jnp.ndarray, dno_order: int, pad_factor: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized over batch dimension. Returns (eta, xi, gxi) of shape (B, NX).

    Matches MATLAB dno_series_gen_data.m:
        eta = a0*cos(k0*x)
        xi  = a0*g*cosh(k0*(eta + h))*sin(k0*x) / (om*cosh(k0*h))
    with omega = sqrt(g*k0*tanh(k0*h)). The cosh(k0*(eta+h))/cosh(k0*h)
    factor is the nonlinear-depth correction at the free surface.
    """
    k0 = n0 * (2.0 * jnp.pi / length)
    g0_mode = k0 * jnp.tanh(k0 * depth)
    omega = jnp.sqrt(gravity * g0_mode)

    theta = k0[:, None] * x[None, :] + phase[:, None]
    eta = a0[:, None] * jnp.cos(theta)

    depth_2d = depth[:, None]
    cosh_factor = jnp.cosh(k0[:, None] * (eta + depth_2d)) / jnp.cosh(k0[:, None] * depth_2d)
    xi = direction[:, None] * (a0[:, None] * gravity / omega[:, None]) * cosh_factor * jnp.sin(theta)

    gxi = dno_series_eval(eta, xi, k, depth_2d, dno_order, pad_factor=pad_factor)
    return eta, xi, gxi


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.json")

    if args.overwrite:
        if output_path.exists(): output_path.unlink()
        if state_path.exists(): state_path.unlink()

    samples_per_batch = args.batch_size
    n_batches = int(math.ceil(args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "linear_single_mode",
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "rollout_dtype": args.rollout_dtype,
                "n0_min": args.n0_min, "n0_max": args.n0_max,
                "a0_min": args.a0_min, "a0_max": args.a0_max,
                "steepness_min": args.steepness_min, "steepness_max": args.steepness_max,
                "rejection_attempts": args.rejection_attempts,
                "a0_distribution": "uniform_with_rejection_on_steepness_bounds",
                "depth_min": args.depth_min, "depth_max": args.depth_max,
                "kh_min": args.kh_min, "kh_max": args.kh_max,
                "depth_distribution": "log_uniform_clipped_to_kh_range",
                "bidirectional": True,
                "dno_order": args.dno_order, "pad_factor": args.pad_factor,
                "seed": args.seed, "rng_stream_id": args.rng_stream_id,
                "case_id_offset": args.case_id_offset,
                "nx": args.nx, "length": args.length, "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float32))
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        save_state(state_path, {
            "output_path": str(output_path), "samples_written": 0, "next_batch": 0,
            "n_batches_planned": n_batches, "complete": False,
        })
    else:
        samples_written = int(existing_state["samples_written"])
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    save_dtype = np.float32 if args.rollout_dtype == "float32" else np.float64
    x_grid_np, k_grid_np = build_grid(args.nx, args.length)
    x_jnp = jnp.asarray(x_grid_np, dtype=rollout_dtype)
    k_jnp = jnp.asarray(k_grid_np, dtype=rollout_dtype)

    @jax.jit
    def jit_build(n0, a0, depth, phase, direction):
        return build_linear_batch(
            x=x_jnp, k=k_jnp, length=args.length, gravity=args.gravity,
            n0=n0, a0=a0, depth=depth, phase=phase, direction=direction,
            dno_order=args.dno_order, pad_factor=args.pad_factor,
        )

    total_start = perf_counter()
    pbar = tqdm(range(next_batch, n_batches), desc="linear",
                initial=next_batch, total=n_batches, dynamic_ncols=True, mininterval=1.0)
    for batch_idx in pbar:
        batch_start = perf_counter()
        rng = make_batch_rng(args.seed, batch_idx, args.rng_stream_id)
        params = sample_linear_case_params(
            rng, batch_size=args.batch_size, length=args.length,
            n0_min=args.n0_min, n0_max=args.n0_max,
            a0_min=args.a0_min, a0_max=args.a0_max,
            steepness_min=args.steepness_min, steepness_max=args.steepness_max,
            depth_min=args.depth_min, depth_max=args.depth_max,
            kh_min=args.kh_min, kh_max=args.kh_max,
            rejection_attempts=args.rejection_attempts,
        )
        eta, xi, gxi = jit_build(
            jnp.asarray(params["n0"], dtype=rollout_dtype),
            jnp.asarray(params["a0"], dtype=rollout_dtype),
            jnp.asarray(params["depth"], dtype=rollout_dtype),
            jnp.asarray(params["phase"], dtype=rollout_dtype),
            jnp.asarray(params["direction"], dtype=rollout_dtype),
        )
        jax.block_until_ready(gxi)

        eta_np = np.asarray(jax.device_get(eta), dtype=save_dtype)
        xi_np = np.asarray(jax.device_get(xi), dtype=save_dtype)
        gxi_np = np.asarray(jax.device_get(gxi), dtype=save_dtype)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )
        depth_arr = params["depth"].astype(np.float32)

        remaining = args.target_samples - samples_written
        keep = min(remaining, eta_np.shape[0])
        eta_np = eta_np[:keep]; xi_np = xi_np[:keep]; gxi_np = gxi_np[:keep]
        global_case_ids = global_case_ids[:keep]
        depth_arr = depth_arr[:keep]

        with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            tag = f"batch_{batch_idx:04d}"
            write_npy_entry(zf, f"eta_{tag}.npy", eta_np)
            write_npy_entry(zf, f"xi_{tag}.npy", xi_np)
            write_npy_entry(zf, f"gxi_{tag}.npy", gxi_np)
            write_npy_entry(zf, f"case_id_{tag}.npy", global_case_ids)
            write_npy_entry(zf, f"depth_{tag}.npy", depth_arr)
            zf.writestr(f"specs_{tag}.json", json.dumps(serialize_specs(params)[:keep], indent=2))

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
        pbar.set_postfix(samples=f"{samples_written}/{args.target_samples}", batch_s=f"{batch_seconds:.2f}")
        save_state(state_path, {
            "output_path": str(output_path), "samples_written": samples_written,
            "next_batch": batch_idx + 1, "n_batches_planned": n_batches,
            "complete": samples_written >= args.target_samples,
            "last_batch_seconds": batch_seconds,
            "elapsed_seconds": perf_counter() - total_start,
        })
        if samples_written >= args.target_samples:
            break


if __name__ == "__main__":
    main()
