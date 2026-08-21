"""Generate a random-sea-state rollout dataset on the L=164 reference domain.

Mirrors generate_tanaka_dataset_v2.py / generate_bf_dataset.py: same rollout
settings, same NPZ + state.json sidecar, same per-sample (eta, xi, gxi, time,
case_id, depth) layout.

Differs from the legacy deep-water random-sea IC generator, preserved in Git
history, in three ways:
1. **Finite-depth xi transfer.** The original deep-water transfer
   xi_hat = -i sign(k) sqrt(g/|k|) eta_hat is replaced by the linearized
   finite-depth dispersion
       omega = sqrt(g |k| tanh(|k| h)),
       xi_hat = -i sign(k) (omega / |k|) eta_hat.
   At h >> 1/k_min this reduces to the deep-water form bit-exact.
2. **Per-case depth sampling.** Each case in a batch gets its own
   h ~ logUniform(depth_min, depth_max). The rollout's SolverParams.depth
   broadcasts as (B, 1) (matching the v2 Tanaka pattern).
3. **Full rollout.** Instead of saving just the IC, we run batched_rollout
   for tmax with subsampling, then evaluate gxi at the saved snapshots.

The default domain preserves the historical experimental convention:
L=164, NX=1024, g=1. Default Hs/kp/bw ranges match the recorded legacy
generator and `notes/random_sea_generation.tex`.
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

from ..solvers.dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol
from ..solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random-sea rollout dataset on the [0, L] domain (default L=164) at finite depth."
    )
    parser.add_argument("--output", default="data/random_sea_2.npz")
    parser.add_argument("--target_samples", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_samples", type=int, default=200)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument(
        "--rollout_dtype", choices=("float32", "float64"), default="float64",
        help="float64 needed: order-6 DNO at k_max=NX/2 amplifies float32 eps to amplitude order under the rollout.",
    )
    # Spectrum hyperparameter ranges preserve the legacy generator defaults.
    parser.add_argument("--Hs_lo", type=float, default=0.1)
    parser.add_argument("--Hs_hi", type=float, default=0.6)
    parser.add_argument("--kp_lo", type=float, default=0.06)
    parser.add_argument("--kp_hi", type=float, default=0.50)
    parser.add_argument("--bw_lo", type=float, default=1.5)
    parser.add_argument("--bw_hi", type=float, default=5.0)
    parser.add_argument(
        "--depth_min", type=float, default=1.0,
        help="logUniform lower bound. At default kp_lo=0.06, kh_min ~ 0.06 (shallow).",
    )
    parser.add_argument(
        "--depth_max", type=float, default=100.0,
        help="logUniform upper bound. At default kp_lo=0.06, kh_max=6 (deep). Spans shallow->deep.",
    )
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_time_indices(n_times: int, keep_samples: int) -> np.ndarray:
    if keep_samples <= 0:
        raise ValueError("--keep_samples must be positive.")
    if keep_samples >= n_times:
        return np.arange(n_times, dtype=np.int32)
    return np.linspace(0, n_times - 1, keep_samples, dtype=np.int32)


def chunked_gxi_over_time(
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    k: jnp.ndarray,
    depth_per_case: jnp.ndarray,  # shape (B, 1)
    dno_order: int,
    pad_factor: int,
    filter_fraction: float,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_steps = int(eta.shape[0])
    for start_idx in range(0, n_steps, chunk_size):
        end_idx = min(start_idx + chunk_size, n_steps)
        gxi_chunk = dno_series_eval(
            eta[start_idx:end_idx],
            xi[start_idx:end_idx],
            k,
            depth_per_case,
            dno_order,
            pad_factor=pad_factor,
        )
        gxi_chunk = apply_lowpass(gxi_chunk, k, filter_fraction)
        chunks.append(np.asarray(jax.device_get(gxi_chunk)))
    return np.concatenate(chunks, axis=0)


def flatten_samples(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    times: np.ndarray,
    global_case_ids: np.ndarray,
    depth_per_case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta_case_major = np.swapaxes(eta, 0, 1).reshape(-1, eta.shape[-1])
    xi_case_major = np.swapaxes(xi, 0, 1).reshape(-1, xi.shape[-1])
    gxi_case_major = np.swapaxes(gxi, 0, 1).reshape(-1, gxi.shape[-1])
    sample_times = np.tile(times, global_case_ids.shape[0])
    sample_case_ids = np.repeat(global_case_ids, times.shape[0])
    sample_depths = np.repeat(depth_per_case, times.shape[0])
    return eta_case_major, xi_case_major, gxi_case_major, sample_times, sample_case_ids, sample_depths


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
    seed_sequence = np.random.SeedSequence([seed, rng_stream_id, batch_idx])
    return np.random.default_rng(seed_sequence)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, shape: tuple[int, ...]) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=shape))


def sample_random_sea_case_params(
    rng: np.random.Generator,
    *,
    batch_size: int,
    Hs_lo: float, Hs_hi: float,
    kp_lo: float, kp_hi: float,
    bw_lo: float, bw_hi: float,
    depth_min: float, depth_max: float,
) -> dict[str, np.ndarray]:
    return {
        "Hs": rng.uniform(Hs_lo, Hs_hi, size=batch_size).astype(np.float64),
        "kp": rng.uniform(kp_lo, kp_hi, size=batch_size).astype(np.float64),
        "bw": rng.uniform(bw_lo, bw_hi, size=batch_size).astype(np.float64),
        "depth": sample_log_uniform(rng, depth_min, depth_max, (batch_size,)).astype(np.float64),
    }


def serialize_random_sea_specs(params: dict[str, np.ndarray]) -> list[dict[str, float]]:
    n = int(params["Hs"].shape[0])
    return [
        {
            "Hs": float(params["Hs"][i]),
            "kp": float(params["kp"][i]),
            "bw": float(params["bw"][i]),
            "depth": float(params["depth"][i]),
        }
        for i in range(n)
    ]


def build_random_sea_initial_conditions(
    rng: np.random.Generator,
    *,
    k_np: np.ndarray,
    params: dict[str, np.ndarray],
    gravity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (B, nx) bidirectional eta and xi from per-case (Hs, kp, bw, depth).

    Spectrum: S(k) = exp(-(|k|-kp)^2 / (2 sigma_k^2)), sigma_k = kp / bw.
    Two independent right- and left-going random fields are sampled with
    the same spectrum and independent phases, then summed:
        eta_R, eta_L  (independent random surfaces, each std = Hs/(4 sqrt(2)))
        xi_R_hat = -i sign(k) (omega/G_0) eta_R_hat   (right-going branch, +omega)
        xi_L_hat = +i sign(k) (omega/G_0) eta_L_hat   (left-going branch,  -omega)
        eta = eta_R + eta_L,  xi = xi_R + xi_L
    so the total energy splits 50/50 between rightward- and leftward-traveling
    components, giving std(eta) = Hs/4. xi_ratio = omega/G_0 reduces to
    sqrt(g/|k|) in deep water, matching the original deep-water formula bit-exact.
    """
    Hs = params["Hs"]
    kp = params["kp"]
    bw = params["bw"]
    depth = params["depth"]
    B = Hs.shape[0]
    n = k_np.shape[0]
    abs_k = np.abs(k_np)
    sign_k = np.sign(k_np)

    sigma_k = (kp / bw)[:, None]
    S = np.exp(-0.5 * ((abs_k[None, :] - kp[:, None]) / sigma_k) ** 2)
    S[:, 0] = 0.0
    amplitude = np.sqrt(S)

    def random_eta_hat() -> np.ndarray:
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(B, n))
        eta_hat = np.zeros((B, n), dtype=np.complex128)
        eta_hat[:, 1 : n // 2] = amplitude[:, 1 : n // 2] * np.exp(1j * phase[:, 1 : n // 2])
        eta_hat[:, n // 2 + 1 :] = np.conj(eta_hat[:, 1 : n // 2][:, ::-1])
        return eta_hat

    eta_R_hat_unit = random_eta_hat()
    eta_L_hat_unit = random_eta_hat()
    eta_R = np.real(np.fft.ifft(eta_R_hat_unit, axis=-1) * n)
    eta_L = np.real(np.fft.ifft(eta_L_hat_unit, axis=-1) * n)
    eta = eta_R + eta_L

    eta_std = np.std(eta, axis=-1, keepdims=True)
    eta_std = np.maximum(eta_std, 1e-14)
    rescale = ((Hs / 4.0)[:, None] / eta_std)
    eta_R = eta_R * rescale
    eta_L = eta_L * rescale
    eta = eta_R + eta_L

    eta_R_fft = np.fft.fft(eta_R, axis=-1) / n
    eta_L_fft = np.fft.fft(eta_L, axis=-1) / n
    G0 = abs_k[None, :] * np.tanh(abs_k[None, :] * depth[:, None])
    omega = np.sqrt(gravity * G0)
    xi_ratio = np.where(G0 > 0, omega / np.maximum(G0, 1e-30), 0.0)
    xi_hat = -1j * sign_k[None, :] * xi_ratio * (eta_R_fft - eta_L_fft)
    xi = np.real(np.fft.ifft(xi_hat, axis=-1) * n)
    return eta, xi


def main() -> None:
    args = parse_args()
    rollout_defaults = make_normalized_rollout_settings()._replace(filter_fraction=0.25)
    rng_stream_id = args.rng_stream_id if args.rng_stream_id is not None else args.case_id_offset

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.json")

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if state_path.exists():
            state_path.unlink()

    times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float32)
    subsample_indices = select_time_indices(times.shape[0], args.keep_samples)
    subsample_times = times[subsample_indices]
    samples_per_batch = int(args.batch_size * subsample_times.shape[0])
    n_batches = int(math.ceil(args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "random_sea_rollout_v2",
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "keep_samples": args.keep_samples,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "rollout_dtype": args.rollout_dtype,
                "dt": args.dt,
                "tmax": args.tmax,
                "n_time_samples_full": int(times.shape[0]),
                "n_time_samples_subsampled": int(subsample_times.shape[0]),
                "method": rollout_defaults.method,
                "substeps": rollout_defaults.substeps_per_interval,
                "implicit_iterations": rollout_defaults.implicit_iterations,
                "filter_fraction": rollout_defaults.filter_fraction,
                "zero_mean_xi": rollout_defaults.zero_mean_xi,
                "Hs_lo": args.Hs_lo, "Hs_hi": args.Hs_hi,
                "kp_lo": args.kp_lo, "kp_hi": args.kp_hi,
                "bw_lo": args.bw_lo, "bw_hi": args.bw_hi,
                "depth_min": args.depth_min, "depth_max": args.depth_max,
                "depth_distribution": "log_uniform",
                "spectrum_kind": "gaussian",
                "xi_dispersion": "finite_depth_linear",
                "seed": args.seed,
                "case_id_offset": args.case_id_offset,
                "rng_stream_id": rng_stream_id,
                "nx": args.nx,
                "length": args.length,
                "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float32))
            write_npy_entry(zf, "subsample_indices.npy", subsample_indices)
            write_npy_entry(zf, "subsample_times.npy", subsample_times)
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": 0,
                "next_batch": 0,
                "n_batches_planned": n_batches,
                "complete": False,
            },
        )
    else:
        samples_written = int(existing_state["samples_written"])
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)
    _, k_grid = build_grid(args.nx, args.length)
    k_np = np.asarray(k_grid)
    k_grid = jnp.asarray(k_grid, dtype=rollout_dtype)

    total_start = perf_counter()
    pbar = tqdm(
        range(next_batch, n_batches),
        desc="batch", initial=next_batch, total=n_batches,
        dynamic_ncols=True, mininterval=1.0,
    )
    for batch_idx in pbar:
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)

        params = sample_random_sea_case_params(
            batch_rng, batch_size=args.batch_size,
            Hs_lo=args.Hs_lo, Hs_hi=args.Hs_hi,
            kp_lo=args.kp_lo, kp_hi=args.kp_hi,
            bw_lo=args.bw_lo, bw_hi=args.bw_hi,
            depth_min=args.depth_min, depth_max=args.depth_max,
        )

        initial_eta_np, initial_xi_np = build_random_sea_initial_conditions(
            batch_rng, k_np=k_np, params=params, gravity=args.gravity,
        )
        initial_eta = jnp.asarray(initial_eta_np, dtype=rollout_dtype)
        initial_xi = jnp.asarray(initial_xi_np, dtype=rollout_dtype)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)
        initial_eta = apply_lowpass(initial_eta, k_grid, rollout_defaults.filter_fraction)
        initial_xi = apply_lowpass(initial_xi, k_grid, rollout_defaults.filter_fraction)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        depth_2d = jnp.asarray(params["depth"], dtype=rollout_dtype)[:, None]
        g0_per_case = make_linear_dno_symbol(k_grid, depth_2d)
        solver_params = SolverParams(
            nx=args.nx, length=args.length, depth=depth_2d, gravity=args.gravity,
            dno_order=rollout_defaults.dno_order, pad_factor=rollout_defaults.pad_factor,
            filter_fraction=rollout_defaults.filter_fraction, k=k_grid, g0=g0_per_case,
        )
        solver_params = cast_solver_params_dtype(solver_params, rollout_dtype)

        rollout_payload = batched_rollout(
            cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
            rollout_times, solver_params, save_gxi=False,
            substeps_per_interval=rollout_defaults.substeps_per_interval,
            method=rollout_defaults.method,
            implicit_iterations=rollout_defaults.implicit_iterations,
            implicit_relaxation=rollout_defaults.implicit_relaxation,
            zero_mean_xi=rollout_defaults.zero_mean_xi,
        )
        jax.block_until_ready(rollout_payload["xi"])

        pred_eta_native = jax.device_get(rollout_payload["eta"][subsample_indices])
        pred_xi_native = jax.device_get(rollout_payload["xi"][subsample_indices])
        pred_gxi_native = chunked_gxi_over_time(
            jnp.asarray(pred_eta_native), jnp.asarray(pred_xi_native),
            k_grid, depth_2d,
            rollout_defaults.dno_order, rollout_defaults.pad_factor,
            rollout_defaults.filter_fraction, args.gxi_chunk_size,
        )
        pred_eta = np.asarray(pred_eta_native, dtype=np.float32)
        pred_xi = np.asarray(pred_xi_native, dtype=np.float32)
        pred_gxi = np.asarray(pred_gxi_native, dtype=np.float32)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )
        eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples(
            pred_eta, pred_xi, pred_gxi, subsample_times, global_case_ids,
            np.asarray(params["depth"], dtype=np.float32),
        )

        remaining = args.target_samples - samples_written
        keep = min(remaining, eta_samples.shape[0])
        eta_samples = eta_samples[:keep]
        xi_samples = xi_samples[:keep]
        gxi_samples = gxi_samples[:keep]
        time_samples = time_samples[:keep]
        case_id_samples = case_id_samples[:keep]
        depth_samples = depth_samples[:keep]

        with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            batch_tag = f"batch_{batch_idx:04d}"
            write_npy_entry(zf, f"eta_{batch_tag}.npy", eta_samples)
            write_npy_entry(zf, f"xi_{batch_tag}.npy", xi_samples)
            write_npy_entry(zf, f"gxi_{batch_tag}.npy", gxi_samples)
            write_npy_entry(zf, f"time_{batch_tag}.npy", time_samples)
            write_npy_entry(zf, f"case_id_{batch_tag}.npy", case_id_samples)
            write_npy_entry(zf, f"depth_{batch_tag}.npy", depth_samples)
            zf.writestr(f"specs_{batch_tag}.json", json.dumps(serialize_random_sea_specs(params), indent=2))

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
        pbar.set_postfix(samples=f"{samples_written}/{args.target_samples}", batch_s=f"{batch_seconds:.1f}")
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": samples_written,
                "next_batch": batch_idx + 1,
                "n_batches_planned": n_batches,
                "complete": samples_written >= args.target_samples,
                "last_batch_seconds": batch_seconds,
                "elapsed_seconds": perf_counter() - total_start,
            },
        )
        print(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "samples_written": samples_written,
                    "target_samples": args.target_samples,
                    "batch_seconds": batch_seconds,
                    "device": str(jax.devices()[0]),
                }
            ),
            flush=True,
        )
        if samples_written >= args.target_samples:
            break


if __name__ == "__main__":
    main()
