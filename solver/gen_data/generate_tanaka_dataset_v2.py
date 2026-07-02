"""Generate the v2 Tanaka multicrest dataset.

Differences from v1 (`generate_tanaka_dataset.py`):
* computational domain fixed at L=2π (the rescaling reference frame from `scaling_notes.tex`);
* per-case depth `h_ref` sampled `logUniform(depth_min, depth_max)` so the operator
  is trained across the shallow→intermediate range in a single dataset;
* per-case crest amplitudes scaled by depth so steepness `amp/h_ref` recovers the v1
  range exactly at `h_ref=1` (and stays bounded across all sampled depths);
* per-case rollout runs in a single jitted batch by broadcasting depth as `(B, 1)`
  through `dno_series_eval` and `apply_linear_flow_hat` (verified bit-exact);
* per-sample `depth` array is stored alongside `eta`/`xi`/`gxi`/`time`/`case_id`.

Defaults: dt=0.01, tmax=200, subsample stride 800 (≈26 snapshots/case).
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

from ..gen_data.multi_crest import (
    DEFAULT_RANDOM_MAX_CRESTS,
    DEFAULT_RANDOM_MIN_CRESTS,
    DEFAULT_RANDOM_MIN_SEPARATION,
    flatten_case_specs,
    sample_sum_budgeted_cases,
    serialize_case_specs,
    specs_to_jax_arrays,
)
from ..solvers.dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol, myfft, myifft
from ..solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    spectral_dx,
)
from ..tanaka_ICs.modified_tanaka import make_default_tanaka_template, solve_modified_tanaka_batched
from .adaptive_sampling import adaptive_indices_from_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v2 Tanaka dataset on the [0, 2π] reference domain with varying depth.")
    parser.add_argument("--output", default="data/tanaka_2.npz")
    parser.add_argument("--target_samples", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_stride", type=int, default=800)
    parser.add_argument("--keep_samples", type=int)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument(
        "--rollout_dtype",
        choices=("float32", "float64"),
        default="float64",
        help="float64 needed at L=2π: k_max=512 with M=6 amplifies float32 eps to amplitude order, NaN'ing the rollout.",
    )
    parser.add_argument("--min_crests", type=int, default=DEFAULT_RANDOM_MIN_CRESTS)
    parser.add_argument("--max_crests", type=int, default=DEFAULT_RANDOM_MAX_CRESTS)
    parser.add_argument(
        "--separation_widths",
        type=float,
        default=3.0,
        help="Crest separation in soliton-widths (≈h_ref). Crest count and gap scale with h_ref so wide solitons don't overlap on [0, 2π].",
    )
    parser.add_argument(
        "--steepness_min",
        type=float,
        default=0.10,
        help="Case-level lower bound on Σ amp_i/h_ref (sum-budgeted across crests).",
    )
    parser.add_argument(
        "--steepness_max",
        type=float,
        default=0.35,
        help="Case-level upper bound on Σ amp_i/h_ref (sum-budgeted across crests).",
    )
    parser.add_argument(
        "--per_crest_steepness_floor",
        type=float,
        default=0.05,
        help="Minimum amp/h_ref per crest after Dirichlet split; below ~0.05 Tanaka is essentially linear.",
    )
    parser.add_argument("--depth_min", type=float, default=0.01)
    parser.add_argument(
        "--depth_max",
        type=float,
        default=0.08,
        help="Matches v1 (h=1, L=164) per-soliton occupancy 2·9.5·h/L ≈ 24% on L=2π. Above ~0.10 the soliton's 1%-tail clips the periodic boundary under random center placement; the resulting IC step gets amplified into the k=80–256 band by the order-M DNO series.",
    )
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use collision/focusing-aware per-case temporal subsampling driven by |d||η_x||²/dt|/||η_x||². "
             "Requires --keep_samples to be set (per-case K).",
    )
    parser.add_argument(
        "--adaptive_alpha",
        type=float,
        default=0.5,
        help="0 = uniform, 1 = activity-only. Mixes activity-weighted density with a uniform prior.",
    )
    parser.add_argument(
        "--adaptive_smooth_sigma",
        type=float,
        default=50.0,
        help="Gaussian smoothing of the activity signal in dense-time steps (dt units).",
    )
    parser.add_argument(
        "--adaptive_signal",
        choices=("envelope", "grad_energy"),
        default="envelope",
        help="Activity proxy: 'envelope' = ||eta||_inf, 'grad_energy' = ||eta_x||^2.",
    )
    parser.add_argument(
        "--adaptive_density_mode",
        choices=("rate", "magnitude"),
        default="rate",
        help="'rate' = |dS/dt|/S (good for localized events like Tanaka collisions). "
             "'magnitude' = S directly (good for continuous modulation like BF envelopes).",
    )
    parser.add_argument(
        "--adaptive_power",
        type=float,
        default=1.0,
        help="Sharpen contrast by raising (signal - per_case_min) to this power.",
    )
    return parser.parse_args()


def select_time_indices(n_times: int, subsample_stride: int, keep_samples: int | None) -> np.ndarray:
    if keep_samples is None:
        return np.arange(0, n_times, subsample_stride, dtype=np.int32)
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
    """Evaluate G(eta)*xi at each saved snapshot, chunked over the time axis.

    Inputs eta, xi have shape (T, B, nx). depth_per_case has shape (B, 1) and
    broadcasts inside dno_series_eval (verified bit-exact vs scalar depth).
    The output gxi is lowpass-filtered to match the rollout's invariant subspace
    (JCP09 eq. 28): without this, dno_series_eval's k^M-amplified high-k garbage
    appears in the saved gxi even though it was filtered out inside the rollout.
    """
    chunks: list[np.ndarray] = []
    n_steps = int(eta.shape[0])
    for start_idx in range(0, n_steps, chunk_size):
        end_idx = min(start_idx + chunk_size, n_steps)
        gxi_chunk = dno_series_eval(
            eta[start_idx:end_idx],
            xi[start_idx:end_idx],
            k,
            depth_per_case,  # (B, 1) broadcasts over T inside the (..., nx) ops
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
    depth_per_case: np.ndarray,  # shape (B,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta_case_major = np.swapaxes(eta, 0, 1).reshape(-1, eta.shape[-1])
    xi_case_major = np.swapaxes(xi, 0, 1).reshape(-1, xi.shape[-1])
    gxi_case_major = np.swapaxes(gxi, 0, 1).reshape(-1, gxi.shape[-1])
    sample_times = np.tile(times, global_case_ids.shape[0])
    sample_case_ids = np.repeat(global_case_ids, times.shape[0])
    sample_depths = np.repeat(depth_per_case, times.shape[0])
    return eta_case_major, xi_case_major, gxi_case_major, sample_times, sample_case_ids, sample_depths


def flatten_samples_per_case(
    eta_BKN: np.ndarray,             # (B, K, N) case-major
    xi_BKN: np.ndarray,
    gxi_BKN: np.ndarray,
    times_BK: np.ndarray,            # (B, K)
    global_case_ids: np.ndarray,     # (B,)
    depth_per_case: np.ndarray,      # (B,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = eta_BKN.shape[-1]
    k_per_case = eta_BKN.shape[1]
    eta_flat = eta_BKN.reshape(-1, nx)
    xi_flat = xi_BKN.reshape(-1, nx)
    gxi_flat = gxi_BKN.reshape(-1, nx)
    sample_times = times_BK.reshape(-1)
    sample_case_ids = np.repeat(global_case_ids, k_per_case)
    sample_depths = np.repeat(depth_per_case, k_per_case)
    return eta_flat, xi_flat, gxi_flat, sample_times, sample_case_ids, sample_depths


@jax.jit
def _grad_energy_traj(eta_traj: jnp.ndarray, length: float) -> jnp.ndarray:
    """Compute ``||η_x(t,b)||^2`` for ``eta_traj`` of shape (T, B, N). Returns (B, T).

    Scanned over time so the complex FFT intermediate is only (B, N) per step
    rather than (T, B, N) — the full-trajectory version OOMs at T=2501, B=256.
    """
    n = eta_traj.shape[-1]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=length / n)

    def step(_, eta_t: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        eta_hat = jnp.fft.fft(eta_t, axis=-1)
        eta_x = jnp.real(jnp.fft.ifft(1j * k * eta_hat, axis=-1))
        return None, jnp.sum(eta_x ** 2, axis=-1)

    _, s_TB = jax.lax.scan(step, None, eta_traj)
    return jnp.swapaxes(s_TB, 0, 1)


@jax.jit
def _envelope_peak_traj(eta_traj: jnp.ndarray) -> jnp.ndarray:
    return jnp.swapaxes(jnp.max(jnp.abs(eta_traj), axis=-1), 0, 1)


def _gather_per_case_jax(traj: jnp.ndarray, idx_BK: jnp.ndarray) -> jnp.ndarray:
    """Slice (T, B, N) by per-case indices (B, K) → (B, K, N)."""
    traj_BT = jnp.swapaxes(traj, 0, 1)  # (B, T, N)
    return jnp.take_along_axis(traj_BT, idx_BK[:, :, None], axis=1)


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


def build_per_case_initial_conditions(
    *,
    template_params,
    case_h_ref: np.ndarray,           # (B,)
    case_specs,                       # multi-crest case specs (length B)
    length: float,
    nx: int,
    gravity: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate per-case Tanaka multi-crest ICs at varying depths.

    Strategy: solve the dimensionless Tanaka template once for all crests
    (template_depth=1.0, amp = per-crest steepness); rescale each crest's
    physical (x, eta) by its case's h_ref; interpolate onto the [0, length]
    periodic grid; recompute per-crest xi via the surface-potential formula
    using per-crest speed = froude * sqrt(g * h_ref); sum crests within a case.
    """
    flat_specs, crest_case_ids = flatten_case_specs(case_specs)
    flat_steepness, flat_centers, flat_directions = specs_to_jax_arrays(flat_specs)
    n_crests = int(flat_steepness.shape[0])
    crest_case_ids_arr = jnp.asarray(crest_case_ids)

    # Dimensionless Tanaka solve (template_depth=1.0). Amplitudes here are
    # interpreted as the dimensionless steepness amp/depth; outputs are
    # dimensionless x_profile and eta_profile in template coordinates.
    tanaka_batch = solve_modified_tanaka_batched(
        template_params,
        flat_steepness,
        centers=jnp.zeros_like(flat_steepness),  # we'll re-place per-case
        directions=flat_directions,
    )
    x_profile = tanaka_batch.x_profile / template_params.depth   # back to dimensionless
    eta_profile = tanaka_batch.eta_profile / template_params.depth
    froude = tanaka_batch.froude  # (M,)

    # Per-crest physical placement on the [0, length] periodic grid.
    crest_h_ref = jnp.asarray(case_h_ref, dtype=x_profile.dtype)[crest_case_ids_arr]  # (M,)
    crest_centers = jnp.asarray(flat_centers, dtype=x_profile.dtype)

    # Place each soliton via fine-grid interp + FFT downsampling. `jnp.interp`
    # is piecewise-linear and injects high-k content that, while small in eta
    # (~1e-4 relative), gets amplified by the order-6 DNO series and produces
    # massive gxi noise. Truncating the FFT to nx modes after interp on a
    # high-resolution grid removes the linear-interp artifacts above the
    # target Nyquist before xi/gxi are computed.
    fine_factor = 8
    nx_fine = nx * fine_factor
    dx_fine = length / nx_fine
    x_fine = dx_fine * jnp.arange(nx_fine, dtype=x_profile.dtype)

    def interp_one(x_row: jnp.ndarray, eta_row: jnp.ndarray, h: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
        # 3-copy Poisson periodic extension: with random off-center placement a
        # plain interp(left=0, right=0) drops the soliton tail at x=0 or x=L,
        # producing a ~k^{-1} step that the order-M DNO amplifies into the
        # k=80–256 band. With depth_max=0.08 the 1% support 9.5·h ≤ 0.76 ≪ L,
        # so 3 copies (offsets {-L, 0, +L}) are exact to machine precision.
        x_shifted = x_row * h + center
        eta_scaled = eta_row * h
        f0 = jnp.interp(x_fine, x_shifted, eta_scaled, left=0.0, right=0.0)
        fm = jnp.interp(x_fine - length, x_shifted, eta_scaled, left=0.0, right=0.0)
        fp = jnp.interp(x_fine + length, x_shifted, eta_scaled, left=0.0, right=0.0)
        eta_fine = f0 + fm + fp
        eta_hat_fine = jnp.fft.rfft(eta_fine)
        n_keep = nx // 2 + 1
        eta_hat_native = eta_hat_fine[:n_keep] / fine_factor
        return jnp.fft.irfft(eta_hat_native, n=nx).astype(eta_row.dtype)

    eta_per_crest = jax.vmap(interp_one)(x_profile, eta_profile, crest_h_ref, crest_centers)  # (M, nx)

    # Per-crest xi via surface potential with per-crest depth/speed. Computed
    # crest-by-crest then summed (matching v1's superposition convention).
    speed_per_crest = froude * jnp.sqrt(gravity * crest_h_ref)
    direction_arr = jnp.asarray(flat_directions, dtype=eta_per_crest.dtype)
    signed_speed = (direction_arr * speed_per_crest)[..., None]
    direction_col = direction_arr[..., None]

    _, k_grid = build_grid(nx, length)
    eta_x = spectral_dx(eta_per_crest, k_grid)
    radical = (1.0 + eta_x**2) * (signed_speed**2 - 2.0 * gravity * eta_per_crest)
    xi_x = signed_speed - direction_col * jnp.sqrt(jnp.maximum(radical, 0.0))
    inv_ik = jnp.where(k_grid != 0.0, 1.0 / (1j * k_grid), 0.0)
    xi_per_crest = myifft(inv_ik * myfft(xi_x, nx))
    xi_per_crest = xi_per_crest - jnp.mean(xi_per_crest, axis=-1, keepdims=True)

    # Aggregate crests into cases.
    batch_size = case_h_ref.shape[0]
    eta_case = jnp.zeros((batch_size, nx), dtype=eta_per_crest.dtype)
    xi_case = jnp.zeros((batch_size, nx), dtype=xi_per_crest.dtype)
    eta_case = eta_case.at[crest_case_ids_arr].add(eta_per_crest)
    xi_case = xi_case.at[crest_case_ids_arr].add(xi_per_crest)
    return eta_case, xi_case


def main() -> None:
    args = parse_args()
    # Override filter_fraction below the shared default of 2/3: at L=2π, k_max=512 with M=6 means k_cut^M·eps_64 ~ 0.16, leaking k^M-amplified noise into the saved state. ν=1/4 → k_cut=128, comfortably above the Tanaka spectrum even at h=0.01.
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
    if args.adaptive:
        if args.keep_samples is None or args.keep_samples <= 1:
            raise SystemExit("--adaptive requires --keep_samples >= 2")
        subsample_indices = None
        subsample_times = None
        samples_per_batch = int(args.batch_size * args.keep_samples)
    else:
        subsample_indices = select_time_indices(times.shape[0], args.subsample_stride, args.keep_samples)
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
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "subsample_stride": args.subsample_stride,
                "keep_samples": args.keep_samples,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "tanaka_dtype": "float64",
                "rollout_dtype": args.rollout_dtype,
                "dt": args.dt,
                "tmax": args.tmax,
                "n_time_samples_full": int(times.shape[0]),
                "n_time_samples_subsampled": int(args.keep_samples) if args.adaptive else int(subsample_times.shape[0]),
                "adaptive_sampling": bool(args.adaptive),
                "adaptive_alpha": float(args.adaptive_alpha) if args.adaptive else None,
                "adaptive_smooth_sigma": float(args.adaptive_smooth_sigma) if args.adaptive else None,
                "adaptive_signal": args.adaptive_signal if args.adaptive else None,
                "adaptive_density_mode": args.adaptive_density_mode if args.adaptive else None,
                "adaptive_power": float(args.adaptive_power) if args.adaptive else None,
                "method": rollout_defaults.method,
                "substeps": rollout_defaults.substeps_per_interval,
                "implicit_iterations": rollout_defaults.implicit_iterations,
                "filter_fraction": rollout_defaults.filter_fraction,
                "zero_mean_xi": rollout_defaults.zero_mean_xi,
                "min_crests": args.min_crests,
                "max_crests": args.max_crests,
                "separation_widths": args.separation_widths,
                "steepness_min": args.steepness_min,
                "steepness_max": args.steepness_max,
                "per_crest_steepness_floor": args.per_crest_steepness_floor,
                "steepness_budget_kind": "sum_dirichlet",
                "depth_min": args.depth_min,
                "depth_max": args.depth_max,
                "depth_distribution": "log_uniform",
                "seed": args.seed,
                "case_id_offset": args.case_id_offset,
                "rng_stream_id": rng_stream_id,
                "nx": args.nx,
                "length": args.length,
                "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float32))
            if not args.adaptive:
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

    # Tanaka template at unit depth — gives the dimensionless soliton, rescaled per case.
    template_params = make_default_tanaka_template(
        depth=1.0,
        gravity=args.gravity,
        direction=1,
        nx=args.nx,
        length=args.length,
        center=0.0,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
    )

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)
    _, k_grid = build_grid(args.nx, args.length)
    k_grid = jnp.asarray(k_grid, dtype=rollout_dtype)

    total_start = perf_counter()
    for batch_idx in range(next_batch, n_batches):
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)

        # Per-case h_ref ~ logUniform(depth_min, depth_max).
        case_h_ref = sample_log_uniform(batch_rng, args.depth_min, args.depth_max, (args.batch_size,))

        # Case-level steepness budget S ~ U(steepness_min, steepness_max), split across
        # crests via Dirichlet. Crest count and separation scale with h_ref because
        # Tanaka soliton width ~ h_ref; at large h_ref only one crest fits on [0, 2π].
        case_specs: list[list] = []
        for hi in case_h_ref:
            min_sep = args.separation_widths * float(hi)
            n_max = int(max(1, math.floor(args.length / min_sep)))
            case_max_crests = min(args.max_crests, n_max)
            case_min_crests = min(args.min_crests, case_max_crests)
            specs_i = sample_sum_budgeted_cases(
                batch_rng,
                1,
                length=args.length,
                case_amplitude_min=args.steepness_min,
                case_amplitude_max=args.steepness_max,
                min_crests=case_min_crests,
                max_crests=case_max_crests,
                min_separation=min_sep,
                per_crest_floor=args.per_crest_steepness_floor,
            )
            case_specs.extend(specs_i)

        initial_eta, initial_xi = build_per_case_initial_conditions(
            template_params=template_params,
            case_h_ref=case_h_ref,
            case_specs=case_specs,
            length=args.length,
            nx=args.nx,
            gravity=args.gravity,
        )
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        # Project IC into the rollout's filtered subspace. Without this, the
        # first RK step calls dno_series_eval on raw high-k content; the order-M
        # series amplifies it (~k^M) and the result corrupts low-k modes via
        # nonlinear products before the per-step filter can act. JCP09 eq. (28)
        # keeps both η̂ and ξ̂ in the filtered subspace at all times.
        initial_eta = apply_lowpass(initial_eta, k_grid, rollout_defaults.filter_fraction)
        initial_xi = apply_lowpass(initial_xi, k_grid, rollout_defaults.filter_fraction)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        # Per-case solver_params: depth (B, 1), g0 (B, nx).
        depth_2d = jnp.asarray(case_h_ref, dtype=rollout_dtype)[:, None]
        g0_per_case = make_linear_dno_symbol(k_grid, depth_2d)
        solver_params = SolverParams(
            nx=args.nx,
            length=args.length,
            depth=depth_2d,
            gravity=args.gravity,
            dno_order=rollout_defaults.dno_order,
            pad_factor=rollout_defaults.pad_factor,
            filter_fraction=rollout_defaults.filter_fraction,
            k=k_grid,
            g0=g0_per_case,
        )
        solver_params = cast_solver_params_dtype(solver_params, rollout_dtype)

        rollout_payload = batched_rollout(
            cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
            rollout_times,
            solver_params,
            save_gxi=False,
            substeps_per_interval=rollout_defaults.substeps_per_interval,
            method=rollout_defaults.method,
            implicit_iterations=rollout_defaults.implicit_iterations,
            implicit_relaxation=rollout_defaults.implicit_relaxation,
            zero_mean_xi=rollout_defaults.zero_mean_xi,
        )
        jax.block_until_ready(rollout_payload["xi"])

        # Compute gxi in rollout_dtype before storing as float32; float32 dno_series_eval at k_max=512, M=6 is dominated by eps amplification.
        if args.adaptive:
            if args.adaptive_signal == "envelope":
                signal_BT = np.asarray(jax.device_get(_envelope_peak_traj(rollout_payload["eta"])))
            else:
                signal_BT = np.asarray(jax.device_get(_grad_energy_traj(rollout_payload["eta"], float(args.length))))
            idx_BK = adaptive_indices_from_energy(
                signal_BT,
                keep_samples=int(args.keep_samples),
                alpha=float(args.adaptive_alpha),
                smooth_sigma_steps=float(args.adaptive_smooth_sigma),
                density_mode=args.adaptive_density_mode,
                power=float(args.adaptive_power),
            )
            idx_BK_j = jnp.asarray(idx_BK)
            pred_eta_BKN = jax.device_get(_gather_per_case_jax(rollout_payload["eta"], idx_BK_j))
            pred_xi_BKN = jax.device_get(_gather_per_case_jax(rollout_payload["xi"], idx_BK_j))
            # gxi computation expects (T, B, N); transpose, chunk, transpose back.
            pred_eta_KBN = np.swapaxes(pred_eta_BKN, 0, 1)
            pred_xi_KBN = np.swapaxes(pred_xi_BKN, 0, 1)
            pred_gxi_KBN = chunked_gxi_over_time(
                jnp.asarray(pred_eta_KBN),
                jnp.asarray(pred_xi_KBN),
                k_grid,
                depth_2d,
                rollout_defaults.dno_order,
                rollout_defaults.pad_factor,
                rollout_defaults.filter_fraction,
                args.gxi_chunk_size,
            )
            pred_gxi_BKN = np.swapaxes(np.asarray(pred_gxi_KBN), 0, 1)
            times_BK = times[idx_BK].astype(np.float32)
            pred_eta_arr = pred_eta_BKN.astype(np.float32)
            pred_xi_arr = pred_xi_BKN.astype(np.float32)
            pred_gxi_arr = pred_gxi_BKN.astype(np.float32)
        else:
            pred_eta_native = jax.device_get(rollout_payload["eta"][subsample_indices])
            pred_xi_native = jax.device_get(rollout_payload["xi"][subsample_indices])
            pred_gxi_native = chunked_gxi_over_time(
                jnp.asarray(pred_eta_native),
                jnp.asarray(pred_xi_native),
                k_grid,
                depth_2d,
                rollout_defaults.dno_order,
                rollout_defaults.pad_factor,
                rollout_defaults.filter_fraction,
                args.gxi_chunk_size,
            )
            pred_eta_arr = np.asarray(pred_eta_native, dtype=np.float32)
            pred_xi_arr = np.asarray(pred_xi_native, dtype=np.float32)
            pred_gxi_arr = np.asarray(pred_gxi_native, dtype=np.float32)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )
        if args.adaptive:
            eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples_per_case(
                pred_eta_arr, pred_xi_arr, pred_gxi_arr, times_BK,
                global_case_ids, np.asarray(case_h_ref, dtype=np.float32),
            )
        else:
            eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples(
                pred_eta_arr, pred_xi_arr, pred_gxi_arr,
                subsample_times, global_case_ids, np.asarray(case_h_ref, dtype=np.float32),
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
            if args.adaptive:
                write_npy_entry(zf, f"subsample_indices_{batch_tag}.npy",
                                 idx_BK[:global_case_ids.shape[0]].astype(np.int32))
            zf.writestr(f"specs_{batch_tag}.json", json.dumps(serialize_case_specs(case_specs), indent=2))

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
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
