"""Collision-aware temporal subsampling for soliton/wave rollouts.

The default uniform subsamplers in the Tanaka and BF generators distribute K
snapshots evenly over [0, tmax]. With ~26 snapshots over 200 s of evolution
the rare, fast events that matter most for FNO accuracy — soliton focusing /
collisions, BF envelope recurrence — fall between samples and are
under-represented at training time.

We use **surface-gradient energy** ``S(t) = ||η_x(t)||^2`` as the activity
signal. The Zakharov Hamiltonian is conserved, so a quantity that isn't
conserved is needed to flag state change; ``S(t)`` spikes during focusing
events and is exactly the quantity that controls the difficulty of
``G(η)ξ`` (the ``1 + η_x²`` factor in the nonlinear RHS). Per-case indices
are drawn from::

    density(t) = (1 - alpha) * uniform + alpha * smooth(|dS/dt| / (S + eps))

by inverse-CDF at K equispaced quantiles — deterministic, no RNG, keeps
samples spread out while concentrating extra mass on focusing events.
"""
from __future__ import annotations

import numpy as np


def _gaussian_smooth_1d(x: np.ndarray, sigma_steps: float) -> np.ndarray:
    if sigma_steps <= 0:
        return x
    radius = int(max(1, np.ceil(3.0 * sigma_steps)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma_steps) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def surface_gradient_energy(eta: np.ndarray, length: float) -> np.ndarray:
    """Per-case ``||η_x||^2`` via spectral derivative.

    Good activity signal for soliton collisions (Tanaka): the destructive
    tail-overlap dips and constructive peak both drive |dS/dt|. Not great for
    BF modulation because the fast carrier oscillation dominates.

    Args:
        eta: shape (T, B, N) — surface elevation trajectory.
        length: physical domain length (period).
    Returns:
        ``S[b, t] = sum_x η_x(t,b,x)^2`` of shape (B, T).
    """
    eta_np = np.asarray(eta, dtype=np.float64)
    n = eta_np.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)  # (N,)
    eta_hat = np.fft.fft(eta_np, axis=-1)
    eta_x = np.real(np.fft.ifft(1j * k * eta_hat, axis=-1))
    s = np.sum(eta_x**2, axis=-1)  # (T, B)
    return np.swapaxes(s, 0, 1)  # (B, T)


def envelope_peak(eta: np.ndarray) -> np.ndarray:
    """Per-case ``||η(t)||_∞`` (peak surface elevation).

    Better activity signal for BF / envelope modulation: peak height tracks
    the envelope amplitude directly, so recurrence focusing events produce
    sharp spikes that survive smoothing.

    Args:
        eta: shape (T, B, N).
    Returns:
        ``M[b, t] = max_x |η(t,b,x)|`` of shape (B, T).
    """
    eta_np = np.asarray(eta, dtype=np.float64)
    m = np.max(np.abs(eta_np), axis=-1)  # (T, B)
    return np.swapaxes(m, 0, 1)


def activity_signal_from_energy(grad_energy_BT: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-case relative ``|d/dt|`` of a precomputed signal of shape (B, T).

    Despite the name (kept for back-compat), this works for any per-case
    activity proxy — ``||η_x||^2``, ``||η||_∞``, etc.
    """
    s = np.asarray(grad_energy_BT, dtype=np.float64)
    ds = np.gradient(s, axis=1)
    return np.abs(ds) / (s + eps)


def activity_signal(eta: np.ndarray, length: float, eps: float = 1e-12) -> np.ndarray:
    """Per-case relative ``|d/dt|`` of surface-gradient energy ``||η_x||^2``.

    The Zakharov Hamiltonian is conserved, so this non-conserved quantity is
    used as a state-change diagnostic; it spikes at soliton focusing / BF
    recurrence events.
    """
    return activity_signal_from_energy(surface_gradient_energy(eta, length), eps)


def adaptive_indices_from_signal(
    signal_BT: np.ndarray,
    *,
    keep_samples: int,
    alpha: float = 0.5,
    smooth_sigma_steps: float = 5.0,
    pin_endpoints: bool = True,
    density_mode: str = "rate",
    power: float = 1.0,
) -> np.ndarray:
    """Pick per-case indices given any precomputed activity-proxy trajectory.

    Args:
        density_mode: how to turn ``signal_BT`` into a sampling density.
            ``"rate"`` uses the relative time-derivative ``|dS/dt|/S`` —
            good for localized events (Tanaka collisions). ``"magnitude"``
            uses ``S`` directly — good for continuous modulation where
            you want extra samples at the peaks (BF envelope).
    """
    s = np.asarray(signal_BT)
    if s.ndim != 2:
        raise ValueError(f"signal_BT must be (B, T); got {s.shape}")
    n_batch, n_times = s.shape
    if keep_samples < 2:
        raise ValueError("keep_samples must be >= 2")
    if keep_samples >= n_times:
        return np.broadcast_to(np.arange(n_times, dtype=np.int32), (n_batch, n_times)).copy()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if density_mode == "rate":
        activity = activity_signal_from_energy(s)
    elif density_mode == "magnitude":
        activity = s.copy()
    else:
        raise ValueError(f"density_mode must be 'rate' or 'magnitude'; got {density_mode!r}")
    if power != 1.0:
        # Sharpens contrast — useful when the signal modulates by only ~20-30%.
        # Subtracting the per-case minimum first prevents the baseline from
        # dominating after raising to a power.
        per_case_min = activity.min(axis=1, keepdims=True)
        activity = np.maximum(activity - per_case_min, 0.0) ** power
    return _quantile_indices(activity, keep_samples, alpha, smooth_sigma_steps, pin_endpoints)


def _quantile_indices(
    activity: np.ndarray,
    keep_samples: int,
    alpha: float,
    smooth_sigma_steps: float,
    pin_endpoints: bool,
) -> np.ndarray:
    n_batch, n_times = activity.shape
    out = np.empty((n_batch, keep_samples), dtype=np.int32)
    uniform = np.full(n_times, 1.0 / n_times, dtype=np.float64)
    for b in range(n_batch):
        s = _gaussian_smooth_1d(activity[b], smooth_sigma_steps)
        s_sum = s.sum()
        s = s / s_sum if s_sum > 0 else uniform.copy()
        density = (1.0 - alpha) * uniform + alpha * s
        density /= density.sum()
        cdf = np.cumsum(density)
        q = (np.arange(keep_samples) + 0.5) / keep_samples
        idx = np.searchsorted(cdf, q, side="left").clip(0, n_times - 1)
        if pin_endpoints:
            idx[0] = 0
            idx[-1] = n_times - 1
        for i in range(1, keep_samples):
            if idx[i] <= idx[i - 1]:
                idx[i] = min(idx[i - 1] + 1, n_times - 1)
        out[b] = idx
    return out


# Back-compat alias used by existing call sites; treats the input as a generic
# activity signal regardless of whether it's gradient energy or peak height.
def adaptive_indices_from_energy(*args, **kwargs):
    return adaptive_indices_from_signal(*args, **kwargs)


def adaptive_time_indices(
    eta: np.ndarray,
    *,
    length: float,
    keep_samples: int,
    alpha: float = 0.5,
    smooth_sigma_steps: float = 5.0,
    pin_endpoints: bool = True,
) -> np.ndarray:
    """Choose ``keep_samples`` time indices per case, weighted toward focusing events.

    Args:
        eta: shape (T, B, N) — full eta trajectory on the dense integrator grid.
        length: physical period (for spectral derivative scaling).
        keep_samples: K snapshots per case.
        alpha: 0 = uniform, 1 = activity-only. 0.5 mixes both.
        smooth_sigma_steps: Gaussian smoothing of the activity signal (in
            dense-time steps). Avoids spiky over-concentration at single steps.
        pin_endpoints: include t=0 and t=T-1 in every case's index set.
    Returns:
        int32 array of shape (B, K) with strictly increasing indices per row.
    """
    eta_np = np.asarray(eta)
    if eta_np.ndim != 3:
        raise ValueError(f"eta must be (T, B, N); got {eta_np.shape}")
    return adaptive_indices_from_energy(
        surface_gradient_energy(eta_np, length),
        keep_samples=keep_samples,
        alpha=alpha,
        smooth_sigma_steps=smooth_sigma_steps,
        pin_endpoints=pin_endpoints,
    )


def gather_per_case(traj: np.ndarray, indices_per_case: np.ndarray) -> np.ndarray:
    """Slice (T, B, N) trajectory using per-case time indices (B, K)."""
    traj_bt = np.swapaxes(traj, 0, 1)  # (B, T, N)
    b_idx = np.arange(traj_bt.shape[0])[:, None]  # (B, 1)
    return traj_bt[b_idx, indices_per_case]  # (B, K, N)
