"""Verify our gen_bf matches Philippe's stokes_bf_dataset.npz across all cases.

Each case = a 1001-snapshot simulation. ICs are at idx = c * 1001 for c = 0..35.
For each IC, we infer (n_carr, eps_carrier, n_l, n_r, eps_pert) from the spectrum,
generate our IC with those params on Philippe's L=2π grid, and report rel-L2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from playground.gen_bf import benjamin_feir_ic
from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def _upsample(y: np.ndarray, n_new: int = 1024) -> np.ndarray:
    n_old = y.shape[-1]
    fy = np.fft.rfft(y, axis=-1)
    fpad = np.zeros(y.shape[:-1] + (n_new // 2 + 1,), dtype=fy.dtype)
    fpad[..., : fy.shape[-1]] = fy * (n_new / n_old)
    return np.fft.irfft(fpad, n=n_new, axis=-1)


def _infer_params(eta: np.ndarray) -> dict:
    """Identify (n_carr, a, n_l, n_r, eps_l, eps_r, phase_l, phase_r) from the spectrum."""
    fk_complex = np.fft.rfft(eta)
    fk = np.abs(fk_complex) / (eta.shape[-1] / 2)
    n_carr = int(np.argmax(fk[1:]) + 1)
    a = float(fk[n_carr])

    harmonics = {2 * n_carr, 3 * n_carr, 4 * n_carr, 5 * n_carr}
    candidates = [(int(m), float(fk[m])) for m in range(1, len(fk))
                  if m != n_carr and m not in harmonics]
    candidates.sort(key=lambda t: -t[1])

    if not candidates or candidates[0][1] / a < 1e-3:
        return {"n_carr": n_carr, "a": a, "eps_l": 0.0, "eps_r": 0.0,
                "phase_l": 0.0, "phase_r": 0.0,
                "n_l": n_carr - 1, "n_r": n_carr + 1}

    sb_modes = sorted([m for m, _ in candidates[:2]])
    n_l, n_r = sb_modes[0], sb_modes[1]
    eps_l = float(fk[n_l]) / a
    eps_r = float(fk[n_r]) / a
    # rfft of cos(k*x + phi) at mode k has angle = phi
    phase_l = float(np.angle(fk_complex[n_l]))
    phase_r = float(np.angle(fk_complex[n_r]))
    return {"n_carr": n_carr, "a": a, "eps_l": eps_l, "eps_r": eps_r,
            "phase_l": phase_l, "phase_r": phase_r, "n_l": n_l, "n_r": n_r}


def main() -> None:
    L = 2.0 * np.pi
    Nx = 1024
    DEPTH = 1000.0

    with np.load("data/stokes_bf_dataset.npz") as f:
        eta_all = np.asarray(f["stokes_eta"])
        xi_all = np.asarray(f["stokes_xi"])
        gxi_all = np.asarray(f["stokes_Gxi"])
    n_total = len(eta_all)
    print(f"Loaded {n_total} samples (Nx_old=512, L=2π)")

    x, k = build_grid(Nx, L)

    # IC detection: a t=0 IC has carrier mode at phase 0 (pure cosine, no time evolution),
    # AND a relatively sparse spectrum (carrier + harmonics + ≤2 sidebands).
    fk_all_complex = np.fft.rfft(eta_all, axis=-1)
    fk_all = np.abs(fk_all_complex)
    carrier = np.argmax(fk_all[:, 1:], axis=-1) + 1
    carrier_phase = np.array([np.angle(fk_all_complex[i, carrier[i]]) for i in range(len(eta_all))])
    is_t0 = np.abs(carrier_phase) < 1e-3

    # Sparsity: carrier+harmonics+2 sidebands = ~7 nonzero modes
    significant = fk_all > (fk_all.max(axis=-1, keepdims=True) * 1e-3)
    n_sig = significant.sum(axis=-1)
    is_sparse = n_sig <= 12

    is_ic = is_t0 & is_sparse

    # Suppress consecutive-clean-frames-from-same-sim: keep only those preceded by
    # a non-IC frame OR by a frame with different (carrier mode, |eta|_2) signature
    norms = np.linalg.norm(eta_all, axis=-1)
    keep = is_ic.copy()
    for i in range(1, len(is_ic)):
        if is_ic[i] and is_ic[i-1]:
            same_carrier = carrier[i] == carrier[i-1]
            same_norm = abs(norms[i] - norms[i-1]) / (norms[i-1] + 1e-12) < 1e-3
            if same_carrier and same_norm:
                keep[i] = False
    ic_idx = np.where(keep)[0]
    print(f"\nDetected {len(ic_idx)} t=0 ICs (carrier phase = 0, sparse spectrum, deduplicated):\n")

    rows = []
    n_cases = len(ic_idx)
    print(f"{'case':>4} {'n_carr':>6} {'a':>9} {'eps_c':>7} {'n_l':>4} {'n_r':>4} "
          f"{'eps_l':>7} {'eps_r':>7} {'rL2(eta)':>9} {'rL2(xi)':>9} {'rL2(gxi)':>9}")
    print("-" * 92)

    for c in range(n_cases):
        idx = int(ic_idx[c])
        eta_P = _upsample(eta_all[idx]).astype(np.float32)
        xi_P = _upsample(xi_all[idx]).astype(np.float32)
        gxi_P = _upsample(gxi_all[idx]).astype(np.float32)

        params = _infer_params(eta_P)
        n_carr = params["n_carr"]
        a = params["a"]
        eps_c = n_carr * a  # k * a, in L=2π frame k = n_carr
        eps_l, eps_r = params["eps_l"], params["eps_r"]
        phase_l, phase_r = params["phase_l"], params["phase_r"]
        n_l, n_r = params["n_l"], params["n_r"]

        eta_us, xi_us = benjamin_feir_ic(
            x, n_carr=n_carr, eps_carrier=eps_c,
            n_l=n_l, n_r=n_r,
            eps_pert_l=eps_l, eps_pert_r=eps_r,
            length=L, depth=DEPTH,
            phase_shift=0.0, phase_l_extra=phase_l, phase_r_extra=phase_r,
        )
        gxi_us = dno_series_eval(eta_us, xi_us, k, DEPTH, 6, pad_factor=8)

        eta_us = np.asarray(eta_us, dtype=np.float32)
        xi_us = np.asarray(xi_us, dtype=np.float32)
        gxi_us = np.asarray(gxi_us, dtype=np.float32)

        def rL2(a_, b_):
            return float(np.linalg.norm(a_ - b_) / (np.linalg.norm(b_) + 1e-14))

        rl2_eta = rL2(eta_us, eta_P)
        rl2_xi = rL2(xi_us, xi_P)
        rl2_gxi = rL2(gxi_us, gxi_P)
        rows.append((c, n_carr, a, eps_c, n_l, n_r, eps_l, eps_r, rl2_eta, rl2_xi, rl2_gxi))

        print(f"{c:>4} {n_carr:>6} {a:>9.5f} {eps_c:>7.4f} {n_l:>4} {n_r:>4} "
              f"{eps_l:>7.4f} {eps_r:>7.4f} "
              f"{rl2_eta:>9.5f} {rl2_xi:>9.5f} {rl2_gxi:>9.5f}")

    # Summary
    rl2_eta_arr = np.array([r[8] for r in rows])
    rl2_xi_arr = np.array([r[9] for r in rows])
    rl2_gxi_arr = np.array([r[10] for r in rows])
    print("\n" + "-" * 92)
    print(f"{'mean':>43}  {rl2_eta_arr.mean():>9.5f} {rl2_xi_arr.mean():>9.5f} {rl2_gxi_arr.mean():>9.5f}")
    print(f"{'median':>43}  {np.median(rl2_eta_arr):>9.5f} {np.median(rl2_xi_arr):>9.5f} {np.median(rl2_gxi_arr):>9.5f}")
    print(f"{'max':>43}  {rl2_eta_arr.max():>9.5f} {rl2_xi_arr.max():>9.5f} {rl2_gxi_arr.max():>9.5f}")


if __name__ == "__main__":
    main()
