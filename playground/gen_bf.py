"""Generate Benjamin-Feir initial conditions per Xu & Guyenne (JCP 2009), eq. (33).

A Stokes carrier wave (k_carr, steepness eps_c = k_carr * a) is modulated by two
Airy sideband waves at (k_l, k_r) = (k_carr - dk, k_carr + dk) with perturbation
amplitude eps_p:

  eta(x,0) = eta_0(x) + eps_p * a * [cos(k_l x - π/4) + cos(k_r x - π/4)]
  xi(x,0)  = xi_0(x)  + eps_p * a * [exp(k_l*eta)/sqrt(k_l) * sin(k_l x - π/4)
                                    + exp(k_r*eta)/sqrt(k_r) * sin(k_r x - π/4)]

Default parameters reproduce JCP09 §4.2.3: k_carr=9, eps_c=0.13, (k_l,k_r)=(7,11),
eps_p=0.1 on a 2π domain.

Note: We do all computations on our standard L=164, Nx=1024 grid (deep water,
h=1000) so the IC is directly compatible with our FNO and surrogate solver. The
JCP09 case is rescaled accordingly: dimensionless wavenumber k_carr=9 in [0,2π]
becomes mode index n_carr=9 on [0,L] (i.e. physical k_carr = 9 * 2π/L).

Usage:
    uv run python playground/gen_bf.py --out data/bf_ics.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def benjamin_feir_ic(
    x: jnp.ndarray,
    *,
    n_carr: int,
    eps_carrier: float,
    n_l: int,
    n_r: int,
    eps_pert: float | None = None,
    eps_pert_l: float | None = None,
    eps_pert_r: float | None = None,
    length: float,
    depth: float,
    gravity: float = 1.0,
    phase_shift: float = 0.0,
    phase_l_extra: float = 0.0,
    phase_r_extra: float = 0.0,
    bf_2nd_order: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a Benjamin-Feir initial condition (carrier + 2 sideband Airy waves).

    n_carr, n_l, n_r are mode indices on [0, length]. eps_carrier = k_carr * a is
    the carrier steepness, where `a` is the *fundamental* amplitude in the eta
    spectrum (Philippe's convention, not the bare Stokes-Taylor parameter).
    Provide either symmetric eps_pert OR asymmetric (eps_pert_l, eps_pert_r).
    phase_shift adds a common phase to both sidebands; phase_l_extra and
    phase_r_extra add per-sideband phase offsets on top of phase_shift.

    bf_2nd_order: if True, add bound 2nd-order cross modes at (kc+kl), (kc+kr)
    with kernel kc/2 (eta) and ωc/2 = √kc/2 (xi). This matches the structure
    of Philippe's stokes_bf_dataset.npz to <0.1% rel-L2.
    """
    if eps_pert is not None and (eps_pert_l is not None or eps_pert_r is not None):
        raise ValueError("Specify either eps_pert OR (eps_pert_l, eps_pert_r), not both")
    if eps_pert is not None:
        eps_l_v, eps_r_v = eps_pert, eps_pert
    else:
        eps_l_v = eps_pert_l if eps_pert_l is not None else 0.0
        eps_r_v = eps_pert_r if eps_pert_r is not None else 0.0

    k_carr = n_carr * (2.0 * jnp.pi / length)
    k_l = n_l * (2.0 * jnp.pi / length)
    k_r = n_r * (2.0 * jnp.pi / length)
    a = eps_carrier / k_carr  # fundamental amplitude (Philippe's convention)
    a_l = eps_l_v * a
    a_r = eps_r_v * a

    # Renormalize Stokes parameter so that the fundamental coefficient equals `a`
    # exactly. In standard 5th-order Stokes-Taylor with input a0,
    #   eta_fund = a0 * (1 + eps²/8 + 121 eps⁴/192),  eps = k*a0.
    # We invert for a0 so that a0 * F(eps) = a.
    eps_s = eps_carrier
    for _ in range(4):
        F = 1.0 + eps_s**2 / 8.0 + 121.0 * eps_s**4 / 192.0
        a0 = a / F
        eps_s = k_carr * a0
    eps2, eps3, eps4 = eps_s**2, eps_s**3, eps_s**4

    theta_c = k_carr * x
    # 5th-order Stokes carrier (deep water, ichoi=0), normalized to fund amp = a.
    eta_c = a0 * (
        (1.0 + eps2 / 8.0 + 121.0 * eps4 / 192.0) * jnp.cos(theta_c)
        + (0.5 * eps_s + 5.0 * eps3 / 6.0) * jnp.cos(2.0 * theta_c)
        + (3.0 * eps2 / 8.0 + 171.0 * eps4 / 128.0) * jnp.cos(3.0 * theta_c)
        + (eps3 / 3.0) * jnp.cos(4.0 * theta_c)
        + (125.0 * eps4 / 384.0) * jnp.cos(5.0 * theta_c)
    )

    # Linear sideband eta (Airy waves with per-sideband phase).
    phase_l = k_l * x + phase_shift + phase_l_extra
    phase_r = k_r * x + phase_shift + phase_r_extra
    eta_sb = a_l * jnp.cos(phase_l) + a_r * jnp.cos(phase_r)

    # 2nd-order BF cross-mode correction (eta) at sum frequencies (kc+kl), (kc+kr)
    # with kernel kc/2; 3rd-order at (2kc+kl), (2kc+kr) with kernel 3kc²/8.
    # Difference frequencies vanish in deep-water eta. Empirical fits from
    # Philippe's stokes_bf_dataset.
    if bf_2nd_order:
        eta_2nd = (k_carr / 2.0) * a * (
            a_l * jnp.cos(theta_c + phase_l) + a_r * jnp.cos(theta_c + phase_r)
        )
        eta_3rd = (3.0 * k_carr**2 / 8.0) * a * a * (
            a_l * jnp.cos(2.0 * theta_c + phase_l)
            + a_r * jnp.cos(2.0 * theta_c + phase_r)
        )
    else:
        eta_2nd = jnp.zeros_like(x)
        eta_3rd = jnp.zeros_like(x)

    eta_total = eta_c + eta_sb + eta_2nd + eta_3rd

    # xi: carrier Stokes (1st-3rd order) on the FULL surface eta_total inside
    # exp(k*eta); linear sidebands also lifted via eta_total. Sidebands inherit
    # the *carrier* angular frequency ω_c (narrowband NLS assumption) rather
    # than their own ω_l, ω_r — empirically how stokes_bf_dataset.npz is built.
    om_c = jnp.sqrt(gravity * k_carr)
    xi_c = (a0 * om_c / k_carr) * (
        jnp.exp(k_carr * eta_total) * jnp.sin(theta_c)
        + 0.5 * eps3 * jnp.exp(2.0 * k_carr * eta_total) * jnp.sin(2.0 * theta_c)
        + (eps4 / 12.0) * jnp.exp(3.0 * k_carr * eta_total) * jnp.sin(3.0 * theta_c)
    )
    xi_sb = (om_c / k_carr) * (
        a_l * jnp.exp(k_l * eta_total) * jnp.sin(phase_l)
        + a_r * jnp.exp(k_r * eta_total) * jnp.sin(phase_r)
    )

    # 2nd-order BF xi correction. Empirically (from Philippe's stokes_bf_dataset):
    # lower sideband (kl<kc) needs +K_l · a·a_l · cos(theta_c) sin(phase_l), placing
    # +amp at (kc+kl) and −amp at (kc-kl); upper sideband (kr>kc) needs an extra
    # overall sign: −K_r · a·a_r · cos(theta_c) sin(phase_r), placing −amp at
    # both (kc+kr) and (kr-kc). Kernels K = (kc−kl)/√kc and (kr−kc)/√kc.
    if bf_2nd_order:
        K_l = (k_carr - k_l) / jnp.sqrt(gravity * k_carr)
        K_r = (k_r - k_carr) / jnp.sqrt(gravity * k_carr)
        xi_2nd = (
            K_l * a * a_l * jnp.cos(theta_c) * jnp.sin(phase_l)
            - K_r * a * a_r * jnp.cos(theta_c) * jnp.sin(phase_r)
        )
    else:
        xi_2nd = jnp.zeros_like(x)

    return eta_total, xi_c + xi_sb + xi_2nd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BF ICs (JCP09 §4.2.3)")
    parser.add_argument("--out", type=str, default="data/bf_ics.npz")
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1000.0)
    parser.add_argument("--n_carr", type=int, nargs="+",
                        default=[5, 7, 9, 11, 14, 18, 22])
    parser.add_argument("--eps_carrier", type=float, nargs="+",
                        default=[0.08, 0.13, 0.18, 0.25])
    parser.add_argument("--eps_pert", type=float, default=0.1)
    parser.add_argument("--side_offset", type=int, default=2,
                        help="kl=n_carr-offset, kr=n_carr+offset")
    args = parser.parse_args()

    x, k = build_grid(args.nx, args.length)

    eta_list, xi_list, gxi_list, labels = [], [], [], []
    for n_carr in args.n_carr:
        n_l = n_carr - args.side_offset
        n_r = n_carr + args.side_offset
        if n_l <= 0:
            continue
        for eps_c in args.eps_carrier:
            eta, xi = benjamin_feir_ic(
                x,
                n_carr=n_carr, eps_carrier=eps_c,
                n_l=n_l, n_r=n_r, eps_pert=args.eps_pert,
                length=args.length, depth=args.depth,
            )
            gxi = dno_series_eval(eta, xi, k, args.depth, 6, pad_factor=8)

            eta_list.append(np.asarray(eta, dtype=np.float32))
            xi_list.append(np.asarray(xi, dtype=np.float32))
            gxi_list.append(np.asarray(gxi, dtype=np.float32))
            labels.append({"n_carr": n_carr, "n_l": n_l, "n_r": n_r,
                           "eps_carrier": eps_c, "eps_pert": args.eps_pert})
            print(f"  n_carr={n_carr}, eps_c={eps_c:.3f}, max|eta|={float(jnp.max(jnp.abs(eta))):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        eta=np.stack(eta_list),
        xi=np.stack(xi_list),
        gxi=np.stack(gxi_list),
        x=np.asarray(x, dtype=np.float32),
    )
    import json
    (out_path.parent / (out_path.stem + ".labels.json")).write_text(json.dumps(labels, indent=2))
    print(f"\nSaved {len(labels)} BF ICs -> {out_path}")


if __name__ == "__main__":
    main()
