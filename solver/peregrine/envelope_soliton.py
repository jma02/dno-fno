"""Generate deep-water NLS envelope soliton snapshots with DNO evaluation.

The envelope soliton of the focusing NLS is

    A(X, T) = A0 * sech(A0 * sqrt(q/(2p)) * X) * exp(i * q * A0^2 * T / 2)

with deep-water coefficients p = omega0 / (8 k0^2), q = omega0 * k0^2 / 2,
and omega0 = sqrt(g * k0).

Lifted to physical variables via the leading-order ansatz:

    eta(x) = 2 * eps * Re[A(X, 0) * exp(i * k0 * x)]
    xi(x)  = -2 * eps * omega0 / k0 * Im[A(X, 0) * exp(i * k0 * x)]

where eps = k0 * a0 is the wave steepness.  The DNO G(eta)xi is evaluated
exactly (to truncation order M) via dno_series_eval at depth = 1000.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..solvers.dno_series_jax import build_grid, dno_series_eval

DEEP_WATER_DEPTH = 1000.0


def _nls_coefficients(
    k0: float, gravity: float = 1.0
) -> tuple[float, float, float, float]:
    """Return (omega0, cg, p, q) for the deep-water NLS."""
    omega0 = float(jnp.sqrt(gravity * k0))
    cg = omega0 / (2.0 * k0)
    p = omega0 / (8.0 * k0**2)
    q = omega0 * k0**2 / 2.0
    return omega0, cg, p, q


def envelope_soliton_snapshot(
    x: jnp.ndarray,
    k: jnp.ndarray,
    k0: float,
    a0: float,
    *,
    T: float = 0.0,
    x_center: float = 0.0,
    gravity: float = 1.0,
    depth: float = DEEP_WATER_DEPTH,
    dno_order: int = 6,
    pad_factor: int = 8,
) -> dict[str, jnp.ndarray | float]:
    """Generate a single envelope soliton snapshot and evaluate the DNO.

    Parameters
    ----------
    x, k : physical and Fourier grids (from build_grid).
    k0 : carrier wavenumber (= n0 * 2pi / L).
    a0 : physical wave amplitude.
    T : slow time (default 0 = peak envelope).
    x_center : center of the envelope in physical coordinates.
    gravity, depth, dno_order, pad_factor : DNO evaluation parameters.

    Returns
    -------
    dict with eta, xi, gxi (arrays of shape (nx,)) and scalar metadata.
    """
    omega0, cg, p, q = _nls_coefficients(k0, gravity)
    eps = k0 * a0
    A0 = a0  # envelope amplitude in the NLS scaling

    # Soliton width parameter
    kappa = A0 * jnp.sqrt(q / (2.0 * p))  # inverse width in X

    # Slow coordinate centered at x_center
    X = eps * (x - x_center)

    # Envelope at slow time T
    phase = 0.5 * q * A0**2 * T
    A = A0 * (1.0 / jnp.cosh(kappa * X)) * jnp.exp(1j * phase)

    # Lift to physical variables
    carrier = jnp.exp(1j * k0 * x)
    eta = 2.0 * eps * jnp.real(A * carrier)
    xi = -2.0 * eps * (omega0 / k0) * jnp.imag(A * carrier)

    # Zero-mean xi (periodic domain)
    xi = xi - jnp.mean(xi)

    # DNO evaluation
    gxi = dno_series_eval(eta, xi, k, depth, dno_order, pad_factor=pad_factor)

    return {
        "eta": eta,
        "xi": xi,
        "gxi": gxi,
        "k0": k0,
        "a0": a0,
        "eps": eps,
        "omega0": omega0,
        "T": T,
        "x_center": x_center,
        "depth": depth,
    }


def envelope_soliton_dataset(
    n0_values: list[int],
    a0_values: list[float],
    *,
    nx: int = 1024,
    length: float = 164.0,
    gravity: float = 1.0,
    depth: float = DEEP_WATER_DEPTH,
    n_centers: int = 5,
    T_values: list[float] | None = None,
    dno_order: int = 6,
    pad_factor: int = 8,
    seed: int = 0,
    steepness_limit: float = 0.10,
) -> dict[str, np.ndarray | list]:
    """Generate a dataset of envelope soliton snapshots.

    Sweeps over (n0, a0, x_center, T), filtering out combos where
    k0 * a0 > steepness_limit (NLS validity breaks down).

    Returns
    -------
    dict with concatenated eta, xi, gxi arrays and metadata.
    """
    if T_values is None:
        T_values = [0.0]

    x, k = build_grid(nx, length)
    rng = np.random.default_rng(seed)

    eta_all: list[np.ndarray] = []
    xi_all: list[np.ndarray] = []
    gxi_all: list[np.ndarray] = []
    labels: list[dict] = []

    for n0 in n0_values:
        k0 = n0 * 2.0 * jnp.pi / length
        k0_f = float(k0)

        for a0 in a0_values:
            eps = k0_f * a0
            if eps > steepness_limit:
                continue

            # Random centers spread across the domain
            centers = rng.uniform(0.0, length, size=n_centers).tolist()

            for x_center in centers:
                for T_val in T_values:
                    snap = envelope_soliton_snapshot(
                        x, k, k0_f, a0,
                        T=T_val,
                        x_center=x_center,
                        gravity=gravity,
                        depth=depth,
                        dno_order=dno_order,
                        pad_factor=pad_factor,
                    )
                    eta_all.append(np.asarray(snap["eta"], dtype=np.float32))
                    xi_all.append(np.asarray(snap["xi"], dtype=np.float32))
                    gxi_all.append(np.asarray(snap["gxi"], dtype=np.float32))
                    labels.append({
                        "n0": n0,
                        "k0": k0_f,
                        "a0": a0,
                        "eps": eps,
                        "T": T_val,
                        "x_center": x_center,
                    })

    return {
        "eta": np.stack(eta_all, axis=0),
        "xi": np.stack(xi_all, axis=0),
        "gxi": np.stack(gxi_all, axis=0),
        "x": np.asarray(x, dtype=np.float32),
        "labels": labels,
        "nx": nx,
        "length": length,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deep-water envelope soliton dataset."
    )
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=DEEP_WATER_DEPTH)
    parser.add_argument("--n0", type=int, nargs="+", default=[4, 8, 14, 20])
    parser.add_argument("--a0", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--n_centers", type=int, default=5)
    parser.add_argument("--T", type=float, nargs="+", default=[0.0])
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--steepness_limit", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Generating envelope soliton dataset...")
    print(f"  n0 = {args.n0}")
    print(f"  a0 = {args.a0}")
    print(f"  n_centers = {args.n_centers}")
    print(f"  T = {args.T}")
    print(f"  steepness_limit = {args.steepness_limit}")

    dataset = envelope_soliton_dataset(
        n0_values=args.n0,
        a0_values=args.a0,
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        n_centers=args.n_centers,
        T_values=args.T,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        seed=args.seed,
        steepness_limit=args.steepness_limit,
    )

    n_samples = dataset["eta"].shape[0]
    print(f"  Generated {n_samples} samples")

    output_dir = Path(args.output) if args.output else Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "envelope_soliton_deep.npz"

    np.savez_compressed(
        output_path,
        eta=dataset["eta"],
        xi=dataset["xi"],
        gxi=dataset["gxi"],
        x=dataset["x"],
    )

    meta = {
        "n_samples": n_samples,
        "nx": args.nx,
        "length": args.length,
        "depth": args.depth,
        "n0_values": args.n0,
        "a0_values": args.a0,
        "n_centers": args.n_centers,
        "T_values": args.T,
        "dno_order": args.dno_order,
        "steepness_limit": args.steepness_limit,
        "seed": args.seed,
        "labels": dataset["labels"],
    }
    meta_path = output_dir / "envelope_soliton_deep.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {output_path}")
    print(f"  Metadata at {meta_path}")


if __name__ == "__main__":
    main()
