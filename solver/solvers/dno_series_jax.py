from __future__ import annotations

import jax
import jax.numpy as jnp

DEFAULT_PAD_FACTOR = 8


def build_grid(nx: int, length: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the periodic physical grid and Craig-Sulem Fourier modes."""
    dx = length / nx
    dk = 2.0 * jnp.pi / length
    x = dx * jnp.arange(nx)
    k = dk * jnp.concatenate(
        (
            jnp.arange(0, nx // 2 + 1),
            jnp.arange(1 - nx // 2, 0),
        )
    )
    return x, k


def make_linear_dno_symbol(
    k: jnp.ndarray,
    depth: float | jax.Array,
) -> jnp.ndarray:
    """Return the flat-surface DNO symbol G0(k) = k tanh(h k)."""
    return k * jnp.tanh(depth * k)


def myfft(y: jnp.ndarray, nx: int) -> jnp.ndarray:
    """FFT along the final axis with coefficient ``nx // 2`` set to zero."""
    fy = jnp.fft.fft(y, axis=-1)
    return fy.at[..., nx // 2].set(0)


def myifft(fy: jnp.ndarray) -> jnp.ndarray:
    """Real inverse FFT along the final axis."""
    return jnp.real(jnp.fft.ifft(fy, axis=-1))


def multiply(
    a: jnp.ndarray, b: jnp.ndarray, nx: int, pad_factor: int = DEFAULT_PAD_FACTOR
) -> jnp.ndarray:
    """Multiply in physical space after padded Fourier extension."""
    ny = pad_factor * nx
    fa = myfft(a, nx)
    fb = myfft(b, nx)

    leading_shape = fa.shape[:-1]
    fya = jnp.zeros((*leading_shape, ny), dtype=fa.dtype)
    fyb = jnp.zeros((*leading_shape, ny), dtype=fb.dtype)

    positive_slice = slice(0, nx // 2 + 1)
    negative_slice = slice(ny - nx // 2, ny)

    fya = fya.at[..., positive_slice].set(fa[..., positive_slice])
    fyb = fyb.at[..., positive_slice].set(fb[..., positive_slice])
    fya = fya.at[..., negative_slice].set(fa[..., nx // 2 :])
    fyb = fyb.at[..., negative_slice].set(fb[..., nx // 2 :])

    ya = myifft(fya)
    yb = myifft(fyb)

    fw = myfft(ya * yb, nx)
    fy = jnp.zeros((*leading_shape, nx), dtype=fw.dtype)
    fy = fy.at[..., positive_slice].set(fw[..., positive_slice])
    fy = fy.at[..., nx // 2 :].set(fw[..., ny - nx // 2 :])

    return pad_factor * myifft(fy)


def compute_gm_term(
    m: int,
    etam: list[jnp.ndarray],
    gm_terms: list[jnp.ndarray],
    xi_x: jnp.ndarray,
    k: jnp.ndarray,
    g0: jnp.ndarray,
    nx: int,
    pad_factor: int = DEFAULT_PAD_FACTOR,
) -> jnp.ndarray:
    """Compute the m-th DNO correction term G_m(eta) xi."""
    if m % 2 == 0:
        r = m // 2
        tmp = multiply(etam[m], xi_x, nx, pad_factor=pad_factor)
        gm_term = -myifft(g0 * k ** (2 * (r - 1)) * (1j * k) * myfft(tmp, nx))
        for s in range(r):
            tmp = multiply(
                etam[2 * (r - s)], gm_terms[2 * s], nx, pad_factor=pad_factor
            )
            gm_term = gm_term - myifft(k ** (2 * (r - s)) * myfft(tmp, nx))

            tmp = multiply(
                etam[2 * (r - s) - 1], gm_terms[2 * s + 1], nx, pad_factor=pad_factor
            )
            gm_term = gm_term - myifft(g0 * k ** (2 * (r - s - 1)) * myfft(tmp, nx))
        return gm_term

    r = (m + 1) // 2
    tmp = multiply(etam[m], xi_x, nx, pad_factor=pad_factor)
    gm_term = -myifft(k ** (2 * (r - 1)) * (1j * k) * myfft(tmp, nx))
    for s in range(r - 1):
        tmp = multiply(
            etam[2 * (r - s) - 1], gm_terms[2 * s], nx, pad_factor=pad_factor
        )
        gm_term = gm_term - myifft(g0 * k ** (2 * (r - s - 1)) * myfft(tmp, nx))

        tmp = multiply(
            etam[2 * (r - s - 1)], gm_terms[2 * s + 1], nx, pad_factor=pad_factor
        )
        gm_term = gm_term - myifft(k ** (2 * (r - s - 1)) * myfft(tmp, nx))

    tmp = multiply(etam[1], gm_terms[2 * (r - 1)], nx, pad_factor=pad_factor)
    gm_term = gm_term - myifft(g0 * myfft(tmp, nx))
    return gm_term


def dno_series_eval(
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    k: jnp.ndarray,
    depth: float | jax.Array,
    order: int,
    pad_factor: int = DEFAULT_PAD_FACTOR,
) -> jnp.ndarray:
    """Evaluate the Craig-Sulem DNO series up to the requested order on arrays of shape (..., nx)."""
    eta = jnp.asarray(eta)
    xi = jnp.asarray(xi)
    k = jnp.asarray(k)

    nx = int(eta.shape[-1])
    g0 = make_linear_dno_symbol(k, depth)

    etam: list[jnp.ndarray] = [jnp.ones_like(eta)]
    for m in range(1, order + 1):
        etam.append(multiply(eta, etam[m - 1], nx, pad_factor=pad_factor) / m)

    fxi = myfft(xi, nx)
    xi_x = myifft(1j * k * fxi)

    gm_terms: list[jnp.ndarray] = [myifft(g0 * fxi)]
    for m in range(1, order + 1):
        gm_terms.append(
            compute_gm_term(
                m,
                etam,
                gm_terms,
                xi_x,
                k,
                g0,
                nx,
                pad_factor=pad_factor,
            )
        )

    g = gm_terms[0]
    for gm_term in gm_terms[1:]:
        g = g + gm_term
    return g
