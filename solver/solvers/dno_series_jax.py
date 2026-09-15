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


def _dno_series_hat(
    eta_hat: jnp.ndarray,
    xi_hat: jnp.ndarray,
    k: jnp.ndarray,
    g0: jnp.ndarray,
    *,
    nx: int,
    order: int,
    pad_factor: int = DEFAULT_PAD_FACTOR,
) -> jnp.ndarray:
    """Evaluate the DNO series while keeping every recurrence term spectral."""
    padded_nx = pad_factor * nx

    def pad(coefficients: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.irfft(coefficients, n=padded_nx, axis=-1)

    def product_hat(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        coefficients = jnp.fft.rfft(a * b, axis=-1)[..., : nx // 2 + 1]
        return pad_factor * coefficients.at[..., nx // 2].set(0)

    eta_padded = pad(eta_hat)
    eta_powers = [eta_padded, eta_padded]
    for m in range(2, order + 1):
        eta_powers.append(pad(product_hat(eta_padded, eta_powers[m - 1]) / m))

    xi_x = pad(1j * k * xi_hat)
    gm_hats = [g0 * xi_hat]
    gm_padded = [pad(gm_hats[0])]
    for m in range(1, order + 1):
        if m % 2 == 0:
            r = m // 2
            products = [(eta_powers[m], xi_x)]
            symbols = [g0 * k ** (2 * (r - 1)) * (1j * k)]
            for s in range(r):
                products.extend(
                    (
                        (eta_powers[2 * (r - s)], gm_padded[2 * s]),
                        (eta_powers[2 * (r - s) - 1], gm_padded[2 * s + 1]),
                    )
                )
                symbols.extend((k ** (2 * (r - s)), g0 * k ** (2 * (r - s - 1))))
        else:
            r = (m + 1) // 2
            products = [(eta_powers[m], xi_x)]
            symbols = [k ** (2 * (r - 1)) * (1j * k)]
            for s in range(r - 1):
                products.extend(
                    (
                        (eta_powers[2 * (r - s) - 1], gm_padded[2 * s]),
                        (eta_powers[2 * (r - s - 1)], gm_padded[2 * s + 1]),
                    )
                )
                symbols.extend((g0 * k ** (2 * (r - s - 1)), k ** (2 * (r - s - 1))))
            products.append((eta_powers[1], gm_padded[2 * (r - 1)]))
            symbols.append(g0)

        gm_hat = -sum(
            (
                symbol * product_hat(*product)
                for symbol, product in zip(symbols, products, strict=True)
            ),
            start=jnp.zeros_like(gm_hats[0]),
        )
        gm_hats.append(gm_hat)
        if m < order:
            gm_padded.append(pad(gm_hat))
    return sum(gm_hats, start=jnp.zeros_like(gm_hats[0]))


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
    k = k[: nx // 2 + 1]
    eta_hat = jnp.fft.rfft(eta, axis=-1).at[..., nx // 2].set(0)
    xi_hat = jnp.fft.rfft(xi, axis=-1).at[..., nx // 2].set(0)
    gxi_hat = _dno_series_hat(
        eta_hat,
        xi_hat,
        k,
        make_linear_dno_symbol(k, depth),
        nx=nx,
        order=order,
        pad_factor=pad_factor,
    )
    return jnp.fft.irfft(gxi_hat, n=nx, axis=-1)
