"""Float64 fixed-iteration GL2 rollout, matching solver/evals/model_rollout.py."""

from collections.abc import Callable
import math

import torch
from torch import Tensor

Predictor = Callable[[Tensor, Tensor], Tensor]


def _fft(field: Tensor) -> Tensor:
    spectrum = torch.fft.fft(field)
    spectrum[..., field.shape[-1] // 2] = 0
    return spectrum


def _resample(field: Tensor, n: int) -> Tensor:
    spectrum = torch.fft.rfft(field)[..., :n // 2 + 1].clone()
    spectrum[..., -1] = 0
    return torch.fft.irfft(spectrum, n=n) * (n / field.shape[-1])


@torch.inference_mode()
def rollout(eta: Tensor, xi: Tensor, times: Tensor, depth: Tensor, length: float,
            predict_gxi: Predictor, *, substeps: int = 8, iterations: int = 4,
            filter_fraction: float = .25, gravity: float = 1.) -> dict[str, Tensor]:
    """Roll out (batch,n) fields; return (time,batch,n) eta/xi/gxi on device.

    The predictor consumes physical eta/xi and returns physical G(eta)xi.
    Integration is float64 on CPU/CUDA; predictor precision is independent.
    Nonfinite evolved states are retained so instability remains measurable.
    """
    if eta.device.type not in ("cpu", "cuda"):
        raise ValueError("Float64 rollout requires CPU or CUDA")
    if eta.ndim != 2 or eta.shape != xi.shape or eta.shape[-1] < 4 or eta.shape[-1] % 2:
        raise ValueError("eta/xi must have identical (batch, even n>=4) shapes")
    if length <= 0 or not math.isfinite(length) or gravity <= 0 or not math.isfinite(gravity):
        raise ValueError("length and gravity must be finite and positive")
    if substeps < 1 or iterations < 1 or not 0 < filter_fraction <= 1:
        raise ValueError("Positive substeps/iterations and 0<filter_fraction<=1 required")
    device = eta.device
    eta, xi, times, depth = (v.to(device=device, dtype=torch.float64) for v in (eta, xi, times, depth))
    if times.ndim != 1 or not times.numel() or not torch.isfinite(times).all() or not (times.diff() > 0).all():
        raise ValueError("times must be a finite, nonempty, strictly increasing vector")
    if depth.numel() not in (1, eta.shape[0]) or not torch.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("depth must contain one positive finite value per sample, or a scalar")
    if not torch.isfinite(eta).all() or not torch.isfinite(xi).all():
        raise ValueError("Initial fields must be finite")
    n = eta.shape[-1]
    k = torch.cat((torch.arange(n // 2 + 1, device=device),
                   torch.arange(1 - n // 2, 0, device=device))).double() * (2 * math.pi / length)
    g0 = k * torch.tanh(depth.reshape(-1, 1) * k)
    omega = (gravity * g0).sqrt()
    mask = k.abs() <= filter_fraction * k.abs().max()
    xi = xi - xi.mean(-1, keepdim=True)

    def linear(state: Tensor, tau: Tensor) -> Tensor:
        a, b = state.unbind(0)
        cosine = torch.cos(omega * tau)
        sine = torch.where(omega > 0, torch.sin(omega * tau) / omega.clamp_min(1e-300), 0.)
        return torch.stack((cosine * a + sine * g0 * b, cosine * b - sine * gravity * a))

    def rhs(state: Tensor, time: Tensor) -> Tensor:
        physical = linear(state, time)
        a, b = torch.fft.ifft(physical).real.unbind(0)
        dx = torch.fft.ifft(1j * k * _fft(torch.stack((a, b)))).real
        gxi = predict_gxi(a, b).double()
        eta_t = gxi - torch.fft.ifft(g0 * _fft(b)).real
        a_x, b_x, padded_gxi = _resample(torch.stack((dx[0], dx[1], gxi)), 2 * n).unbind(0)
        xi_t = _resample(-.5 * b_x.square() + .5 * (padded_gxi + a_x * b_x).square()
                         / (1 + a_x.square()), n)
        fields = torch.stack((eta_t, xi_t))
        # Match the reference's physical-space filter followed by myfft.
        if filter_fraction < 1:
            fields = torch.fft.ifft(_fft(fields) * mask).real
        return linear(_fft(fields), -time)

    sqrt3 = math.sqrt(3)
    c1, c2 = .5 - sqrt3 / 6, .5 + sqrt3 / 6
    a12, a21 = .25 - sqrt3 / 6, .25 + sqrt3 / 6
    saved = [torch.stack((eta, xi, predict_gxi(eta, xi).double()))]
    for index in range(times.numel() - 1):
        dt = (times[index + 1] - times[index]) / substeps
        for substep in range(substeps):
            t = times[index] + substep * dt
            v0 = linear(_fft(torch.stack((eta, xi))), -t)
            stage1, stage2 = v0, v0
            for _ in range(iterations):
                f1, f2 = rhs(stage1, t + c1 * dt), rhs(stage2, t + c2 * dt)
                stage1, stage2 = v0 + dt * (.25 * f1 + a12 * f2), v0 + dt * (a21 * f1 + .25 * f2)
            f1, f2 = rhs(stage1, t + c1 * dt), rhs(stage2, t + c2 * dt)
            next_hat = linear(v0 + .5 * dt * (f1 + f2), t + dt)
            if filter_fraction < 1:
                next_hat = next_hat * mask
            eta, xi = torch.fft.ifft(next_hat).real.unbind(0)
            xi = xi - xi.mean(-1, keepdim=True)
        saved.append(torch.stack((eta, xi, predict_gxi(eta, xi).double())))
    trajectory = torch.stack(saved)
    return {"times": times, "eta": trajectory[:, 0], "xi": trajectory[:, 1], "gxi": trajectory[:, 2]}
