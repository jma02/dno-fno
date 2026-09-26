"""CPU float64 parity against the original JAX surrogate rollout."""

import json
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "torch-attention"), str(ROOT)]
from model import baseline  # noqa: E402
from rollout import rollout  # noqa: E402
from solver.evals.model_rollout import rollout_surrogate  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402

jax.config.update("jax_enable_x64", True)
torch.set_num_threads(1)
rng = np.random.default_rng(42)
rows = []
started = perf_counter()
for n in (16, 32, 64):
    length = 7.3
    x = np.arange(n) * (2 * np.pi / n)
    eta = .003 * np.cos(x)[None] + rng.normal(0., .0001, (2, n))
    xi = .002 * np.sin(x)[None] + rng.normal(0., .0001, (2, n)) + .7
    depth = np.array([.4, 2.1])[:, None]
    times = np.array([.37, .373, .381, .389])
    td = torch.from_numpy(depth)
    k = jnp.arange(n // 2 + 1) * (2 * np.pi / length)
    symbol = k * jnp.tanh(jnp.asarray(depth) * k)

    def predict_jax(a: jax.Array, b: jax.Array) -> jax.Array:
        spectrum = jnp.fft.rfft(b)
        g0 = jnp.fft.irfft(symbol * spectrum, n=n)
        dx = jnp.fft.irfft(1j * k * spectrum, n=n)
        g1 = jnp.fft.irfft(-symbol * jnp.fft.rfft(a * g0) - 1j * k * jnp.fft.rfft(a * dx), n=n)
        result = g0 + g1
        return result - result.mean(-1, keepdims=True)

    calls = 0

    def predict_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        global calls
        calls += 1
        result = baseline(a, b, td, length)
        return result - result.mean(-1, keepdim=True)

    for fraction, substeps in ((1., 1), (2 / 3, 2), (.25, 8)):
        params = ti.make_solver_params(n, length, jnp.asarray(depth), filter_fraction=fraction)
        reference = rollout_surrogate(ti.State(jnp.asarray(eta), jnp.asarray(xi)),
                                      jnp.asarray(times), params, predict_jax, substeps=substeps)
        calls = 0
        actual = rollout(torch.from_numpy(eta), torch.from_numpy(xi), torch.from_numpy(times),
                         td, length, predict_torch, filter_fraction=fraction, substeps=substeps)
        errors = {}
        for field in ("eta", "xi", "gxi"):
            a, b = actual[field].numpy(), np.asarray(reference[field])
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-12)
            errors[field] = float(np.max(np.abs(a - b)))
        assert calls == 1 + (len(times) - 1) * (substeps * 10 + 1)
        assert actual["xi"].mean(-1).abs().max() < 1e-15
        one = rollout(torch.from_numpy(eta), torch.from_numpy(xi), torch.from_numpy(times[:1]),
                      td, length, predict_torch)
        torch.testing.assert_close(one["eta"][0], torch.from_numpy(eta), rtol=0, atol=0)
        torch.testing.assert_close(one["xi"][0], torch.from_numpy(xi - xi.mean(-1, keepdims=True)), rtol=0, atol=2e-16)
        rows.append({"n": n, "filter_fraction": fraction, "substeps": substeps, "max_abs_difference": errors})
        print(rows[-1], flush=True)

# The zero DNO residual gives an exact linear flow when nonlinear xi terms vanish.
zero = torch.zeros(2, 16, dtype=torch.float64)
rest = rollout(zero, zero + 3, torch.tensor([0., .1, .3]), torch.tensor([.5, 1.]),
               2 * np.pi, lambda a, b: torch.zeros_like(a))
assert all(torch.count_nonzero(rest[k]) == 0 for k in ("eta", "xi", "gxi"))
report = {"cases": rows, "seconds": perf_counter() - started,
          "rest_state_and_zero_mean_passed": True, "single_time_passed": True,
          "operator": "identical float64 analytic G0+G1 in both frameworks; no learned-model accuracy claim"}
Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
