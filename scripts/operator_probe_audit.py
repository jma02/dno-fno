"""Compare fixed-surface DNO actions on matched and independent potentials.

The four scrambled potentials preserve the original retained Fourier amplitudes.
Four additional potentials add 10% of those probes to the original potential.
Surface elevations are unchanged; potentials and DNO outputs retain modes 1..128.
No training or trajectory integration is performed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver.evals.model_rollout import build_predict_gxi_batched, load_run  # noqa: E402
from solver.solvers.dno_series_jax import dno_series_eval  # noqa: E402

jax.config.update("jax_enable_x64", True)
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
SEED = 20261001
Predictor = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def retained(values: jax.Array) -> jax.Array:
    coefficients = jnp.fft.rfft(values, axis=-1)
    coefficients = coefficients.at[..., 0].set(0).at[..., 129:].set(0)
    return jnp.fft.irfft(coefficients, n=values.shape[-1], axis=-1)


@partial(jax.jit, static_argnames=("order",))
def classical(eta: jax.Array, xi: jax.Array, depth: jax.Array, *, order: int) -> jax.Array:
    return retained(dno_series_eval(
        eta, xi, jnp.arange(513, dtype=jnp.float64), depth[:, None],
        order=order, pad_factor=8,
    ))


def evaluate(
    predict: Predictor, eta: np.ndarray, xi: np.ndarray, depths: np.ndarray,
    batch_size: int, label: str,
) -> np.ndarray:
    started = perf_counter()
    output = np.empty_like(xi, dtype=np.float64)
    for start in range(0, len(xi), batch_size):
        stop = min(start + batch_size, len(xi))
        indices = np.minimum(np.arange(start, start + batch_size), len(xi) - 1)
        values = predict(*(jnp.asarray(field[indices], dtype=jnp.float64)
                           for field in (eta, xi, depths)))
        jax.block_until_ready(values)
        output[start:stop] = np.asarray(values)[:stop - start]
        if start == 0 or stop == len(xi) or stop % (8 * batch_size) == 0:
            print(f"{label}: {stop}/{len(xi)} actions, {perf_counter() - started:.1f}s", flush=True)
    return output


def statistics(values: np.ndarray) -> dict[str, int | float | None]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size), "finite_count": int(finite.size),
        "median_finite": float(np.median(finite)) if finite.size else None,
        "p95_finite": float(np.quantile(finite, 0.95)) if finite.size else None,
    }


def linearity_error(values: np.ndarray) -> np.ndarray:
    probes = values.reshape(-1, 9, 1024)
    expected = probes[:, :1] + 0.1 * probes[:, 1:5]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.linalg.norm(probes[:, 5:9] - expected, axis=-1) / np.linalg.norm(expected, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", choices=("best", "latest", "final"), nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    checkpoints = cast(list[Literal["best", "latest", "final"]],
                       args.checkpoints or ["best"] * len(args.runs))
    if len(checkpoints) != len(args.runs):
        parser.error("--checkpoints must provide one selection per run")
    with np.load(args.cases, allow_pickle=False) as saved:
        eta, xi, depths = (np.asarray(saved[key], dtype=np.float64) for key in ("eta", "xi", "depths"))
        family_ids, simulation_ids = (saved[key] for key in ("family_ids", "simulation_ids"))
    n_cases = len(eta)
    if (eta.shape != (n_cases, 1024) or xi.shape != eta.shape or depths.shape != (n_cases,)
            or family_ids.shape != (n_cases,) or simulation_ids.shape != (n_cases,)
            or not n_cases or not np.isin(family_ids, np.arange(4)).all()):
        parser.error("expected N=1024 cases and family_ids 0..3 (Stokes/Tanaka/BF/JON)")
    if not (np.isfinite(eta).all() and np.isfinite(xi).all() and np.isfinite(depths).all()
            and np.all(depths > 0) and np.all(eta + depths[:, None] > 0)):
        parser.error("cases must be finite with positive depth and water columns")
    coefficients = np.fft.rfft(xi, axis=-1, norm="forward")
    coefficients[:, 0] = 0
    coefficients[:, 129:] = 0
    original = np.fft.irfft(coefficients, n=1024, axis=-1, norm="forward")
    phases = np.random.default_rng(SEED).uniform(0, 2 * np.pi, size=(n_cases, 4, 513))
    scrambled = np.fft.irfft(
        np.abs(coefficients)[:, None, :] * np.exp(1j * phases),
        n=1024, axis=-1, norm="forward",
    )
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(scrambled, axis=-1, norm="forward")),
        np.broadcast_to(np.abs(coefficients)[:, None, :], (n_cases, 4, 513)),
        rtol=1e-10, atol=1e-14,
    )
    np.testing.assert_allclose(
        np.linalg.norm(scrambled, axis=-1),
        np.broadcast_to(np.linalg.norm(original, axis=-1)[:, None], (n_cases, 4)),
        rtol=1e-12, atol=1e-14,
    )
    potentials = np.concatenate((original[:, None, :], scrambled, original[:, None, :] + 0.1 * scrambled), axis=1)
    flat_eta, flat_depths = np.repeat(eta, 9, axis=0), np.repeat(depths, 9)
    flat_xi = potentials.reshape(-1, 1024)
    device_kind = str(jax.devices()[0].device_kind)
    print(f"backend={jax.default_backend()}, device={device_kind}, cases={n_cases}, actions={len(flat_xi)}, seed={SEED}", flush=True)
    reference = evaluate(partial(classical, order=6), flat_eta, flat_xi, flat_depths, args.batch_size, "M6")
    reference_norm = np.linalg.norm(reference, axis=-1)
    if not np.isfinite(reference).all() or np.any(reference_norm == 0):
        raise FloatingPointError("M6 reference must be finite and nonzero on every probe")
    results: dict[str, object] = {
        "cases": str(args.cases), "n_cases": n_cases, "seed": SEED,
        "device_kind": device_kind,
        "reference": "M6, actual depth, N1024, length 2pi, padding8, mean-free modes1..128",
        "neural_precision": "float32 network, float64 physical inputs and output projection",
        "probe_protocol": "original; four same-spectrum random phases; original + 0.1 times each scrambled potential",
        "aggregation": "case-first: median of four probes per case, then family median/p95; original has one action per case",
        "reference_finite_actions": int(np.isfinite(reference).all(axis=-1).sum()),
        "reference_linearity_relative_l2": statistics(linearity_error(reference)),
        "positive_water_column_cases": int(np.all(eta + depths[:, None] > 0, axis=-1).sum()),
        "methods": {},
    }
    methods: dict[str, object] = {}
    raw_errors: list[np.ndarray] = []
    method_names: list[str] = []
    for method_index in range(2 + len(args.runs)):
        started = perf_counter()
        metadata: dict[str, object] = {}
        if method_index < 2:
            order = (2, 8)[method_index]
            name = f"M{order}"
            prediction = evaluate(partial(classical, order=order), flat_eta, flat_xi, flat_depths, args.batch_size, name)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"{name} reference check contains nonfinite values")
        else:
            run_dir, checkpoint = args.runs[method_index - 2], checkpoints[method_index - 2]
            name = run_dir.name
            print(f"Loading {name}, checkpoint={checkpoint}", flush=True)
            loaded = load_run(run_dir, checkpoint=checkpoint)
            predict = build_predict_gxi_batched(loaded, center_xi=False)

            @jax.jit
            def neural(eta: jax.Array, xi: jax.Array, depth: jax.Array) -> jax.Array:
                return retained(predict(eta, xi, jnp.log(depth)))

            prediction = evaluate(neural, flat_eta, flat_xi, flat_depths, args.batch_size, name)
            metadata = {"run_dir": str(run_dir), "checkpoint": checkpoint, "epoch": loaded.epoch, "center_xi": False}
        errors = (np.linalg.norm(prediction - reference, axis=-1) / reference_norm).reshape(n_cases, 9)
        errors[~np.isfinite(errors)] = np.inf
        per_family: dict[str, object] = {}
        for family_id, family in enumerate(FAMILIES):
            selected = errors[family_ids == family_id]
            per_type: dict[str, object] = {}
            compared_to_m2: dict[str, object] = {}
            for label, section in (("original", slice(0, 1)), ("scrambled", slice(1, 5)),
                                   ("perturbed_10pct", slice(5, 9))):
                case_errors = np.median(selected[:, section], axis=1)
                per_type[label] = statistics(case_errors)
                if raw_errors:
                    baseline = np.median(raw_errors[0][family_ids == family_id, section], axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratios = case_errors / baseline
                    compared_to_m2[label] = {
                        "wins": int(np.sum(np.isfinite(case_errors) & (case_errors < baseline))),
                        "count": int(case_errors.size), "ratios": statistics(ratios),
                    }
            with np.errstate(divide="ignore", invalid="ignore"):
                per_type["scrambled_median_over_original"] = statistics(np.median(selected[:, 1:5], axis=1) / selected[:, 0])
                per_type["perturbed_median_over_original"] = statistics(np.median(selected[:, 5:9], axis=1) / selected[:, 0])
            per_family[family] = per_type
            if compared_to_m2:
                per_type["case_median_vs_m2"] = compared_to_m2
        methods[name] = {**metadata, "wall_seconds": perf_counter() - started,
                         "finite_actions": int(np.isfinite(errors).sum()),
                         "all_probes_finite_cases": int(np.isfinite(errors).all(axis=1).sum()),
                         "linearity_relative_l2": statistics(linearity_error(prediction)),
                         "per_family": per_family}
        method_names.append(name)
        raw_errors.append(errors)
        print(f"{name}: {np.isfinite(errors).sum()}/{errors.size} finite errors", flush=True)
    results["methods"] = methods
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), relative_l2=np.stack(raw_errors),
                        method_names=np.asarray(method_names), family_ids=family_ids, simulation_ids=simulation_ids,
                        probe_names=np.asarray(["original", *[f"scrambled_{i}" for i in range(4)],
                                                *[f"perturbed_10pct_{i}" for i in range(4)]]))
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {args.output.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
