"""Run only the paired T=20 portion of the Tanaka tangent validation."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")

import jax  # noqa: E402

from run_tanaka_tangent_stratified_panel import (  # noqa: E402
    BASE_NX,
    DEFAULT_COLLOCATION_POINTS,
    ROOT,
    build_legacy,
    build_tangent,
    make_panel,
    rollout_panel,
    serialize_cases,
    summarize_gates,
)


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "tanaka_tangent_stratified_panel_20260723"
    / "rollout_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmax", type=float, default=20.0)
    parser.add_argument("--output-dt", type=float, default=0.08)
    parser.add_argument("--gxi-chunk-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("This validation must run on CPU.")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("This validation requires JAX float64.")
    if args.tmax <= 0.0 or args.output_dt <= 0.0:
        raise ValueError("--tmax and --output-dt must be positive.")
    if args.gxi_chunk_size <= 0:
        raise ValueError("--gxi-chunk-size must be positive.")

    cases = make_panel()
    start = time.perf_counter()
    print("building N=1024 tangent states", flush=True)
    tangent = build_tangent(
        cases,
        BASE_NX,
        DEFAULT_COLLOCATION_POINTS,
    )
    print("building N=1024 piecewise-linear controls", flush=True)
    legacy = build_legacy(
        cases,
        BASE_NX,
        DEFAULT_COLLOCATION_POINTS,
    )
    print(f"running paired GL2 rollout through T={args.tmax:g}", flush=True)
    rollout, gates = rollout_panel(
        cases=cases,
        tangent=tangent,
        legacy=legacy.fields,
        tmax=args.tmax,
        output_dt=args.output_dt,
        gxi_chunk_size=args.gxi_chunk_size,
    )
    summary = {
        "protocol": {
            "device": str(jax.devices()[0]),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "purpose": "dynamic follow-up to the passed static panel",
            "static_panel": (
                "outputs/tanaka_tangent_stratified_panel_20260723/"
                "static_summary.json"
            ),
        },
        "cases": serialize_cases(cases),
        "rollout": rollout,
        "gates": summarize_gates(gates),
        "total_seconds": time.perf_counter() - start,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": summary["gates"]["passed"],
                "failed": summary["gates"]["failed"],
                "failed_names": summary["gates"]["failed_names"],
                "total_seconds": summary["total_seconds"],
                "rollout_timing_seconds": rollout["timing_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not summary["gates"]["passed"]:
        failed = ", ".join(summary["gates"]["failed_names"])
        raise AssertionError(f"Tanaka tangent dynamic validation failed: {failed}")


if __name__ == "__main__":
    main()
