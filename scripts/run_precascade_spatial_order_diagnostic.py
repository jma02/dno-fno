"""Diagnose a late spectral cascade by controlled spatial/order restarts.

This script is intentionally separate from the frozen full-horizon panel.  It
loads the same finite state from the completed ``dt=0.01`` arm for every
comparison, then changes exactly one of

* the base Fourier grid size,
* the padding factor used inside Craig--Sulem products,
* the retained Fourier cutoff, or
* the Craig--Sulem truncation order.

The default production matrix is

    baseline       N=1024, pad=8,  K=128, M=6
    grid_2048      N=2048, pad=8,  K=128, M=6
    padding_16     N=1024, pad=16, K=128, M=6
    cutoff_192     N=1024, pad=8,  K=192, M=6
    order_5        N=1024, pad=8,  K=128, M=5
    order_4        N=1024, pad=8,  K=128, M=4

No adaptive limiter or empirical filter is added.  Each arm retains the same
hard Fourier projection used by the full-horizon calculation, with its cutoff
set to the declared ``K``.

The dynamic restart is a local diagnosis, not a replacement for spatial
convergence from the original initial condition.  A promising arm must
subsequently be rerun over the full horizon.

Production use (GPU required unless explicitly overridden):

    uv run python scripts/run_precascade_spatial_order_diagnostic.py

The cheaper static Craig--Sulem order precursor can be run by itself:

    uv run python scripts/run_precascade_spatial_order_diagnostic.py \
      --static-only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    rollout,
)

jax.config.update("jax_enable_x64", True)

FloatArray: TypeAlias = NDArray[np.float64]
ArrayDict: TypeAlias = dict[str, NDArray[Any]]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ARM = (
    ROOT
    / "outputs/full_horizon_refinement_panel_20260725"
    / "tanaka_benjamin_feir_dt_0p010.npz"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/precascade_spatial_order_diagnostic_20260725"
)


@dataclass(frozen=True)
class ArmDefinition:
    """One dynamic arm in the one-factor comparison."""

    arm_id: str
    nx: int
    pad_factor: int
    delivered_wavenumber: float
    dno_order: int
    changed_factor: str


@dataclass(frozen=True)
class CaseProbe:
    """One exact source state and its short post-restart horizon."""

    case_id: str
    restart_time: float
    terminal_time: float
    static_times: tuple[float, ...]


@dataclass(frozen=True)
class DiagnosticContract:
    """Numerical choices shared by all dynamic arms."""

    length: float = 2.0 * math.pi
    gravity: float = 1.0
    dt: float = 0.005
    saved_dt: float = 0.08
    gl2_residual_tolerance: float = 1.0e-8
    gl2_iteration_cap: int = 8
    common_wavenumber: float = 128.0
    high_band_lower_wavenumber: float = 80.0
    dtype: str = "float64"


@dataclass(frozen=True)
class SourceCase:
    """Source trajectory data needed by one case probe."""

    case_id: str
    depth: float
    source_times: FloatArray
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    stage_converged: NDArray[np.bool_]
    step_times: FloatArray


def production_arm_definitions() -> tuple[ArmDefinition, ...]:
    """Return the minimal one-factor dynamic matrix."""

    return (
        ArmDefinition("baseline", 1024, 8, 128.0, 6, "none"),
        ArmDefinition("grid_2048", 2048, 8, 128.0, 6, "nx"),
        ArmDefinition("padding_16", 1024, 16, 128.0, 6, "pad_factor"),
        ArmDefinition(
            "cutoff_192", 1024, 8, 192.0, 6, "delivered_wavenumber"
        ),
        ArmDefinition("order_5", 1024, 8, 128.0, 5, "dno_order"),
        ArmDefinition("order_4", 1024, 8, 128.0, 4, "dno_order"),
    )


def production_case_probes() -> tuple[CaseProbe, ...]:
    """Return saved states immediately before clear high-band growth."""

    return (
        CaseProbe(
            case_id="tanaka_steep_upper_seam",
            restart_time=80.0,
            terminal_time=122.0,
            static_times=(0.0, 20.0, 40.0, 80.0),
        ),
        CaseProbe(
            case_id="bf_jcp09_canonical",
            restart_time=88.0,
            terminal_time=112.0,
            static_times=(0.0, 20.0, 40.0, 56.0, 88.0),
        ),
    )


def changed_arm_fields(
    baseline: ArmDefinition,
    candidate: ArmDefinition,
) -> tuple[str, ...]:
    """Return numerical fields in which ``candidate`` differs."""

    fields = ("nx", "pad_factor", "delivered_wavenumber", "dno_order")
    return tuple(
        field
        for field in fields
        if getattr(candidate, field) != getattr(baseline, field)
    )


def validate_arm_matrix(arms: tuple[ArmDefinition, ...]) -> None:
    """Require a unique baseline and one changed factor in every other arm."""

    if not arms or arms[0].arm_id != "baseline":
        raise ValueError("the first arm must be named 'baseline'")
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("arm identifiers must be unique")
    baseline = arms[0]
    for arm in arms:
        if arm.nx <= 0 or arm.nx % 2:
            raise ValueError(f"{arm.arm_id}: nx must be positive and even")
        if arm.pad_factor < 1:
            raise ValueError(f"{arm.arm_id}: pad_factor must be positive")
        if not 0.0 < arm.delivered_wavenumber < arm.nx / 2:
            raise ValueError(
                f"{arm.arm_id}: K must lie strictly below the Nyquist mode"
            )
        if arm.dno_order < 0:
            raise ValueError(f"{arm.arm_id}: dno_order must be nonnegative")
        differences = changed_arm_fields(baseline, arm)
        if arm is baseline:
            if differences or arm.changed_factor != "none":
                raise ValueError("baseline must not change a factor")
        elif differences != (arm.changed_factor,):
            raise ValueError(
                f"{arm.arm_id}: expected only {arm.changed_factor!r} to "
                f"change, found {differences}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _time_index(times: FloatArray, target: float) -> int:
    matches = np.flatnonzero(
        np.isclose(times, target, rtol=0.0, atol=2.0e-12)
    )
    if matches.size != 1:
        raise ValueError(
            f"source archive does not contain the unique saved time {target}"
        )
    return int(matches[0])


def load_source_cases(
    source_path: Path,
    probes: tuple[CaseProbe, ...],
) -> tuple[dict[str, SourceCase], dict[str, Any]]:
    """Load only the declared cases and validate their finite restart history."""

    with np.load(source_path, allow_pickle=False) as archive:
        identifiers = tuple(str(value) for value in archive["case_id"])
        missing = [
            probe.case_id
            for probe in probes
            if probe.case_id not in identifiers
        ]
        if missing:
            raise ValueError(f"{missing!r} are absent from {source_path}")
        selected_indices = np.asarray(
            [identifiers.index(probe.case_id) for probe in probes],
            dtype=np.int64,
        )
        source_times = np.asarray(archive["times"], dtype=np.float64)
        source_eta = np.asarray(
            archive["eta"][:, selected_indices], dtype=np.float64
        )
        source_xi = np.asarray(
            archive["xi"][:, selected_indices], dtype=np.float64
        )
        source_gxi = np.asarray(
            archive["gxi"][:, selected_indices], dtype=np.float64
        )
        source_depth = np.asarray(
            archive["depth"][selected_indices], dtype=np.float64
        )
        stage_converged = np.asarray(
            archive["gl2_converged"][:, selected_indices], dtype=np.bool_
        )
        step_times = np.asarray(archive["gl2_step_times"], dtype=np.float64)
        schema = str(np.asarray(archive["schema"]).item())
        source_fingerprint = str(
            np.asarray(archive["config_fingerprint"]).item()
        )
        source_dt = float(np.asarray(archive["dt"]).item())

    cases: dict[str, SourceCase] = {}
    for selected_index, probe in enumerate(probes):
        restart_index = _time_index(source_times, probe.restart_time)
        for static_time in probe.static_times:
            _time_index(source_times, static_time)
        fields = (
            source_eta[: restart_index + 1, selected_index],
            source_xi[: restart_index + 1, selected_index],
            source_gxi[: restart_index + 1, selected_index],
        )
        if not all(np.isfinite(field).all() for field in fields):
            raise ValueError(
                f"{probe.case_id}: source is nonfinite through restart"
            )
        completed_steps = step_times < probe.restart_time - 1.0e-13
        case_stages = stage_converged[:, selected_index]
        if not np.all(case_stages[completed_steps]):
            raise ValueError(
                f"{probe.case_id}: source has an unsolved stage before restart"
            )
        cases[probe.case_id] = SourceCase(
            case_id=probe.case_id,
            depth=float(source_depth[selected_index]),
            source_times=source_times,
            eta=source_eta[:, selected_index],
            xi=source_xi[:, selected_index],
            gxi=source_gxi[:, selected_index],
            stage_converged=case_stages,
            step_times=step_times,
        )
    return cases, {
        "path": str(source_path.resolve()),
        "sha256": _sha256(source_path),
        "schema": schema,
        "config_fingerprint": source_fingerprint,
        "dt": source_dt,
    }


def resize_project_periodic_real(
    field: FloatArray,
    *,
    output_nx: int,
    length: float,
    maximum_wavenumber: float,
) -> FloatArray:
    """Project a real periodic array and evaluate it on ``output_nx`` points."""

    values = np.asarray(field, dtype=np.float64)
    input_nx = values.shape[-1]
    if input_nx % 2 or output_nx % 2:
        raise ValueError("input and output grid sizes must be even")
    mode_scale = 2.0 * math.pi / length
    maximum_mode = int(
        math.floor(maximum_wavenumber / mode_scale + 1.0e-12)
    )
    maximum_available = min(input_nx // 2 - 1, output_nx // 2 - 1)
    keep = min(maximum_mode, maximum_available)
    source = np.fft.rfft(values, axis=-1) / input_nx
    target = np.zeros(
        (*values.shape[:-1], output_nx // 2 + 1),
        dtype=np.complex128,
    )
    target[..., : keep + 1] = source[..., : keep + 1]
    return np.asarray(
        np.fft.irfft(target * output_nx, n=output_nx, axis=-1),
        dtype=np.float64,
    )


def _saved_times(probe: CaseProbe, contract: DiagnosticContract) -> FloatArray:
    duration = probe.terminal_time - probe.restart_time
    count = int(round(duration / contract.saved_dt))
    if not math.isclose(
        count * contract.saved_dt,
        duration,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        raise ValueError("the probe horizon must lie on the saved-time grid")
    return probe.restart_time + contract.saved_dt * np.arange(
        count + 1, dtype=np.float64
    )


def _substeps(contract: DiagnosticContract) -> int:
    count = int(round(contract.saved_dt / contract.dt))
    if not math.isclose(
        count * contract.dt,
        contract.saved_dt,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("saved_dt must be an integer multiple of dt")
    return count


def _filter_fraction(arm: ArmDefinition, contract: DiagnosticContract) -> float:
    del contract
    return 2.0 * arm.delivered_wavenumber / arm.nx


def _evaluate_saved_gxi(
    eta: jax.Array,
    xi: jax.Array,
    *,
    depth: float,
    arm: ArmDefinition,
    contract: DiagnosticContract,
) -> FloatArray:
    _, wavenumbers = build_grid(arm.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    result = np.empty(eta.shape, dtype=np.float64)
    chunk_size = 4
    for start in range(0, eta.shape[0], chunk_size):
        stop = min(start + chunk_size, eta.shape[0])
        value = dno_series_eval(
            eta[start:stop],
            xi[start:stop],
            k,
            jnp.asarray([[depth]], dtype=jnp.float64),
            arm.dno_order,
            pad_factor=arm.pad_factor,
        )
        value = apply_lowpass(value, k, _filter_fraction(arm, contract))
        value -= jnp.mean(value, axis=-1, keepdims=True)
        result[start:stop] = np.asarray(
            jax.device_get(value), dtype=np.float64
        )
    return result


def run_dynamic_arm(
    source: SourceCase,
    probe: CaseProbe,
    arm: ArmDefinition,
    contract: DiagnosticContract,
) -> tuple[ArrayDict, dict[str, float]]:
    """Run one arm from the exact declared source snapshot."""

    source_index = _time_index(source.source_times, probe.restart_time)
    eta0 = resize_project_periodic_real(
        source.eta[source_index],
        output_nx=arm.nx,
        length=contract.length,
        maximum_wavenumber=contract.common_wavenumber,
    )
    xi0 = resize_project_periodic_real(
        source.xi[source_index],
        output_nx=arm.nx,
        length=contract.length,
        maximum_wavenumber=contract.common_wavenumber,
    )
    xi0 -= np.mean(xi0)
    times = _saved_times(probe, contract)
    _, wavenumbers = build_grid(arm.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depth = jnp.asarray([[source.depth]], dtype=jnp.float64)
    parameters = SolverParams(
        nx=arm.nx,
        length=contract.length,
        depth=depth,
        gravity=contract.gravity,
        dno_order=arm.dno_order,
        pad_factor=arm.pad_factor,
        filter_fraction=_filter_fraction(arm, contract),
        k=k,
        g0=make_linear_dno_symbol(k, depth),
    )

    start = perf_counter()
    payload = rollout(
        State(
            eta=jnp.asarray(eta0[None, :], dtype=jnp.float64),
            xi=jnp.asarray(xi0[None, :], dtype=jnp.float64),
        ),
        jnp.asarray(times, dtype=jnp.float64),
        parameters,
        save_gxi=False,
        substeps_per_interval=_substeps(contract),
        method="gl2_if",
        implicit_iterations=contract.gl2_iteration_cap,
        implicit_residual_tolerance=contract.gl2_residual_tolerance,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )
    jax.block_until_ready(payload["xi"])
    rollout_seconds = perf_counter() - start

    start = perf_counter()
    gxi = _evaluate_saved_gxi(
        payload["eta"],
        payload["xi"],
        depth=source.depth,
        arm=arm,
        contract=contract,
    )
    gxi_seconds = perf_counter() - start

    start = perf_counter()
    arrays: ArrayDict = {
        name: np.asarray(jax.device_get(value))
        for name, value in payload.items()
    }
    arrays["eta"] = arrays["eta"][:, 0]
    arrays["xi"] = arrays["xi"][:, 0]
    arrays["gxi"] = gxi[:, 0]
    for name in (
        "gl2_stage_residual",
        "gl2_iterations",
        "gl2_converged",
        "gl2_stage_finite",
        "gl2_state_finite",
        "gl2_hit_iteration_cap",
    ):
        arrays[name] = arrays[name][:, 0]
    for name in (
        "gl2_all_stages_converged",
        "gl2_max_stage_residual",
        "gl2_first_failed_step",
    ):
        arrays[name] = np.asarray(arrays[name][0])
    transfer_seconds = perf_counter() - start
    del payload
    gc.collect()
    return arrays, {
        "rollout_including_compilation": rollout_seconds,
        "saved_gxi_including_compilation": gxi_seconds,
        "remaining_host_transfer": transfer_seconds,
        "compute_total": (
            rollout_seconds + gxi_seconds + transfer_seconds
        ),
    }


def _artifact_path(
    output_dir: Path,
    probe: CaseProbe,
    arm: ArmDefinition,
) -> Path:
    return output_dir / f"{probe.case_id}__{arm.arm_id}.npz"


def _arm_fingerprint(
    *,
    source_record: dict[str, Any],
    probe: CaseProbe,
    arm: ArmDefinition,
    contract: DiagnosticContract,
    script_sha256: str,
) -> str:
    payload = {
        "source": source_record,
        "probe": asdict(probe),
        "arm": asdict(arm),
        "contract": asdict(contract),
        "script_sha256": script_sha256,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def save_dynamic_arm(
    path: Path,
    arrays: ArrayDict,
    *,
    source_record: dict[str, Any],
    probe: CaseProbe,
    arm: ArmDefinition,
    contract: DiagnosticContract,
    script_sha256: str,
    timings: dict[str, float],
) -> None:
    """Write one restartable dynamic artifact."""

    fingerprint = _arm_fingerprint(
        source_record=source_record,
        probe=probe,
        arm=arm,
        contract=contract,
        script_sha256=script_sha256,
    )
    metadata = {
        "schema": "precascade_spatial_order_dynamic_arm_v1",
        "config_fingerprint": fingerprint,
        "source": source_record,
        "probe": asdict(probe),
        "arm": asdict(arm),
        "contract": asdict(contract),
        "timing_seconds": timings,
    }
    _write_npz_atomic(
        path,
        {
            **arrays,
            "schema": np.asarray(metadata["schema"]),
            "config_fingerprint": np.asarray(fingerprint),
            "metadata_json": np.asarray(_canonical_json(metadata)),
            "case_id": np.asarray(probe.case_id),
        },
    )


def load_dynamic_arm(
    path: Path,
    *,
    source_record: dict[str, Any],
    probe: CaseProbe,
    arm: ArmDefinition,
    contract: DiagnosticContract,
    script_sha256: str,
) -> ArrayDict:
    """Load a matching arm or refuse an ambiguous resume."""

    expected = _arm_fingerprint(
        source_record=source_record,
        probe=probe,
        arm=arm,
        contract=contract,
        script_sha256=script_sha256,
    )
    with np.load(path, allow_pickle=False) as archive:
        stored = str(np.asarray(archive["config_fingerprint"]).item())
        if stored != expected:
            raise RuntimeError(
                f"{path} belongs to another configuration; use --overwrite"
            )
        excluded = {
            "schema",
            "config_fingerprint",
            "metadata_json",
            "case_id",
        }
        return {
            name: np.asarray(archive[name])
            for name in archive.files
            if name not in excluded
        }


def _rms(field: FloatArray) -> FloatArray:
    return np.sqrt(np.mean(np.square(field), axis=-1))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _first_false_time(
    condition: NDArray[np.bool_],
    step_times: FloatArray,
) -> float | None:
    indices = np.flatnonzero(~condition)
    return float(step_times[int(indices[0])]) if indices.size else None


def _band_energy(
    field: FloatArray,
    *,
    length: float,
    lower_wavenumber: float,
    upper_wavenumber: float,
) -> FloatArray:
    values = np.asarray(field, dtype=np.float64)
    coefficients = np.fft.rfft(values, axis=-1) / values.shape[-1]
    wavenumbers = (
        2.0
        * math.pi
        * np.arange(coefficients.shape[-1], dtype=np.float64)
        / length
    )
    mask = (wavenumbers >= lower_wavenumber) & (
        wavenumbers <= upper_wavenumber
    )
    return np.sum(np.abs(coefficients[..., mask]) ** 2, axis=-1)


def _growth_threshold_times(
    growth: FloatArray,
    times: FloatArray,
) -> dict[str, float | None]:
    thresholds = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
    result: dict[str, float | None] = {}
    for threshold in thresholds:
        indices = np.flatnonzero(growth >= threshold)
        result[f"{threshold:.0e}"] = (
            float(times[int(indices[0])]) if indices.size else None
        )
    return result


def summarize_source_restart(
    source: SourceCase,
    probe: CaseProbe,
    contract: DiagnosticContract,
) -> dict[str, Any]:
    """Verify that the chosen source snapshot precedes clear tail growth."""

    restart_index = _time_index(source.source_times, probe.restart_time)
    times = source.source_times[: restart_index + 1]
    gxi = source.gxi[: restart_index + 1]
    high_energy = _band_energy(
        gxi,
        length=contract.length,
        lower_wavenumber=contract.high_band_lower_wavenumber,
        upper_wavenumber=contract.common_wavenumber,
    )
    total_initial_energy = _band_energy(
        gxi[:1],
        length=contract.length,
        lower_wavenumber=2.0 * math.pi / contract.length,
        upper_wavenumber=contract.common_wavenumber,
    )[0]
    growth = np.maximum(high_energy - high_energy[0], 0.0) / max(
        float(total_initial_energy),
        np.finfo(np.float64).tiny,
    )
    restart_growth = float(growth[-1])
    return {
        "definition": (
            "positive growth of q energy in 80<=|k|<=128, divided by "
            "the total nonzero-mode q energy at source time zero"
        ),
        "normalized_growth_at_restart": restart_growth,
        "below_predeclared_1e-4_guard": restart_growth < 1.0e-4,
        "growth_threshold_times_through_restart": _growth_threshold_times(
            growth, times
        ),
    }


def summarize_dynamic_arm(
    arrays: ArrayDict,
    *,
    arm: ArmDefinition,
    contract: DiagnosticContract,
) -> dict[str, Any]:
    """Summarize finiteness, GL2 convergence, and spectral-tail growth."""

    times = np.asarray(arrays["times"], dtype=np.float64)
    step_times = np.asarray(arrays["gl2_step_times"], dtype=np.float64)
    converged = np.asarray(arrays["gl2_converged"], dtype=np.bool_)
    finite_fields = all(
        np.isfinite(np.asarray(arrays[name])).all()
        for name in ("eta", "xi", "gxi")
    )
    high_energy = _band_energy(
        np.asarray(arrays["gxi"], dtype=np.float64),
        length=contract.length,
        lower_wavenumber=contract.high_band_lower_wavenumber,
        upper_wavenumber=contract.common_wavenumber,
    )
    total_initial_energy = _band_energy(
        np.asarray(arrays["gxi"][:1], dtype=np.float64),
        length=contract.length,
        lower_wavenumber=2.0 * math.pi / contract.length,
        upper_wavenumber=contract.common_wavenumber,
    )[0]
    growth = np.maximum(high_energy - high_energy[0], 0.0) / max(
        float(total_initial_energy),
        np.finfo(np.float64).tiny,
    )
    extra_band_energy = _band_energy(
        np.asarray(arrays["gxi"], dtype=np.float64),
        length=contract.length,
        lower_wavenumber=contract.common_wavenumber
        + 2.0 * math.pi / contract.length,
        upper_wavenumber=arm.delivered_wavenumber,
    )
    return {
        "all_fields_finite": finite_fields,
        "all_gl2_stages_converged": bool(np.all(converged)),
        "first_failed_stage_time": _first_false_time(converged, step_times),
        "maximum_finite_stage_residual": _finite_or_none(
            float(
                np.max(
                    np.asarray(arrays["gl2_stage_residual"])[
                        np.isfinite(arrays["gl2_stage_residual"])
                    ]
                )
            )
            if np.any(np.isfinite(arrays["gl2_stage_residual"]))
            else float("inf")
        ),
        "maximum_normalized_q_high_band_growth": _finite_or_none(
            float(np.max(growth))
        ),
        "q_high_band_growth_threshold_times": _growth_threshold_times(
            growth, times
        ),
        "maximum_q_energy_above_common_band": _finite_or_none(
            float(np.max(extra_band_energy))
        ),
    }


def project_dynamic_to_common_grid(
    arrays: ArrayDict,
    contract: DiagnosticContract,
) -> dict[str, FloatArray]:
    """Put one arm on the baseline grid and common delivered band."""

    return {
        name: resize_project_periodic_real(
            np.asarray(arrays[name], dtype=np.float64),
            output_nx=1024,
            length=contract.length,
            maximum_wavenumber=contract.common_wavenumber,
        )
        for name in ("eta", "xi", "gxi")
    }


def compare_with_baseline(
    baseline: ArrayDict,
    candidate: ArrayDict,
    contract: DiagnosticContract,
) -> dict[str, Any]:
    """Report raw common-band trajectory sensitivity to one changed factor."""

    if not np.array_equal(baseline["times"], candidate["times"]):
        raise ValueError("candidate and baseline do not share saved times")
    reference = project_dynamic_to_common_grid(baseline, contract)
    comparison = project_dynamic_to_common_grid(candidate, contract)
    fields: dict[str, Any] = {}
    for name in ("eta", "xi", "gxi"):
        numerator = _rms(comparison[name] - reference[name])
        denominator = np.maximum(
            np.maximum(_rms(comparison[name]), _rms(reference[name])),
            1.0e-14,
        )
        relative = numerator / denominator
        fields[name] = {
            "maximum_relative_common_band_difference": _finite_or_none(
                float(np.max(relative))
            ),
            "terminal_relative_common_band_difference": _finite_or_none(
                float(relative[-1])
            ),
        }
    return {
        "definition": (
            "RMS(P128(candidate-reference)) divided at each saved time by "
            "the larger RMS of the two P128 fields; no phase alignment"
        ),
        "fields": fields,
    }


def evaluate_static_order_precursor(
    source_cases: dict[str, SourceCase],
    probes: tuple[CaseProbe, ...],
    contract: DiagnosticContract,
    *,
    include_order_seven: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate fixed-state Craig--Sulem partial sums and term norms."""

    nx = 1024
    pad_factor = 8
    maximum_order = 7 if include_order_seven else 6
    _, wavenumbers = build_grid(nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    filter_fraction = 2.0 * contract.common_wavenumber / nx

    row_case_ids: list[str] = []
    row_times: list[float] = []
    eta_rows: list[FloatArray] = []
    xi_rows: list[FloatArray] = []
    depth_rows: list[float] = []
    for probe in probes:
        source = source_cases[probe.case_id]
        for time in probe.static_times:
            index = _time_index(source.source_times, time)
            eta = resize_project_periodic_real(
                source.eta[index],
                output_nx=nx,
                length=contract.length,
                maximum_wavenumber=contract.common_wavenumber,
            )
            xi = resize_project_periodic_real(
                source.xi[index],
                output_nx=nx,
                length=contract.length,
                maximum_wavenumber=contract.common_wavenumber,
            )
            eta_rows.append(eta)
            xi_rows.append(xi)
            depth_rows.append(source.depth)
            row_case_ids.append(probe.case_id)
            row_times.append(time)

    eta_batch = jnp.asarray(np.stack(eta_rows), dtype=jnp.float64)
    xi_batch = jnp.asarray(np.stack(xi_rows), dtype=jnp.float64)
    depth_batch = jnp.asarray(
        np.asarray(depth_rows, dtype=np.float64)[:, None],
        dtype=jnp.float64,
    )
    partial_sums = []
    for order in range(maximum_order + 1):
        value = dno_series_eval(
            eta_batch,
            xi_batch,
            k,
            depth_batch,
            order,
            pad_factor=pad_factor,
        )
        value = apply_lowpass(value, k, filter_fraction)
        value -= jnp.mean(value, axis=-1, keepdims=True)
        partial_sums.append(
            np.asarray(jax.device_get(value), dtype=np.float64)
        )
    partial_sum_array = np.stack(partial_sums, axis=1)

    rows: list[dict[str, Any]] = []
    restart_by_case = {
        probe.case_id: probe.restart_time for probe in probes
    }
    for row_index, (case_id, time) in enumerate(
        zip(row_case_ids, row_times, strict=True)
    ):
        cumulative_array = partial_sum_array[row_index]
        terms = np.concatenate(
            (
                cumulative_array[:1],
                np.diff(cumulative_array, axis=0),
            ),
            axis=0,
        )
        partial_norm = _rms(cumulative_array)
        term_norm = _rms(terms)
        base_norm = max(float(term_norm[0]), np.finfo(np.float64).tiny)
        order_records = []
        for order in range(maximum_order + 1):
            previous_term = (
                float(term_norm[order - 1]) if order > 0 else None
            )
            current_partial = max(
                float(partial_norm[order]), np.finfo(np.float64).tiny
            )
            order_records.append(
                {
                    "order": order,
                    "partial_sum_rms": float(partial_norm[order]),
                    "term_rms": float(term_norm[order]),
                    "term_rms_over_g0_rms": float(
                        term_norm[order] / base_norm
                    ),
                    "term_rms_over_previous_term_rms": (
                        float(term_norm[order] / previous_term)
                        if previous_term is not None
                        and previous_term > 0.0
                        else None
                    ),
                    "successive_partial_sum_difference_over_partial_sum": (
                        float(term_norm[order] / current_partial)
                        if order > 0
                        else None
                    ),
                }
            )
        rows.append(
            {
                "case_id": case_id,
                "time": time,
                "restart_snapshot": math.isclose(
                    time,
                    restart_by_case[case_id],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "all_partial_sums_finite": bool(
                    np.isfinite(cumulative_array).all()
                ),
                "orders": order_records,
                "cancellation_index_at_maximum_order": float(
                    np.sum(term_norm)
                    / max(
                        float(partial_norm[-1]),
                        np.finfo(np.float64).tiny,
                    )
                ),
            }
        )

    metadata = {
        "schema": "precascade_static_cs_order_precursor_v1",
        "definition": {
            "grid_points": nx,
            "padding_factor": pad_factor,
            "delivered_wavenumber": contract.common_wavenumber,
            "orders": list(range(maximum_order + 1)),
            "term": (
                "projected G_m equals the order-m partial sum minus the "
                "order-(m-1) partial sum; projected G_0 is term zero"
            ),
            "rms": "square root of the spatial mean of the squared field",
            "interpretation": (
                "growth or non-decay of successive term norms is evidence "
                "that the truncated Craig--Sulem series is not asymptotically "
                "ordered at that fixed state"
            ),
        },
        "rows": rows,
    }
    arrays = {
        "schema": np.asarray(metadata["schema"]),
        "metadata_json": np.asarray(_canonical_json(metadata)),
        "case_id": np.asarray(row_case_ids),
        "time": np.asarray(row_times, dtype=np.float64),
        "order": np.arange(maximum_order + 1, dtype=np.int32),
        "partial_sum_gxi": partial_sum_array,
    }
    return metadata, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arm", type=Path, default=DEFAULT_SOURCE_ARM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Optional subset of arm identifiers; baseline is always included.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Write the fixed-state Craig--Sulem order precursor, then stop.",
    )
    parser.add_argument(
        "--include-order-seven-static",
        action="store_true",
        help=(
            "Also evaluate static order seven. The default stops at six "
            "because order seven is not part of the production model."
        ),
    )
    parser.add_argument(
        "--allow-cpu-production",
        action="store_true",
        help="Allow the expensive dynamic matrix when no GPU is visible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching static and dynamic artifacts.",
    )
    return parser.parse_args()


def _selected_arms(
    requested: list[str] | None,
) -> tuple[ArmDefinition, ...]:
    arms = production_arm_definitions()
    validate_arm_matrix(arms)
    if requested is None:
        return arms
    known = {arm.arm_id: arm for arm in arms}
    unknown = sorted(set(requested) - known.keys())
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    identifiers = ["baseline", *requested]
    return tuple(known[name] for name in dict.fromkeys(identifiers))


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume the static precursor and selected dynamic arms."""

    contract = DiagnosticContract()
    probes = production_case_probes()
    arms = _selected_arms(args.arms)
    devices = [
        {
            "id": int(device.id),
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    if (
        not args.static_only
        and not args.allow_cpu_production
        and not any(
            device["platform"] in {"gpu", "cuda", "rocm"}
            for device in devices
        )
    ):
        raise RuntimeError(
            "the dynamic matrix requires a GPU; use --allow-cpu-production "
            "only if the long CPU calculation is intentional"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cases, source_record = load_source_cases(
        args.source_arm.resolve(), probes
    )
    script_sha256 = _sha256(Path(__file__).resolve())
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema": "precascade_spatial_order_diagnostic_v1",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_revision": _source_revision(),
        "script_sha256": script_sha256,
        "source": source_record,
        "contract": asdict(contract),
        "cases": [asdict(probe) for probe in probes],
        "arms": [asdict(arm) for arm in arms],
        "devices": devices,
        "one_factor_interpretation": {
            "grid_resolution": (
                "Compare grid_2048 with baseline. A change implicates the "
                "base collocation grid at fixed pad/K/M."
            ),
            "product_padding": (
                "Compare padding_16 with baseline. A change implicates "
                "Craig--Sulem product padding at fixed N/K/M."
            ),
            "cutoff_crowding": (
                "Compare cutoff_192 with baseline, including both P128 "
                "differences and energy entering 129<=|k|<=192. A delayed "
                "cascade implicates crowding at the K=128 projection wall; "
                "a later K=256 run is then required to establish a trend."
            ),
            "series_order": (
                "Compare M=4, M=5, and M=6 at fixed N/pad/K. Monotone "
                "agreement with increasing M supports order convergence; "
                "strong non-monotone changes indicate order sensitivity."
            ),
        },
        "limitations": [
            (
                "Every restart inherits its history from the baseline "
                "N=1024, pad=8, K=128, M=6 coarse trajectory."
            ),
            (
                "The restart comparison diagnoses local post-restart "
                "sensitivity; it cannot establish full-horizon convergence."
            ),
            (
                "Any promising numerical contract must be rerun from the "
                "original initial condition over the complete horizon."
            ),
        ],
        "static_order_precursor": {},
        "dynamic_cases": [],
    }
    summary["cases"] = [
        {
            **asdict(probe),
            "source_restart_guard": summarize_source_restart(
                source_cases[probe.case_id], probe, contract
            ),
        }
        for probe in probes
    ]
    _write_json_atomic(summary_path, summary)

    static_json_path = output_dir / "static_order_precursor.json"
    static_npz_path = output_dir / "static_order_precursor.npz"
    if (
        args.overwrite
        or not static_json_path.exists()
        or not static_npz_path.exists()
    ):
        static_start = perf_counter()
        static_metadata, static_arrays = evaluate_static_order_precursor(
            source_cases,
            probes,
            contract,
            include_order_seven=args.include_order_seven_static,
        )
        static_metadata["timing_seconds"] = perf_counter() - static_start
        static_arrays["metadata_json"] = np.asarray(
            _canonical_json(static_metadata)
        )
        _write_npz_atomic(static_npz_path, static_arrays)
        _write_json_atomic(static_json_path, static_metadata)
    summary["static_order_precursor"] = {
        "json": static_json_path.name,
        "json_sha256": _sha256(static_json_path),
        "npz": static_npz_path.name,
        "npz_sha256": _sha256(static_npz_path),
    }
    _write_json_atomic(summary_path, summary)

    if args.static_only:
        summary["status"] = "complete"
        summary["completed_at"] = datetime.now().astimezone().isoformat()
        summary["result"] = "STATIC PRECURSOR COMPLETE; dynamic arms not run"
        _write_json_atomic(summary_path, summary)
        return summary

    for probe in probes:
        arm_arrays: dict[str, ArrayDict] = {}
        arm_records: list[dict[str, Any]] = []
        for arm in arms:
            path = _artifact_path(output_dir, probe, arm)
            reused = path.exists() and not args.overwrite
            if reused:
                arrays = load_dynamic_arm(
                    path,
                    source_record=source_record,
                    probe=probe,
                    arm=arm,
                    contract=contract,
                    script_sha256=script_sha256,
                )
                timing: dict[str, float] | None = None
            else:
                arrays, timing = run_dynamic_arm(
                    source_cases[probe.case_id],
                    probe,
                    arm,
                    contract,
                )
                save_dynamic_arm(
                    path,
                    arrays,
                    source_record=source_record,
                    probe=probe,
                    arm=arm,
                    contract=contract,
                    script_sha256=script_sha256,
                    timings=timing,
                )
            arm_arrays[arm.arm_id] = arrays
            arm_records.append(
                {
                    "arm": asdict(arm),
                    "artifact": path.name,
                    "artifact_sha256": _sha256(path),
                    "reused": reused,
                    "timing_seconds": timing,
                    "telemetry_and_spectrum": summarize_dynamic_arm(
                        arrays,
                        arm=arm,
                        contract=contract,
                    ),
                    "comparison_with_baseline": (
                        None
                        if arm.arm_id == "baseline"
                        else compare_with_baseline(
                            arm_arrays["baseline"], arrays, contract
                        )
                    ),
                }
            )
            summary["dynamic_cases"] = [
                record
                for record in summary["dynamic_cases"]
                if record["case_id"] != probe.case_id
            ]
            summary["dynamic_cases"].append(
                {
                    "case_id": probe.case_id,
                    "restart_time": probe.restart_time,
                    "terminal_time": probe.terminal_time,
                    "arms": arm_records,
                }
            )
            _write_json_atomic(summary_path, summary)
        del arm_arrays
        gc.collect()

    summary["status"] = "complete"
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    summary["result"] = (
        "LOCAL RESTART DIAGNOSTIC COMPLETE; inspect one-factor comparisons"
    )
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_diagnostic(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "result": summary["result"],
                "output": str(args.output_dir.resolve() / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
