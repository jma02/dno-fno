"""Finite-time full-state phase/action regularization.

The learned and order-six Craig--Sulem vector fields are rolled out from the
same snapshot with the production GL2 integrating-factor map.  Differentiating
the path loss through the learned rollout is the discrete full-state adjoint;
unlike the older translation loss, no instantaneous ``eta_x`` projection is
used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from solver.solvers.time_integrator import (
    SpectralState,
    State,
    _hat_to_state,
    _state_to_hat,
    _tree_add,
    _tree_scale,
    apply_lowpass,
    make_linear_dno_symbol,
)
from stage_tangent_regularizer import (
    ReferenceIFParams,
    _apply_linear_flow_batched,
    gl2_coefficients,
    make_model_F,
    make_reference_F,
)


VectorField = Callable[[SpectralState, jax.Array], SpectralState]
Diagnostics = dict[str, jax.Array]


@dataclass(frozen=True)
class FiniteTimePhaseConfig:
    """Configuration for a sparse paired short-rollout loss."""

    dt: float = 0.01
    substeps: int = 8
    picard_iterations: int = 4
    filter_fraction: float = 0.25
    gravity: float = 1.0
    reference_order: int = 6
    reference_pad: int = 8
    k_max: float = 128.0
    active_scale_relative: float = 1e-4
    denominator_floor_relative: float = 1e-6
    absolute_floor: float = 1e-24
    huber_delta: float = 1.0
    ratio_cap: float = 100.0


def validate_config(config: FiniteTimePhaseConfig) -> None:
    """Reject settings that do not define a finite, nonempty rollout."""
    if config.dt <= 0.0:
        raise ValueError(f"dt must be positive, got {config.dt}")
    if config.substeps < 1:
        raise ValueError(f"substeps must be at least one, got {config.substeps}")
    if config.picard_iterations < 1:
        raise ValueError(
            "picard_iterations must be at least one, got "
            f"{config.picard_iterations}"
        )
    if not 0.0 < config.filter_fraction <= 1.0:
        raise ValueError(
            "filter_fraction must lie in (0, 1], got "
            f"{config.filter_fraction}"
        )
    if config.gravity <= 0.0:
        raise ValueError(f"gravity must be positive, got {config.gravity}")
    if config.reference_order < 0:
        raise ValueError(
            f"reference_order must be nonnegative, got {config.reference_order}"
        )
    if config.reference_pad < 1:
        raise ValueError(
            f"reference_pad must be at least one, got {config.reference_pad}"
        )
    if config.k_max <= 0.0:
        raise ValueError(f"k_max must be positive, got {config.k_max}")
    for name, value in (
        ("active_scale_relative", config.active_scale_relative),
        ("denominator_floor_relative", config.denominator_floor_relative),
        ("absolute_floor", config.absolute_floor),
        ("huber_delta", config.huber_delta),
        ("ratio_cap", config.ratio_cap),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")


def sample_source_microbatch(
    rng: jax.Array,
    eta: jax.Array,
    xi: jax.Array,
    depth: jax.Array,
    source: jax.Array,
    source_id: jax.Array,
    local_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample a fixed-size microbatch from one source without dynamic shapes.

    The returned validity mask handles the extraordinarily unlikely event that
    a device-local batch contains fewer than ``local_size`` matching rows.
    """
    if local_size < 1:
        raise ValueError(f"local_size must be at least one, got {local_size}")
    if local_size > eta.shape[0]:
        raise ValueError(
            f"local_size {local_size} exceeds local batch size {eta.shape[0]}"
        )
    eligible = source == source_id.astype(source.dtype)
    priority = jax.random.uniform(rng, source.shape, dtype=eta.dtype)
    priority = jnp.where(eligible, priority, jnp.asarray(-1.0, eta.dtype))
    _, indices = jax.lax.top_k(priority, local_size)
    valid = eligible[indices].astype(eta.dtype)
    return eta[indices], xi[indices], depth[indices], valid


def _postprocess_if_state(
    v_hat: SpectralState,
    time: jax.Array,
    params: ReferenceIFParams,
) -> tuple[SpectralState, State]:
    """Apply the production post-step filter and zero-mean-xi gauge."""
    physical_hat = _apply_linear_flow_batched(
        v_hat,
        time,
        params.k,
        params.g0,
        params.gravity,
    )
    state = _hat_to_state(physical_hat)
    if params.filter_fraction < 1.0:
        state = State(
            eta=apply_lowpass(state.eta, params.k, params.filter_fraction),
            xi=apply_lowpass(state.xi, params.k, params.filter_fraction),
        )
    state = State(
        eta=state.eta,
        xi=state.xi - jnp.mean(state.xi, axis=-1, keepdims=True),
    )
    filtered_hat = _state_to_hat(state, params.nx)
    return (
        _apply_linear_flow_batched(
            filtered_hat,
            -time,
            params.k,
            params.g0,
            params.gravity,
        ),
        state,
    )


def gl2_if_path(
    vector_field: VectorField,
    initial_state: State,
    params: ReferenceIFParams,
    *,
    dt: float,
    substeps: int,
    picard_iterations: int,
    rematerialize: bool,
) -> State:
    """Return all post-step physical states of a batched GL2-IF rollout."""
    dtype = initial_state.eta.dtype
    dt_array = jnp.asarray(dt, dtype=dtype)
    coeffs = gl2_coefficients(dtype=dtype)
    initial = State(
        eta=initial_state.eta,
        xi=initial_state.xi
        - jnp.mean(initial_state.xi, axis=-1, keepdims=True),
    )
    v0 = _state_to_hat(initial, params.nx)

    def body(
        v_start: SpectralState,
        step_index: jax.Array,
    ) -> tuple[SpectralState, State]:
        start_time = step_index.astype(dtype) * dt_array
        stage1, stage2 = v_start, v_start
        for _ in range(picard_iterations):
            f1 = vector_field(stage1, start_time + coeffs.c1 * dt_array)
            f2 = vector_field(stage2, start_time + coeffs.c2 * dt_array)
            stage1 = _tree_add(
                v_start,
                _tree_scale(
                    _tree_add(
                        _tree_scale(f1, coeffs.a11),
                        _tree_scale(f2, coeffs.a12),
                    ),
                    dt_array,
                ),
            )
            stage2 = _tree_add(
                v_start,
                _tree_scale(
                    _tree_add(
                        _tree_scale(f1, coeffs.a21),
                        _tree_scale(f2, coeffs.a22),
                    ),
                    dt_array,
                ),
            )
        f1 = vector_field(stage1, start_time + coeffs.c1 * dt_array)
        f2 = vector_field(stage2, start_time + coeffs.c2 * dt_array)
        v_next = _tree_add(
            v_start,
            _tree_scale(_tree_add(f1, f2), 0.5 * dt_array),
        )
        return _postprocess_if_state(
            v_next,
            start_time + dt_array,
            params,
        )

    scan_body = jax.checkpoint(body) if rematerialize else body
    _, states = jax.lax.scan(
        scan_body,
        v0,
        jnp.arange(substeps, dtype=jnp.int32),
    )
    return states


def _capped_pseudo_huber(
    squared_ratio: jax.Array,
    delta: jax.Array,
    cap: jax.Array,
) -> jax.Array:
    clipped_ratio = jnp.minimum(squared_ratio, cap)
    scaled_ratio = clipped_ratio / delta**2
    return 2.0 * clipped_ratio / (jnp.sqrt(1.0 + scaled_ratio) + 1.0)


def full_state_path_loss(
    model_path: State,
    reference_path: State,
    depth: jax.Array,
    k_rfft: jax.Array,
    sample_weight: jax.Array,
    config: FiniteTimePhaseConfig,
) -> tuple[jax.Array, Diagnostics]:
    """Score phase, action, and polarization growth along a paired path."""
    validate_config(config)
    dtype = reference_path.eta.dtype
    model_eta = model_path.eta.astype(dtype)
    model_xi = model_path.xi.astype(dtype)
    reference_eta = jax.lax.stop_gradient(reference_path.eta.astype(dtype))
    reference_xi = jax.lax.stop_gradient(reference_path.xi.astype(dtype))
    depth = depth.astype(dtype)
    weights_sample = sample_weight.astype(dtype)

    model_eta_hat = jnp.fft.rfft(model_eta, axis=-1, norm="forward")
    model_xi_hat = jnp.fft.rfft(model_xi, axis=-1, norm="forward")
    reference_eta_hat = jnp.fft.rfft(reference_eta, axis=-1, norm="forward")
    reference_xi_hat = jnp.fft.rfft(reference_xi, axis=-1, norm="forward")

    k_abs = jnp.abs(k_rfft).astype(dtype)
    g0 = (
        k_abs[None, :] * jnp.tanh(depth[:, None] * k_abs[None, :])
    ).astype(dtype)
    gravity = jnp.asarray(config.gravity, dtype=dtype)
    g0_path = g0[None, :, :]

    eta_difference = model_eta_hat - reference_eta_hat
    xi_difference = model_xi_hat - reference_xi_hat
    direct_error = (
        gravity * jnp.abs(eta_difference) ** 2
        + g0_path * jnp.abs(xi_difference) ** 2
    )
    reference_energy = (
        gravity * jnp.abs(reference_eta_hat) ** 2
        + g0_path * jnp.abs(reference_xi_hat) ** 2
    )
    model_energy = (
        gravity * jnp.abs(model_eta_hat) ** 2
        + g0_path * jnp.abs(model_xi_hat) ** 2
    )
    cross = (
        gravity * model_eta_hat * jnp.conj(reference_eta_hat)
        + g0_path * model_xi_hat * jnp.conj(reference_xi_hat)
    )

    band = jnp.logical_and(
        k_abs > jnp.asarray(0.0, dtype=dtype),
        k_abs <= jnp.asarray(config.k_max, dtype=dtype),
    )
    band_energy = jnp.where(band[None, None, :], reference_energy, 0.0)
    path_scale = jnp.max(band_energy, axis=-1, keepdims=True)
    absolute_floor = jnp.asarray(config.absolute_floor, dtype=dtype)
    denominator_floor = (
        jnp.asarray(config.denominator_floor_relative, dtype=dtype) * path_scale
        + absolute_floor
    )
    activity_floor = (
        jnp.asarray(config.active_scale_relative, dtype=dtype) * path_scale
        + absolute_floor
    )
    soft_activity = reference_energy / (reference_energy + activity_floor)
    mode_weight = jax.lax.stop_gradient(
        jnp.where(band[None, None, :], soft_activity, 0.0)
    )

    times = (
        jnp.arange(1, config.substeps + 1, dtype=dtype)
        * jnp.asarray(config.dt, dtype=dtype)
    )
    time_squared = times[:, None, None] ** 2
    growth_ratio = direct_error / (
        (reference_energy + denominator_floor) * time_squared
    )
    penalty = _capped_pseudo_huber(
        growth_ratio,
        jnp.asarray(config.huber_delta, dtype=dtype),
        jnp.asarray(config.ratio_cap, dtype=dtype),
    )
    mode_weight_sum = jnp.sum(mode_weight, axis=-1)
    per_state_loss = jnp.sum(mode_weight * penalty, axis=-1) / jnp.maximum(
        mode_weight_sum,
        absolute_floor,
    )
    per_sample_loss = jnp.mean(per_state_loss, axis=0)
    selected_count = jnp.sum(weights_sample)
    loss = jnp.sum(weights_sample * per_sample_loss) / jnp.maximum(
        selected_count,
        jnp.asarray(1.0, dtype=dtype),
    )

    sqrt_model_energy = jnp.sqrt(jnp.maximum(model_energy, 0.0))
    sqrt_reference_energy = jnp.sqrt(jnp.maximum(reference_energy, 0.0))
    cross_abs = jnp.abs(cross)
    action_error = (sqrt_model_energy - sqrt_reference_energy) ** 2
    phase_error = 2.0 * jnp.maximum(cross_abs - jnp.real(cross), 0.0)
    polarization_error = 2.0 * jnp.maximum(
        sqrt_model_energy * sqrt_reference_energy - cross_abs,
        0.0,
    )

    def diagnostic_rate(component: jax.Array) -> jax.Array:
        ratio = component / (
            (reference_energy + denominator_floor) * time_squared
        )
        per_state = jnp.sum(mode_weight * ratio, axis=-1) / jnp.maximum(
            mode_weight_sum,
            absolute_floor,
        )
        per_sample = jnp.mean(per_state, axis=0)
        return jnp.sum(weights_sample * per_sample) / jnp.maximum(
            selected_count,
            jnp.asarray(1.0, dtype=dtype),
        )

    total_mode_weight = jnp.sum(
        mode_weight * weights_sample[None, :, None]
    )
    total_mode_weight_safe = jnp.maximum(total_mode_weight, absolute_floor)
    diagnostics: Diagnostics = {
        "loss": loss,
        "raw_growth_rate_rms": jnp.sqrt(
            jnp.sum(
                mode_weight
                * weights_sample[None, :, None]
                * growth_ratio
            )
            / total_mode_weight_safe
        ),
        "action_growth_rate_rms": jnp.sqrt(diagnostic_rate(action_error)),
        "phase_growth_rate_rms": jnp.sqrt(diagnostic_rate(phase_error)),
        "polarization_growth_rate_rms": jnp.sqrt(
            diagnostic_rate(polarization_error)
        ),
        "active_modes": jnp.sum(
            weights_sample[None, :] * mode_weight_sum
        )
        / jnp.maximum(
            jnp.asarray(config.substeps, dtype=dtype) * selected_count,
            jnp.asarray(1.0, dtype=dtype),
        ),
        "selected_samples": selected_count,
        "clipped_mode_fraction": jnp.sum(
            mode_weight
            * weights_sample[None, :, None]
            * (growth_ratio >= config.ratio_cap).astype(dtype)
        )
        / total_mode_weight_safe,
    }
    return loss, diagnostics


def compute_finite_time_phase_regularizer(
    *,
    apply_fn: Callable,
    model_params: object,
    eta_phys: jax.Array,
    xi_phys: jax.Array,
    depth_phys: jax.Array,
    sample_weight: jax.Array,
    batch_depth_local: jax.Array,
    norm_inputs_fn: Callable[[jax.Array, jax.Array], jax.Array],
    denorm_targets_fn: Callable[[jax.Array], jax.Array],
    filter_predictions_fn: Callable[[jax.Array], jax.Array],
    k: jax.Array,
    config: FiniteTimePhaseConfig,
    model_dtype: jnp.dtype,
    reference_dtype: jnp.dtype = jnp.float64,
) -> tuple[jax.Array, Diagnostics]:
    """Build paired paths and return their finite-time modal growth loss."""
    validate_config(config)
    batch_size, nx = eta_phys.shape
    del batch_size

    eta_reference = eta_phys.astype(reference_dtype)
    xi_reference = xi_phys.astype(reference_dtype)
    depth_reference = depth_phys.astype(reference_dtype)
    k_reference = k.astype(reference_dtype)
    g0_reference = jax.vmap(
        lambda h: make_linear_dno_symbol(k_reference, h)
    )(depth_reference)
    reference_params = ReferenceIFParams(
        k=k_reference,
        g0=g0_reference,
        depth=depth_reference,
        gravity=config.gravity,
        dno_order=config.reference_order,
        pad_factor=config.reference_pad,
        filter_fraction=config.filter_fraction,
        nx=nx,
    )
    reference_field = make_reference_F(
        reference_params,
        zero_mean_gxi=True,
    )
    reference_path = gl2_if_path(
        reference_field,
        State(eta=eta_reference, xi=xi_reference),
        reference_params,
        dt=config.dt,
        substeps=config.substeps,
        picard_iterations=config.picard_iterations,
        rematerialize=False,
    )
    reference_path = jax.lax.stop_gradient(reference_path)

    state_dtype = reference_dtype
    eta_model_state = eta_phys.astype(state_dtype)
    xi_model_state = xi_phys.astype(state_dtype)
    depth_model_state = depth_phys.astype(state_dtype)
    k_model_state = k.astype(state_dtype)
    g0_model_state = jax.vmap(
        lambda h: make_linear_dno_symbol(k_model_state, h)
    )(depth_model_state)
    model_params_bundle = ReferenceIFParams(
        k=k_model_state,
        g0=g0_model_state,
        depth=depth_model_state,
        gravity=config.gravity,
        dno_order=config.reference_order,
        pad_factor=config.reference_pad,
        filter_fraction=config.filter_fraction,
        nx=nx,
    )
    model_field = make_model_F(
        apply_fn=apply_fn,
        model_params=model_params,
        batch_depth=batch_depth_local,
        norm_inputs_fn=norm_inputs_fn,
        denorm_targets_fn=denorm_targets_fn,
        filter_predictions_fn=filter_predictions_fn,
        rp=model_params_bundle,
        model_dtype=model_dtype,
        zero_mean_gxi=True,
    )
    model_path = gl2_if_path(
        model_field,
        State(eta=eta_model_state, xi=xi_model_state),
        model_params_bundle,
        dt=config.dt,
        substeps=config.substeps,
        picard_iterations=config.picard_iterations,
        rematerialize=True,
    )

    return full_state_path_loss(
        model_path=model_path,
        reference_path=reference_path,
        depth=depth_reference,
        k_rfft=jnp.abs(k_reference[: nx // 2 + 1]),
        sample_weight=sample_weight,
        config=config,
    )
