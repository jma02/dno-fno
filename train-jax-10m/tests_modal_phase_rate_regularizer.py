"""CPU invariants for the centered low-mode modal phase-rate loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from modal_phase_rate_regularizer import (
    ModalPhaseRateConfig,
    compute_modal_phase_rate_loss,
)
from source_conditioning import source_conditioning_for_dataset


jax.config.update("jax_enable_x64", True)


def _grid(n: int = 512) -> tuple[jax.Array, jax.Array]:
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k_rfft = jnp.fft.rfftfreq(n, d=1.0 / n)
    return x, k_rfft


def _loss(
    eta: jax.Array,
    prediction: jax.Array,
    target: jax.Array,
    source: jax.Array | None = None,
    config: ModalPhaseRateConfig = ModalPhaseRateConfig(),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    _, k_rfft = _grid(eta.shape[-1])
    if source is None:
        source = jnp.full((eta.shape[0],), 5, dtype=jnp.int32)
    return compute_modal_phase_rate_loss(
        eta,
        prediction,
        target,
        source,
        k_rfft,
        config,
    )


def test_exact_prediction_has_zero_loss() -> None:
    x, _ = _grid()
    eta = (jnp.sin(5.0 * x) + 0.2 * jnp.cos(11.0 * x))[None, :]
    target = (-0.4 * jnp.gradient(eta[0], x[1] - x[0]))[None, :]
    loss, diagnostics = _loss(eta, target, target)

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    for name in (
        "phase_rate_loss",
        "phase_rate_rms",
        "amplitude_rate_loss",
        "amplitude_rate_rms",
        "kinematic_growth_loss",
        "kinematic_growth_rms",
    ):
        np.testing.assert_allclose(diagnostics[name], 0.0, atol=1e-15)
    assert all(value.shape == () for value in diagnostics.values())


def test_independent_constant_q_offsets_do_not_change_loss() -> None:
    x, _ = _grid()
    eta = (jnp.sin(7.0 * x) + 0.3 * jnp.cos(13.0 * x))[None, :]
    target = (0.4 * jnp.cos(7.0 * x) - 0.2 * jnp.sin(13.0 * x))[None, :]
    prediction = target - 0.03 * jnp.cos(7.0 * x)[None, :]
    loss, diagnostics = _loss(eta, prediction, target)
    shifted_loss, shifted_diagnostics = _loss(
        eta,
        prediction + 2.75,
        target - 8.5,
    )

    np.testing.assert_allclose(shifted_loss, loss, rtol=1e-12, atol=1e-15)
    for name in diagnostics:
        np.testing.assert_allclose(
            shifted_diagnostics[name], diagnostics[name], rtol=1e-12, atol=1e-15
        )


def test_rigid_translation_defect_has_exact_energy_weighted_scaling() -> None:
    x, k_rfft = _grid()
    modes = jnp.asarray([3.0, 11.0], dtype=x.dtype)
    amplitudes = jnp.asarray([1.0, 0.35], dtype=x.dtype)
    eta = jnp.sum(amplitudes[:, None] * jnp.sin(modes[:, None] * x[None, :]), axis=0)[
        None, :
    ]
    eta_hat = jnp.fft.rfft(eta, axis=-1, norm="forward")
    eta_x = jnp.fft.irfft(
        1j * k_rfft[None, :] * eta_hat,
        n=x.size,
        axis=-1,
        norm="forward",
    )
    target_speed = 0.4
    speed_error = 0.017
    target = -target_speed * eta_x
    prediction = target - speed_error * eta_x
    loss, diagnostics = _loss(eta, prediction, target)

    mode_energy = amplitudes**2
    expected = speed_error**2 * jnp.sum(mode_energy * modes**2) / jnp.sum(mode_energy)
    np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        diagnostics["phase_rate_rms"], jnp.sqrt(expected), rtol=1e-12
    )
    np.testing.assert_allclose(diagnostics["amplitude_rate_loss"], 0.0, atol=1e-28)
    np.testing.assert_allclose(
        diagnostics["kinematic_growth_loss"], loss, rtol=1e-12, atol=1e-15
    )


def test_amplitude_growth_is_diagnostic_only() -> None:
    x, _ = _grid()
    eta = (jnp.cos(9.0 * x) + 0.2 * jnp.sin(15.0 * x))[None, :]
    target = 0.3 * jnp.sin(9.0 * x)[None, :]
    amplitude_rate_error = 0.025
    prediction = target + amplitude_rate_error * eta
    loss, diagnostics = _loss(eta, prediction, target)

    np.testing.assert_allclose(loss, 0.0, atol=1e-28)
    np.testing.assert_allclose(
        diagnostics["amplitude_rate_loss"], amplitude_rate_error**2, rtol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["kinematic_growth_loss"],
        diagnostics["amplitude_rate_loss"],
        rtol=1e-12,
    )


def test_target_only_activity_mask_ignores_unrepresented_modes() -> None:
    x, _ = _grid()
    eta = jnp.sin(7.0 * x)[None, :]
    target = jnp.cos(7.0 * x)[None, :]
    prediction = target + 10.0 * jnp.cos(63.0 * x)[None, :]
    loss, diagnostics = _loss(eta, prediction, target)

    np.testing.assert_allclose(loss, 0.0, atol=1e-25)
    np.testing.assert_allclose(diagnostics["kinematic_growth_loss"], 0.0, atol=1e-25)
    np.testing.assert_allclose(diagnostics["active_modes"], 1.0, atol=1e-15)


def test_configured_band_excludes_high_modes() -> None:
    x, _ = _grid()
    eta = (jnp.sin(7.0 * x) + jnp.sin(150.0 * x))[None, :]
    target = jnp.zeros_like(eta)
    prediction = 2.0 * jnp.cos(150.0 * x)[None, :]
    loss, diagnostics = _loss(eta, prediction, target)

    np.testing.assert_allclose(loss, 0.0, atol=1e-25)
    np.testing.assert_allclose(diagnostics["active_modes"], 1.0, atol=1e-15)


def test_source_selection_and_empty_samples_are_finite() -> None:
    x, _ = _grid()
    eta = jnp.stack((jnp.sin(5.0 * x), jnp.zeros_like(x)))
    target = jnp.zeros_like(eta)
    prediction = jnp.stack((0.1 * jnp.cos(5.0 * x), 3.0 + jnp.cos(9.0 * x)))
    source = jnp.asarray([0, 5], dtype=jnp.int32)
    loss, diagnostics = _loss(eta, prediction, target, source)

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics["selected_samples"], 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics["active_modes"], 0.0, atol=1e-15)
    assert all(bool(jnp.isfinite(value)) for value in diagnostics.values())


def test_dataset_aware_tanaka_selection_does_not_mix_source_schemas() -> None:
    x, _ = _grid()
    eta = jnp.sin(5.0 * x)[None, :]
    target = jnp.zeros_like(eta)
    prediction = 0.1 * jnp.cos(5.0 * x)[None, :]
    schema_v2 = source_conditioning_for_dataset({"split_id": object()})
    schema_v2_loss, schema_v2_diagnostics = _loss(
        eta,
        prediction,
        target,
        jnp.asarray([2], dtype=jnp.int32),
        ModalPhaseRateConfig(source_ids=schema_v2.tanaka_source_ids),
    )
    assert float(schema_v2_loss) > 0.0
    np.testing.assert_allclose(
        schema_v2_diagnostics["selected_samples"], 1.0, atol=1e-15
    )

    legacy = source_conditioning_for_dataset({})
    assert ModalPhaseRateConfig().source_ids == legacy.tanaka_source_ids
    legacy_loss, legacy_diagnostics = _loss(
        eta,
        prediction,
        target,
        jnp.asarray([2], dtype=jnp.int32),
        ModalPhaseRateConfig(source_ids=legacy.tanaka_source_ids),
    )
    np.testing.assert_allclose(legacy_loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        legacy_diagnostics["selected_samples"], 0.0, atol=1e-15
    )


def test_gradient_is_finite_nonzero_and_zero_mean() -> None:
    x, k_rfft = _grid()
    eta = jnp.sin(6.0 * x)[None, :]
    target = -0.4 * 6.0 * jnp.cos(6.0 * x)[None, :]
    prediction = target - 0.02 * 6.0 * jnp.cos(6.0 * x)[None, :]
    source = jnp.asarray([6], dtype=jnp.int32)

    def objective(value: jax.Array) -> jax.Array:
        return compute_modal_phase_rate_loss(
            eta,
            value,
            target,
            source,
            k_rfft,
            ModalPhaseRateConfig(),
        )[0]

    gradient = jax.jit(jax.grad(objective))(prediction)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
    np.testing.assert_allclose(jnp.sum(gradient, axis=-1), 0.0, atol=1e-14)


def test_selected_count_shard_weighting_matches_unsharded_loss() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        return

    x, k_rfft = _grid()
    eta_one = jnp.sin(7.0 * x)
    eta = jnp.broadcast_to(eta_one, (6, x.size))
    eta_x = 7.0 * jnp.cos(7.0 * x)
    speed_errors = jnp.asarray([0.01, 0.9, 0.9, 0.02, 0.03, 0.9], dtype=x.dtype)
    prediction = -speed_errors[:, None] * eta_x[None, :]
    target = jnp.zeros_like(prediction)
    source = jnp.asarray([5, 0, 0, 5, 14, 0], dtype=jnp.int32)
    config = ModalPhaseRateConfig()
    expected, _ = compute_modal_phase_rate_loss(
        eta, prediction, target, source, k_rfft, config
    )
    mesh = Mesh(np.asarray(devices[:2]), ("batch",))

    def distributed_loss(
        eta_local: jax.Array,
        prediction_local: jax.Array,
        target_local: jax.Array,
        source_local: jax.Array,
    ) -> jax.Array:
        local_loss, diagnostics = compute_modal_phase_rate_loss(
            eta_local,
            prediction_local,
            target_local,
            source_local,
            k_rfft,
            config,
        )
        local_count = diagnostics["selected_samples"]
        global_count = jax.lax.psum(local_count, axis_name="batch")
        device_count = jax.lax.psum(
            jnp.asarray(1.0, dtype=eta_local.dtype), axis_name="batch"
        )
        shard_weight = device_count * local_count / jnp.maximum(global_count, 1.0)
        return jax.lax.pmean(local_loss * shard_weight, axis_name="batch")

    sharded_loss = jax.jit(
        shard_map(
            distributed_loss,
            mesh=mesh,
            in_specs=(P("batch"), P("batch"), P("batch"), P("batch")),
            out_specs=P(),
            check_rep=False,
        )
    )(eta, prediction, target, source)
    np.testing.assert_allclose(sharded_loss, expected, rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    tests = (
        test_exact_prediction_has_zero_loss,
        test_independent_constant_q_offsets_do_not_change_loss,
        test_rigid_translation_defect_has_exact_energy_weighted_scaling,
        test_amplitude_growth_is_diagnostic_only,
        test_target_only_activity_mask_ignores_unrepresented_modes,
        test_configured_band_excludes_high_modes,
        test_source_selection_and_empty_samples_are_finite,
        test_dataset_aware_tanaka_selection_does_not_mix_source_schemas,
        test_gradient_is_finite_nonzero_and_zero_mean,
        test_selected_count_shard_weighting_matches_unsharded_loss,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
