"""Checkpoint loading and surrogate-driven rollout for JAX operator models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import checkpoints

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _directory in (
    REPO_ROOT,
    REPO_ROOT / "models" / "fno-jax",
    REPO_ROOT / "models" / "dno-net",
):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from dno_net_v2 import (  # noqa: E402
    CraigSulemDNO,
    validate_fixed_craig_sulem_config,
)
from fno1d import FNO1d  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import myfft, myifft  # noqa: E402


Predictor = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
BatchedPredictor = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
GL2_ITERATIONS = 4


class LoadedRun(NamedTuple):
    model: Any
    params: dict[str, Any]
    config: dict[str, Any]
    stats: dict[str, Any]
    norm_mode: str
    epoch: int


def load_run(
    run_dir: str | Path,
    *,
    checkpoint: Literal["best", "final"] = "best",
    domain_length_override: float | None = None,
) -> LoadedRun:
    """Restore a JAX FNO or fixed CS-DNO training checkpoint."""
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if domain_length_override is not None:
        config["domain_length"] = float(domain_length_override)
    checkpoint_dir = (
        run_dir
        / {
            "best": "best_val_ckpt",
            "final": "final_ckpt",
        }[checkpoint]
    )
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    stats = metadata["stats"]
    norm_mode = str(config["norm"])

    model_name = str(config["model"])

    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])

    if model_name == "cs_dno":
        validate_fixed_craig_sulem_config(config)
        model = CraigSulemDNO(
            width=int(config["width"]),
            n_blocks=int(config["n_blocks"]),
            latent=int(config["latent"]),
            n_polys=int(config["cs_n_polys"]),
            use_first_deriv=bool(config["cs_use_first_deriv"]),
            use_second_deriv=bool(config["cs_use_second_deriv"]),
            use_half_deriv=bool(config["cs_use_half_deriv"]),
            use_hilbert=bool(config["cs_use_hilbert"]),
            mult_hidden=int(config["cs_mult_hidden"]),
            domain_length=float(config["domain_length"]),
            xi_scale=float(config["xi_scale"]),
            eta_scale=float(config["eta_scale"]),
            target_scale=float(config["target_scale"]),
        )
    elif model_name == "fno":
        model = FNO1d(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config["n_blocks"]),
            domain_length=float(config["domain_length"]),
            xi_scale=float(config["xi_scale"]),
            target_scale=float(config["target_scale"]),
            eta_features=bool(config.get("fno_eta_features", False)),
        )
    else:
        raise ValueError(f"unsupported checkpoint model: {model_name!r}")

    return LoadedRun(
        model=model,
        params=params,
        config=config,
        stats=stats,
        norm_mode=norm_mode,
        epoch=int(metadata["epoch"]),
    )


def build_predict_gxi_batched(loaded: LoadedRun) -> BatchedPredictor:
    """Build a JIT-compiled ``(eta, xi, log_depth) -> G(eta)xi`` predictor.

    ``eta`` and ``xi`` have shape ``(batch, nx)``. ``log_depth`` has shape
    ``(batch,)`` or ``(batch, 1)``. The returned field is mean-free per sample.
    """
    model_dtype = jnp.float32
    if loaded.norm_mode == "scale":
        feature_absmax = jnp.asarray(
            loaded.stats["feature_absmax"], dtype=model_dtype
        ).reshape(1, 1, 2)
        feature_absmax = jnp.where(feature_absmax > 0, feature_absmax, 1.0)
        target_scale = float(loaded.stats["target_absmax"]) or 1.0

        @jax.jit
        def predict(
            eta: jnp.ndarray,
            xi: jnp.ndarray,
            log_depth: jnp.ndarray,
        ) -> jnp.ndarray:
            features = jnp.stack(
                (eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1
            )
            depth = jnp.asarray(log_depth, dtype=model_dtype).reshape(-1, 1)
            output = loaded.model.apply(
                {"params": loaded.params}, features / feature_absmax, depth
            )
            gxi = (output[..., 0] * target_scale).astype(eta.dtype)
            return gxi - gxi.mean(axis=-1, keepdims=True)

        return predict

    if loaded.norm_mode != "minmax":
        raise ValueError(f"unsupported normalization mode: {loaded.norm_mode!r}")
    feature_min = jnp.asarray(loaded.stats["feature_min"], dtype=model_dtype).reshape(
        1, 1, 2
    )
    feature_max = jnp.asarray(loaded.stats["feature_max"], dtype=model_dtype).reshape(
        1, 1, 2
    )
    feature_range = feature_max - feature_min + 1e-8
    target_min = float(loaded.stats["target_min"])
    target_range = float(loaded.stats["target_max"]) - target_min + 1e-8

    @jax.jit
    def predict(
        eta: jnp.ndarray,
        xi: jnp.ndarray,
        log_depth: jnp.ndarray,
    ) -> jnp.ndarray:
        features = jnp.stack((eta.astype(model_dtype), xi.astype(model_dtype)), axis=-1)
        normalized = ((features - feature_min) / feature_range) * 2.0 - 1.0
        depth = jnp.asarray(log_depth, dtype=model_dtype).reshape(-1, 1)
        output = loaded.model.apply({"params": loaded.params}, normalized, depth)
        denormalized = ((output[..., 0] + 1.0) * 0.5) * target_range + target_min
        gxi = denormalized.astype(eta.dtype)
        return gxi - gxi.mean(axis=-1, keepdims=True)

    return predict


def _rhs_nonlinear_surrogate(
    state: ti.State,
    params: ti.SolverParams,
    predict_gxi: Predictor,
) -> ti.State:
    eta_x = ti.spectral_dx(state.eta, params.k)
    xi_x = ti.spectral_dx(state.xi, params.k)
    gxi = predict_gxi(state.eta, state.xi)
    eta_t = gxi - ti.linear_dno_action(state.xi, params.g0)
    xi_t = ti.dealiased_zakharov_xi_rhs(eta_x, xi_x, gxi)
    if params.filter_fraction < 1.0:
        eta_t = ti.apply_lowpass(eta_t, params.k, params.filter_fraction)
        xi_t = ti.apply_lowpass(xi_t, params.k, params.filter_fraction)
    return ti.State(eta=eta_t, xi=xi_t)


def _rhs_nonlinear_if_surrogate(
    state_hat: ti.SpectralState,
    time_value: float | jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi: Predictor,
) -> ti.SpectralState:
    physical_hat = ti.apply_linear_flow_hat(state_hat, time_value, params)
    physical_state = ti.State(
        eta=myifft(physical_hat.eta_hat),
        xi=myifft(physical_hat.xi_hat),
    )
    nonlinear = _rhs_nonlinear_surrogate(physical_state, params, predict_gxi)
    nonlinear_hat = ti.SpectralState(
        eta_hat=myfft(nonlinear.eta, params.nx),
        xi_hat=myfft(nonlinear.xi, params.nx),
    )
    return ti.apply_linear_flow_hat(nonlinear_hat, -time_value, params)


def _gl2_if_step_surrogate(
    state: ti.State,
    time_value: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi: Predictor,
) -> ti.State:
    sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=state.eta.dtype))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    state_hat = ti.SpectralState(
        eta_hat=myfft(state.eta, params.nx),
        xi_hat=myfft(state.xi, params.nx),
    )
    initial_if_state = ti.apply_linear_flow_hat(state_hat, -time_value, params)

    def add(
        left: ti.SpectralState,
        right: ti.SpectralState,
    ) -> ti.SpectralState:
        return jax.tree_util.tree_map(
            lambda left_value, right_value: left_value + right_value,
            left,
            right,
        )

    def scale(
        value: ti.SpectralState,
        factor: float | jnp.ndarray,
    ) -> ti.SpectralState:
        return jax.tree_util.tree_map(
            lambda item: factor * item,
            value,
        )

    def body_fn(
        _: int,
        stages: tuple[ti.SpectralState, ti.SpectralState],
    ) -> tuple[ti.SpectralState, ti.SpectralState]:
        stage1, stage2 = stages
        rhs1 = _rhs_nonlinear_if_surrogate(
            stage1, time_value + c1 * dt, params, predict_gxi
        )
        rhs2 = _rhs_nonlinear_if_surrogate(
            stage2, time_value + c2 * dt, params, predict_gxi
        )
        candidate1 = add(
            initial_if_state,
            scale(add(scale(rhs1, a11), scale(rhs2, a12)), dt),
        )
        candidate2 = add(
            initial_if_state,
            scale(add(scale(rhs1, a21), scale(rhs2, a22)), dt),
        )
        return candidate1, candidate2

    stage1, stage2 = jax.lax.fori_loop(
        0,
        GL2_ITERATIONS,
        body_fn,
        (initial_if_state, initial_if_state),
    )
    rhs1 = _rhs_nonlinear_if_surrogate(
        stage1, time_value + c1 * dt, params, predict_gxi
    )
    rhs2 = _rhs_nonlinear_if_surrogate(
        stage2, time_value + c2 * dt, params, predict_gxi
    )
    next_if_state = add(initial_if_state, scale(add(rhs1, rhs2), 0.5 * dt))
    next_hat = ti.apply_linear_flow_hat(next_if_state, time_value + dt, params)
    next_state = ti.State(
        eta=myifft(next_hat.eta_hat),
        xi=myifft(next_hat.xi_hat),
    )
    if params.filter_fraction >= 1.0:
        return next_state
    return ti.State(
        eta=ti.apply_lowpass(next_state.eta, params.k, params.filter_fraction),
        xi=ti.apply_lowpass(next_state.xi, params.k, params.filter_fraction),
    )


def rollout_surrogate(
    initial: ti.State,
    times: jnp.ndarray,
    params: ti.SolverParams,
    predict_gxi: Predictor,
    *,
    substeps: int = 8,
) -> dict[str, jnp.ndarray]:
    """Integrate a surrogate DNO action with the fixed hard-filtered GL2 scheme."""
    state = ti.State(
        eta=jnp.asarray(initial.eta),
        xi=jnp.asarray(initial.xi),
    )
    state = ti.project_zero_mean_xi(state)
    initial_gxi = predict_gxi(state.eta, state.xi)
    if times.shape[0] == 1:
        return {
            "times": times,
            "eta": state.eta[None, :],
            "xi": state.xi[None, :],
            "gxi": initial_gxi[None, :],
        }

    def step_fn(
        carry: ti.State,
        data: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[ti.State, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        current_time, interval = data
        substep_dt = interval / substeps

        def body_fn(substep: int, substate: ti.State) -> ti.State:
            next_state = _gl2_if_step_surrogate(
                substate,
                current_time + substep_dt * substep,
                substep_dt,
                params,
                predict_gxi,
            )
            return ti.project_zero_mean_xi(next_state)

        next_state = jax.lax.fori_loop(0, substeps, body_fn, carry)
        saved_gxi = predict_gxi(next_state.eta, next_state.xi)
        return next_state, (next_state.eta, next_state.xi, saved_gxi)

    intervals = times[1:] - times[:-1]
    _, (saved_eta, saved_xi, saved_gxi) = jax.lax.scan(
        step_fn,
        state,
        (times[:-1], intervals),
    )
    return {
        "times": times,
        "eta": jnp.concatenate((state.eta[None, :], saved_eta), axis=0),
        "xi": jnp.concatenate((state.xi[None, :], saved_xi), axis=0),
        "gxi": jnp.concatenate((initial_gxi[None, :], saved_gxi), axis=0),
    }
