"""Deterministic, full-parameter second-order refinement of an EMA checkpoint.

Requires an explicitly chosen TRAIN bundle. Trial evaluations keep its rows, teacher
states, HD probes, batch partition, and fully warmed loss coefficients fixed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
import signal
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import checkpoints
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver.evals.model_rollout import LoadedRun, load_run, rollout_surrogate  # noqa: E402
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from util import build_dataset_split_indices, get_batches, load_dataset_arrays, make_normalizers  # noqa: E402

trainer = importlib.import_module("1d_dno_fno_jax_ema_grad")
jax.config.update("jax_enable_x64", True)
Batch = tuple[jax.Array, ...]
Matvec = Callable[[np.ndarray], np.ndarray]


def batch_residuals(
    params: Any, batch: Batch, teacher: Batch, key: jax.Array, *,
    components: Callable[..., Any], hadamard: Callable[..., jax.Array],
    coefficients: jax.Array, teacher_weight: float, hadamard_weight: float,
    hd_active: bool, teacher_active: bool,
) -> jax.Array:
    """IRLS residuals for curvature, not an alternative acceptance objective.

    The gradient of 0.5 * ||r||^2 matches the native scalar at the current point.
    Robust weights are frozen by autodiff within the underlying loss functions.
    """
    pieces = []
    sources = [(batch, 1.0)]
    if teacher_active:
        sources.append((teacher[:5], teacher_weight))
    for fields, weight in sources:
        values = jax.vmap(
            lambda _: components(params, *fields, return_irls_residuals=True), axis_name="batch",
        )(jnp.arange(1))
        pieces.extend(jnp.sqrt(coefficient * weight) * field[0]
                      for coefficient, field in zip(coefficients, values[:4]))
    if hd_active:
        pieces.append(jnp.sqrt(hadamard_weight) * hadamard(
            key, params, batch[0], batch[1], batch[3], return_irls_residuals=True,
        ))
    return jnp.concatenate([
        jnp.stack((piece.real, piece.imag), axis=-1).ravel() if jnp.iscomplexobj(piece) else piece.ravel()
        for piece in pieces
    ])


def normal_product(residuals: Callable[[Any], jax.Array], params: Any, tangent: Any) -> Any:
    """Apply the real IRLS Jacobian's J-transpose-J without materializing it."""
    _, linear = jax.linearize(residuals, params)
    return jax.linear_transpose(linear, params)(linear(tangent))[0]


def run_lbfgs(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]], initial: np.ndarray, initial_value: float,
    on_accept: Callable[[np.ndarray, float, int], None], *, maxiter: int, maxls: int, seconds: float,
) -> dict[str, Any]:
    """Reject invalid search trials without letting them update weights or EMA."""
    statistics: dict[str, Any] = {"evaluations": 0, "invalid_trials": 0, "interrupted": False}
    started = perf_counter()
    last_vector = initial.copy()
    last_value = accepted_value = initial_value

    def evaluate(vector: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal last_vector, last_value
        if perf_counter() - started >= seconds:
            raise TimeoutError("L-BFGS wall-time cap reached")
        statistics["evaluations"] += 1
        last_vector = vector.copy()
        try:
            value, gradient = objective(vector)
        except FloatingPointError:
            value, gradient = np.nan, np.zeros_like(vector)
        if perf_counter() - started >= seconds:
            raise TimeoutError("L-BFGS wall-time cap reached")
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            last_value = np.nan
            statistics["invalid_trials"] += 1
            # A finite rejection value avoids inf/inf in cubic interpolation.
            # Accepted normalized losses never exceed 1. This placeholder is
            # not a derivative and must never reach a secant or EMA update.
            return 2.0, np.zeros_like(vector)
        last_value = value
        return value / initial_value, gradient / initial_value

    def accept(intermediate_result: Any) -> None:
        nonlocal accepted_value
        if perf_counter() - started >= seconds:
            raise TimeoutError("L-BFGS wall-time cap reached")
        if (not np.isfinite(last_value) or not np.array_equal(intermediate_result.x, last_vector)
                or float(intermediate_result.fun) != last_value / initial_value
                or last_value >= accepted_value):
            raise FloatingPointError("Rejected invalid or nondecreasing L-BFGS accepted iterate")
        accepted_value = last_value
        on_accept(last_vector.copy(), last_value, statistics["evaluations"])

    try:
        result = minimize(evaluate, initial, method="L-BFGS-B", jac=True, callback=accept,
                          options={"maxiter": maxiter, "maxls": maxls, "maxcor": 10,
                                   "ftol": 1e-12, "gtol": 1e-12})
        statistics["termination"] = str(result.message)
    except (TimeoutError, FloatingPointError) as error:
        statistics["termination"] = str(error)
    except KeyboardInterrupt:
        statistics.update(termination="Interrupted; retained the last accepted iterate", interrupted=True)
    statistics["optimization_seconds"] = perf_counter() - started
    return statistics


def run_lm(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    curvature_factory: Callable[[np.ndarray], Matvec], round_parameters: Matvec,
    initial: np.ndarray, on_accept: Callable[[np.ndarray, float, int], None], *,
    maxaccepted: int, maxcg: int, maxattempts: int, seconds: float,
    initial_damping: float = 1e-3, cg_rtol: float = 1e-3, accept_ratio: float = 1e-4,
) -> dict[str, Any]:
    """Matrix-free damped Gauss--Newton with native-objective acceptance.

    The factory freezes symmetric PSD curvature at each accepted center. The
    actual rounded displacement determines predicted reduction; damping is not
    part of that prediction. Inexact CG proposals face the same acceptance test.
    External calls cannot be preempted, but late results cannot be accepted.
    """
    started = perf_counter()
    x = np.asarray(round_parameters(initial), dtype=np.float64).copy()
    value = float("nan")
    damping, rejection_factor = initial_damping, 2.0
    statistics: dict[str, Any] = {
        "evaluations": 0, "matvecs": 0, "cg_iterations": 0,
        "accepted_steps": 0, "attempts": [], "interrupted": False,
    }

    def check_budget() -> None:
        if perf_counter() - started >= seconds:
            raise TimeoutError("LM wall-time cap reached")

    def evaluate(vector: np.ndarray) -> tuple[float, np.ndarray]:
        check_budget()
        statistics["evaluations"] += 1
        scalar, gradient = objective(vector)
        check_budget()
        if not np.isfinite(scalar) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite native objective/gradient")
        return float(scalar), np.asarray(gradient, dtype=np.float64)

    try:
        value, gradient = evaluate(x)
        curvature: Matvec | None = None
        for attempt in range(maxattempts):
            check_budget()
            if curvature is None:
                curvature = curvature_factory(x.copy())
                check_budget()

            def curvature_times(vector: np.ndarray) -> np.ndarray:
                check_budget()
                statistics["matvecs"] += 1
                result = np.asarray(cast(Matvec, curvature)(vector), dtype=np.float64)
                check_budget()
                if not np.isfinite(result).all():
                    raise FloatingPointError("Non-finite curvature product")
                return result

            def cg_iteration(_: np.ndarray) -> None:
                statistics["cg_iterations"] += 1
                check_budget()

            row: dict[str, Any] = {"attempt": attempt + 1, "damping": damping, "accepted": False}
            statistics["attempts"].append(row)
            try:
                operator = cast(Any, LinearOperator)(
                    (x.size, x.size), matvec=lambda v: curvature_times(v) + damping * v,
                    dtype=np.float64,
                )
                step, info = cg(operator, -gradient, rtol=cg_rtol, atol=0.0,
                                maxiter=maxcg, callback=cg_iteration)
                row["cg_info"] = int(info)
                if info < 0 or not np.isfinite(step).all():
                    raise FloatingPointError("CG numerical failure")
                trial = np.asarray(round_parameters(x + step), dtype=np.float64)
                displacement = trial - x
                row["rounded_step_norm"] = float(np.linalg.norm(displacement))
                if not np.isfinite(trial).all():
                    raise FloatingPointError("Non-finite rounded proposal")
                if not np.any(displacement):
                    statistics["termination"] = "No representable rounded update"
                    break
                predicted = float(-gradient @ displacement - 0.5 * displacement @ curvature_times(displacement))
                row["predicted_reduction"] = predicted
                if not np.isfinite(predicted) or predicted <= 0:
                    raise FloatingPointError("Nonpositive predicted reduction")
                trial_value, trial_gradient = evaluate(trial)
                actual = value - trial_value
                ratio = actual / predicted
                row.update(objective=trial_value, actual_reduction=actual, trust_ratio=ratio)
            except FloatingPointError as error:
                row["reason"] = str(error)
            else:
                if actual > 0 and np.isfinite(ratio) and ratio >= accept_ratio:
                    check_budget()
                    on_accept(trial.copy(), trial_value, statistics["evaluations"])
                    x, value, gradient = trial, trial_value, trial_gradient
                    statistics["accepted_steps"] += 1
                    row["accepted"] = True
                    damping *= max(1.0 / 3.0, 1.0 - (2.0 * min(ratio, 1.0) - 1.0)**3)
                    rejection_factor, curvature = 2.0, None
                    if statistics["accepted_steps"] >= maxaccepted:
                        statistics["termination"] = "LM accepted-step cap reached"
                        break
                    continue
                row["reason"] = "Native decrease/trust-ratio test failed"
            damping *= rejection_factor
            rejection_factor *= 2.0
            if not np.isfinite(damping):
                statistics["termination"] = "Damping overflow after rejected trials"
                break
        else:
            statistics["termination"] = "LM attempt cap reached"
    except (TimeoutError, FloatingPointError) as error:
        statistics["termination"] = str(error)
    except KeyboardInterrupt:
        statistics.update(termination="Interrupted; retained last accepted iterate", interrupted=True)
    statistics.update(accepted_objective=value, final_damping=damping,
                      optimization_seconds=perf_counter() - started)
    return statistics


def save_iterate(
    run_dir: Path, params: Any, iteration: int, source: LoadedRun,
    kind: Literal["raw", "weight_ema"],
) -> None:
    destination = run_dir / ("weight_ema_ckpt" if kind == "weight_ema" else "latest_ckpt")
    destination.mkdir()
    checkpoints.save_checkpoint(
        destination.resolve(), jax.device_get({"params": params, "step": iteration}),
        step=source.epoch, prefix="ckpt_", orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    # Fine-tuning iterations are not additional full-data training epochs.
    (destination / "metadata.json").write_text(json.dumps(
        {"epoch": source.epoch, "fine_tune_iteration": iteration,
         "stats": source.stats, "params_kind": kind}, indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, required=True,
                        help="Explicitly approved balanced TRAIN bundle; repeated objective evaluations reuse it.")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--optimizer", choices=("lbfgs", "lm"), default="lbfgs")
    parser.add_argument("--matmul-precision", choices=("default", "highest"), default="highest",
                        help="Refinement arithmetic only; frozen teachers and deployed validation use DEFAULT.")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--max-line-search", type=int, default=8)
    parser.add_argument("--max-cg", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--initial-damping", type=float, default=1e-3)
    parser.add_argument("--seconds", type=float, default=900.0,
                        help="Optimizer wall-time cap; preparation, full validation and rollouts are separate.")
    parser.add_argument("--check-only", action="store_true",
                        help="Check repeatability and derivatives without any parameter updates or saved checkpoint.")
    args = parser.parse_args()
    if args.train_rows <= 0 or args.train_rows % 256:
        parser.error("--train-rows must be a positive multiple of 256 (balanced B16 batches, every-16 cadence)")
    if (not np.isfinite(args.seconds) or args.seconds <= 0 or args.maxiter <= 0 or args.max_line_search <= 0
            or args.max_cg <= 0 or args.max_attempts <= 0
            or not np.isfinite(args.initial_damping) or args.initial_damping <= 0):
        parser.error("Iteration, line-search, CG, damping, and wall-time budgets must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "default")
    loaded = load_run(args.run_dir, checkpoint="best_weight_ema", checkpoint_dir=args.checkpoint_dir)
    config = trainer.build_parser().parse_args(["--dataset", loaded.config["dataset"]])
    vars(config).update(loaded.config)
    if args.optimizer == "lm" and (
        config.data_phase_projection_weight or config.data_xi_projection_weight
        or config.data_modal_residual_floor or config.rollout_training_rhs
        or config.translation_tangent_geometry != "local" or config.hadamard_mode != "self"
    ):
        parser.error("LM residuals require plain relative data loss, local tangent geometry, and self Hadamard loss")
    dataset = load_dataset_arrays(ROOT / config.dataset)
    families = np.load(ROOT / config.dataset / "family_id.npy", mmap_mode="r")
    train_ids, validation_ids, _ = build_dataset_split_indices(dataset)
    rng = np.random.default_rng(args.seed)
    selected = np.stack([
        rng.choice(train_ids[families[train_ids] == family], args.train_rows // 4, replace=False)
        for family in (1, 2, 3, 4)
    ], axis=1).reshape(-1)
    np.save(args.output_dir / "train_rows.npy", selected)
    nx = dataset["eta"].shape[1]
    length = float(config.domain_length)
    apply_model = trainer.build_apply_model(config, loaded.model, nx, length)
    components, hadamard = trainer.build_loss_functions(
        config, loaded.model, apply_model, nx, length, loaded.stats, jnp.float32, jnp.float32, 1,
    )
    normalize, _, denormalize = make_normalizers(loaded.stats, config.norm, dtype=np.dtype(np.float32))
    coefficients = jnp.asarray([1.0, config.mode_balanced_weight,
                                config.translation_tangent_weight, config.phase_mechanism_weight])

    @jax.jit
    def generate_teacher(batch: Batch) -> Batch:
        eta, xi, target, log_depth, family = (field[:config.rollout_training_microbatch] for field in batch)
        depth = jnp.exp(log_depth.astype(jnp.float64))

        def predict(eta_phys: jax.Array, xi_phys: jax.Array) -> jax.Array:
            prediction = apply_model(
                {"params": loaded.params}, normalize(eta_phys.astype(jnp.float32), xi_phys.astype(jnp.float32)),
                log_depth.astype(jnp.float32),
            )
            physical = denormalize(prediction)[..., 0].astype(jnp.float64)
            return physical - jnp.mean(physical, axis=-1, keepdims=True)

        trajectory = rollout_surrogate(
            ti.State(eta.astype(jnp.float64), xi.astype(jnp.float64)),
            jnp.array([0.0, 0.01 * config.rollout_training_steps], dtype=jnp.float64),
            ti.make_solver_params(nx, length, depth, filter_fraction=min(1.0, 128.0 * length / (jnp.pi * nx))),
            predict, substeps=config.rollout_training_steps,
        )
        eta_new, xi_new = trajectory["eta"][-1], trajectory["xi"][-1]
        valid = jnp.all(jnp.isfinite(eta_new) & jnp.isfinite(xi_new) & (eta_new + depth > 0), axis=-1)
        fields = compute_dno_target(
            jnp.where(valid[:, None], eta_new, eta), jnp.where(valid[:, None], xi_new, xi), depth,
            nx=nx, length=length, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
        )
        valid &= jnp.all(jnp.isfinite(jnp.stack(fields)), axis=(0, 2))
        valid &= jnp.all(fields[0] + depth > 0, axis=-1)
        converted = tuple(field.astype(original.dtype) for field, original in zip(fields, (eta, xi, target)))
        valid &= jnp.all(jnp.isfinite(jnp.stack(converted)), axis=(0, 2))
        return (*(cast(jax.Array, jnp.where(valid[:, None], field, original))
                  for field, original in zip(converted, (eta, xi, target))), log_depth, family, valid)

    def batch_loss(params: Any, batch: Batch, teacher: Batch, key: jax.Array,
                   *, hd_active: bool, teacher_active: bool) -> jax.Array:
        clean = jax.vmap(lambda _: components(params, *batch), axis_name="batch")(jnp.arange(1))
        data, mode, tangent, mechanism = (field[0] for field in clean[:4])
        physics = jnp.float32(0.0)
        mode_weight = jnp.float32(config.mode_balanced_weight)
        if teacher_active:
            generated = jax.vmap(lambda _: components(params, *teacher[:5]), axis_name="batch")(jnp.arange(1))
            generated_data, generated_mode, generated_tangent, generated_mechanism = (
                field[0].astype(jnp.float64) for field in generated[:4]
            )
            data += config.rollout_training_weight * generated_data
            physics += config.rollout_training_weight * (
                mode_weight * generated_mode + config.translation_tangent_weight * generated_tangent
                + config.phase_mechanism_weight * generated_mechanism
            )
        physics += mode_weight * mode
        physics += config.translation_tangent_weight * tangent
        physics += config.phase_mechanism_weight * mechanism
        if hd_active:
            physics += jnp.float32(config.hadamard_weight * hadamard(key, params, batch[0], batch[1], batch[3]))
        return data + physics

    loss_grad = jax.jit(jax.value_and_grad(batch_loss), static_argnames=("hd_active", "teacher_active"))
    batches: list[tuple[Batch, Batch, jax.Array, bool, bool]] = []
    preparation_start = perf_counter()
    for index, (eta, xi, target, depth, ids) in enumerate(get_batches(
        dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"], selected, 16, None,
    )):
        batch = tuple(map(jnp.asarray, (eta, xi, target, depth, families[ids])))
        hd_active = index % config.hadamard_interval == 0
        teacher_active = index % config.rollout_training_interval == 0
        teacher = generate_teacher(batch) if teacher_active else (*batch, jnp.ones(16, dtype=bool))
        jax.block_until_ready(teacher)
        if teacher_active and not bool(jnp.all(teacher[-1])):
            raise FloatingPointError("The fixed teacher bundle contains failed trajectories or labels")
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), index)
        batches.append((batch, teacher, key, hd_active, teacher_active))
    # Keep the teacher trajectory fixed under its original deployment arithmetic.
    # Higher-accuracy dot products address the observed AD/finite-difference mismatch.
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    flat_initial, unravel = ravel_pytree(loaded.params)
    x_initial = np.asarray(flat_initial, dtype=np.float64)

    @jax.jit(static_argnames=("hd_active", "teacher_active"))
    def batch_curvature(params: Any, tangent: Any, batch: Batch, teacher: Batch, key: jax.Array,
                        *, hd_active: bool, teacher_active: bool) -> Any:
        return normal_product(
            lambda p: batch_residuals(
                p, batch, teacher, key, components=components, hadamard=hadamard,
                coefficients=coefficients, teacher_weight=config.rollout_training_weight,
                hadamard_weight=config.hadamard_weight, hd_active=hd_active, teacher_active=teacher_active,
            ), params, tangent,
        )

    def curvature_factory(vector: np.ndarray) -> Matvec:
        params = unravel(jnp.asarray(vector, dtype=flat_initial.dtype))

        def multiply(direction: np.ndarray) -> np.ndarray:
            tangent = unravel(jnp.asarray(direction, dtype=flat_initial.dtype))
            total = np.zeros_like(direction)
            for batch, teacher, key, hd_active, teacher_active in batches:
                product = batch_curvature(params, tangent, batch, teacher, key,
                                          hd_active=hd_active, teacher_active=teacher_active)
                total += np.asarray(ravel_pytree(product)[0], dtype=np.float64)
            return total / len(batches)

        return multiply

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        params = unravel(jnp.asarray(vector, dtype=flat_initial.dtype))
        value = 0.0
        gradient = np.zeros_like(vector)
        for batch, teacher, key, hd_active, teacher_active in batches:
            scalar, tree_grad = loss_grad(params, batch, teacher, key,
                                         hd_active=hd_active, teacher_active=teacher_active)
            flat_grad, _ = ravel_pytree(tree_grad)
            value += float(scalar)
            gradient += np.asarray(flat_grad, dtype=np.float64)
        value /= len(batches)
        gradient /= len(batches)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite native objective/gradient")
        return value, gradient

    initial_value, initial_gradient = objective(x_initial)
    check_start = perf_counter()
    repeated_value, repeated_gradient = objective(x_initial)
    seconds_per_evaluation = perf_counter() - check_start
    if repeated_value != initial_value or not np.array_equal(initial_gradient, repeated_gradient):
        raise RuntimeError("Line-search objective is not deterministic")
    curvature_seconds = None
    if args.optimizer == "lm":
        initial_curvature = curvature_factory(x_initial)
        direction = initial_gradient / max(np.linalg.norm(initial_gradient), np.finfo(float).tiny)
        if not np.isfinite(initial_curvature(direction)).all():
            raise FloatingPointError("Non-finite initial curvature product")
        curvature_start = perf_counter()
        product = initial_curvature(direction)
        curvature_seconds = perf_counter() - curvature_start
        if not np.isfinite(product).all() or direction @ product < 0:
            raise FloatingPointError("Initial curvature is not finite positive semidefinite")
    summary: dict[str, Any] = {
        "optimizer": "IRLS Levenberg-Marquardt" if args.optimizer == "lm" else "L-BFGS-B (unbounded)",
        "matmul_precision": args.matmul_precision, "validation_matmul_precision": "default",
        "initial_checkpoint": str(args.checkpoint_dir),
        "initial_params_kind": "weight_ema", "initial_epoch": loaded.epoch,
        "train_rows": int(selected.size), "validation_rows": int(validation_ids.size),
        "initial_objective": initial_value, "gradient_norm": float(np.linalg.norm(initial_gradient)),
        "seconds_per_objective_gradient": seconds_per_evaluation,
        "seconds_per_curvature_product": curvature_seconds,
        "preparation_and_compilation_seconds": perf_counter() - preparation_start,
        "gradient_ema": False, "weight_ema_decay": config.weight_ema_decay,
        "teacher": "fixed initial-EMA trajectories; full loss; original every-16-batch cadence",
        "weight_decay": "No decoupled AdamW shrinkage is applied outside the acceptance objective",
    }
    print(json.dumps(summary), flush=True)
    if args.check_only:
        random_direction = rng.normal(size=x_initial.size)
        random_direction /= np.linalg.norm(random_direction)
        checks = {}
        for name, direction in (("random", random_direction),
                                ("negative_gradient", -initial_gradient / np.linalg.norm(initial_gradient))):
            differences = []
            for epsilon in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
                forward_point = (x_initial + epsilon * direction).astype(np.float32).astype(np.float64)
                backward_point = (x_initial - epsilon * direction).astype(np.float32).astype(np.float64)
                forward, _ = objective(forward_point)
                backward, _ = objective(backward_point)
                differences.append({"epsilon": epsilon, "central_difference": (forward - backward) / (2 * epsilon),
                                    "rounded_secant_derivative": float(
                                        initial_gradient @ (forward_point - backward_point) / (2 * epsilon)),
                                    "forward_objective": forward, "backward_objective": backward})
            checks[name] = {"directional_derivative": float(initial_gradient @ direction),
                            "finite_differences": differences}
        summary.update(check_only=True, gradient_checks=checks)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return

    # All candidate steps update the complete pytree. Only accepted iterates enter
    # the independent weight EMA; second-order gradients are never smoothed.
    accepted = x_initial.copy()
    shadow = loaded.params
    accepted_steps = 0
    accepted_value = initial_value
    optimization_start = perf_counter()

    def accept(vector: np.ndarray, value: float, evaluations: int) -> None:
        nonlocal accepted, shadow, accepted_steps, accepted_value
        accepted = vector
        accepted_value = value
        shadow = optax.incremental_update(unravel(jnp.asarray(accepted, dtype=flat_initial.dtype)),
                                          shadow, 1.0 - config.weight_ema_decay)
        accepted_steps += 1
        record = {"iteration": accepted_steps, "objective": accepted_value,
                  "evaluations": evaluations, "seconds": perf_counter() - optimization_start}
        with (args.output_dir / "iterations.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    previous_term_handler = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        if args.optimizer == "lm":
            optimization = run_lm(
                objective, curvature_factory, lambda x: x.astype(np.float32).astype(np.float64),
                x_initial, accept, maxaccepted=args.maxiter, maxcg=args.max_cg,
                maxattempts=args.max_attempts, seconds=args.seconds, initial_damping=args.initial_damping,
            )
        else:
            optimization = run_lbfgs(objective, x_initial, initial_value, accept, maxiter=args.maxiter,
                                     maxls=args.max_line_search, seconds=args.seconds)
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
    applied = np.asarray(jnp.asarray(accepted, dtype=flat_initial.dtype), dtype=np.float64)
    summary.update(optimization)
    summary.update(accepted_steps=accepted_steps, accepted_objective=accepted_value,
                   fp32_parameter_change_norm=float(np.linalg.norm(applied - x_initial)),
                   weight_ema_initial_coefficient=config.weight_ema_decay**accepted_steps)
    config_payload = vars(config) | {"run_name": args.output_dir.name, "gradient_ema_decay": 0.0,
                                     "fine_tune_optimizer": summary["optimizer"], "weight_decay": 0.0,
                                     "fine_tune_matmul_precision": args.matmul_precision,
                                     "fine_tune_iterations": accepted_steps, "epochs": loaded.epoch,
                                     "fine_tune_train_rows": args.train_rows}
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    save_iterate(args.output_dir, unravel(jnp.asarray(accepted, dtype=flat_initial.dtype)), accepted_steps, loaded, "raw")
    save_iterate(args.output_dir, shadow, accepted_steps, loaded, "weight_ema")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if optimization["interrupted"]:
        print(json.dumps(summary), flush=True)
        return

    jax.config.update("jax_default_matmul_precision", "default")

    @jax.jit(static_argnames=("hd_active",))
    def validation_metrics(params: Any, batch: Batch, key: jax.Array, *, hd_active: bool) -> jax.Array:
        values = jax.vmap(lambda _: components(params, *batch, collect_family_metrics=True),
                          axis_name="batch")(jnp.arange(1))
        hd = hadamard(key, params, batch[0], batch[1], batch[3]) if hd_active else jnp.float64(0.0)
        return jnp.concatenate((jnp.stack([field[0] for field in values[:6]]), hd[None],
                                jnp.stack([field[0] for field in values[6]])))

    # Match the trainer's full validation aggregation: TG is selected-row weighted,
    # HD is logged separately on sampled batches and is not in the selection score.
    for name, params in (("initial", loaded.params), ("accepted", unravel(jnp.asarray(accepted, dtype=flat_initial.dtype))),
                         ("weight_ema", shadow)):
        totals = np.zeros(27, dtype=np.float64)
        tangent_sum = tangent_count = hd_count = 0.0
        for index, (eta, xi, target, depth, ids) in enumerate(get_batches(
            dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"], validation_ids, 16, None,
        )):
            batch = tuple(map(jnp.asarray, (eta, xi, target, depth, families[ids])))
            hd_active = index % config.val_hadamard_interval == 0
            values = np.asarray(validation_metrics(
                params, batch, jax.random.fold_in(
                    jax.random.fold_in(jax.random.PRNGKey(config.seed + 2), index), 0,
                ), hd_active=hd_active,
            ))
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite {name} validation metrics in batch {index}")
            totals += values * len(ids)
            tangent_sum += values[2] * values[4] * len(ids)
            tangent_count += values[4] * len(ids)
            hd_count += len(ids) if hd_active else 0
        means = totals / validation_ids.size
        means[2] = tangent_sum / max(tangent_count, 1.0)
        means[6] = totals[6] / hd_count
        summary[f"validation_{name}"] = {
            "val_loss": float(np.dot(np.asarray(coefficients), means[:4])),
            "val_l2_loss": float(means[5]), "val_hadamard_loss": float(means[6]),
            "families": {
                family: {"relative_l2": float(row[1] / row[0]),
                         "baseline_relative": float(row[2] / row[0]), "baseline_win_fraction": float(row[3] / row[0])}
                for family, row in zip(("stokes", "tanaka", "benjamin_feir", "jonswap_tma"), totals[7:].reshape(4, 5))
            },
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({name: summary[f"validation_{name}"]}), flush=True)


if __name__ == "__main__":
    main()
