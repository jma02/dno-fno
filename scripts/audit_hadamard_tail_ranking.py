"""Rank C25 rollout tails by held-out Hadamard shape defect.

The audit evaluates the production finite-secant regularizer on archived truth
states without advancing a trajectory or differentiating with respect to model
parameters.  Each selected state is repeated over fixed random probes, and the
same probe keys are reused across cases at the same sampled time index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import hypergeom, mannwhitneyu, pearsonr, rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "train-jax-10m"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for directory in (REPO_ROOT, TRAIN_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from audit_teacher_forced_gxi import (  # noqa: E402
    CaseTrajectory,
    directory_sha256,
    discover_all_requests,
    load_cases,
    low_pass,
    relative_l2_series,
)
from compare_paired_translation_errors import archive_metrics  # noqa: E402
from hadamard_shape_regularizer import (  # noqa: E402
    HadamardRegConfig,
    compute_hadamard_reg,
)
from solver.evals.model_rollout import (  # noqa: E402
    LoadedRun,
    build_predict_gxi_batched,
    load_run,
)
from solver.solvers.dno_series_jax import build_grid  # noqa: E402


Array: TypeAlias = jax.Array
SampleEvaluator: TypeAlias = Callable[
    [Array, Array, Array, Array], tuple[Array, Array, Array, Array]
]
Predictor: TypeAlias = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class CaseHadamardResult:
    """Probe-resolved Hadamard values and a strict-JSON case summary."""

    summary: dict[str, Any]
    frame_indices: np.ndarray
    times: np.ndarray
    loss: np.ndarray
    defect: np.ndarray
    residual_h1: np.ndarray
    forcing_h1: np.ndarray
    teacher_q_h1: np.ndarray
    teacher_q_l2: np.ndarray
    teacher_q_p8_l2: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--path-frames", type=int, default=17)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument("--tail-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series-output", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "path frames": args.path_frames,
        "probes": args.probes,
        "batch size": args.batch_size,
        "length": args.length,
        "tail threshold": args.tail_threshold,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"all audit controls must be positive: {invalid}")
    if args.probes < 2:
        raise ValueError("at least two probes are required for a Monte Carlo error bar")


def select_frame_indices(n_frames: int, requested: int) -> np.ndarray:
    """Return unique, evenly spaced frame indices including both endpoints."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if requested < 1:
        raise ValueError(f"requested must be positive, got {requested}")
    count = min(n_frames, requested)
    if count == 1:
        return np.asarray([n_frames - 1], dtype=np.int64)
    indices = np.rint(np.linspace(0, n_frames - 1, count)).astype(np.int64)
    return np.unique(indices)


def make_normalizers(
    loaded: LoadedRun,
) -> tuple[
    Callable[[Array, Array], Array],
    Callable[[Array], Array],
]:
    """Recreate the trainer's input and target normalization maps."""
    if loaded.norm_mode == "scale":
        feature_scale = jnp.asarray(
            loaded.stats["feature_absmax"], dtype=jnp.float32
        )
        target_scale = jnp.asarray(
            loaded.stats["target_absmax"], dtype=jnp.float32
        )

        def normalize_inputs(eta: Array, xi: Array) -> Array:
            return jnp.stack((eta, xi), axis=-1) / feature_scale

        def denormalize_targets(values: Array) -> Array:
            return values * target_scale

    else:
        feature_min = jnp.asarray(loaded.stats["feature_min"], dtype=jnp.float32)
        feature_range = jnp.asarray(
            np.asarray(loaded.stats["feature_max"])
            - np.asarray(loaded.stats["feature_min"])
            + 1e-8,
            dtype=jnp.float32,
        )
        target_min = jnp.asarray(loaded.stats["target_min"], dtype=jnp.float32)
        target_range = jnp.asarray(
            float(loaded.stats["target_max"])
            - float(loaded.stats["target_min"])
            + 1e-8,
            dtype=jnp.float32,
        )

        def normalize_inputs(eta: Array, xi: Array) -> Array:
            stacked = jnp.stack((eta, xi), axis=-1)
            return ((stacked - feature_min) / feature_range) * 2.0 - 1.0

        def denormalize_targets(values: Array) -> Array:
            return ((values + 1.0) * 0.5) * target_range + target_min

    filter_fraction = float(loaded.config.get("filter_gxi_fraction", 1.0))
    limiter_enabled = bool(loaded.config.get("gxi_highband_limiter", False))
    if filter_fraction != 1.0 or limiter_enabled:
        raise ValueError(
            "this audit currently requires the identity production prediction "
            "filter; checkpoint config enables a limiter or filter"
        )

    return normalize_inputs, denormalize_targets


def build_sample_evaluator(
    loaded: LoadedRun,
    nx: int,
    length: float,
) -> tuple[SampleEvaluator, HadamardRegConfig]:
    """Vmap the exact production Hadamard loss over singleton states."""
    normalize_inputs, denormalize_targets = make_normalizers(loaded)
    _, wave_numbers = build_grid(nx, length)
    config = HadamardRegConfig(
        k_max=float(loaded.config["hadamard_k_max"]),
        sobolev_order=int(loaded.config["hadamard_sobolev_order"]),
        relative_eps_min=float(loaded.config["hadamard_relative_eps_min"]),
        relative_eps_max=float(loaded.config["hadamard_relative_eps_max"]),
        eta_scale_floor=float(loaded.config["hadamard_eta_scale_floor"]),
        denominator_floor=float(loaded.config["hadamard_denominator_floor"]),
    )
    wave_numbers = jnp.asarray(wave_numbers, dtype=jnp.float64)

    def evaluate_one(
        key: Array,
        eta: Array,
        xi: Array,
        log_depth: Array,
    ) -> tuple[Array, Array, Array, Array]:
        loss, diagnostics = compute_hadamard_reg(
            rng=key,
            apply_fn=loaded.model.apply,
            model_params=loaded.params,
            eta_phys=eta[None, :],
            xi_phys=xi[None, :],
            batch_depth_local=log_depth.reshape(1, 1),
            norm_inputs_fn=normalize_inputs,
            denorm_targets_fn=denormalize_targets,
            k=wave_numbers,
            cfg=config,
            dtype=jnp.float64,
        )
        return (
            loss,
            diagnostics["hadamard_defect_rms"],
            diagnostics["hadamard_residual_hs_rms"],
            diagnostics["hadamard_forcing_hs_rms"],
        )

    return jax.jit(jax.vmap(evaluate_one)), config


def common_probe_keys(seed: int, n_frames: int, probes: int) -> np.ndarray:
    """Build common random keys indexed only by sampled frame and probe."""
    base_key = jax.random.PRNGKey(seed)
    indices = jnp.arange(n_frames * probes, dtype=jnp.uint32)
    keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(indices)
    return np.asarray(jax.device_get(keys), dtype=np.uint32)


def evaluate_in_fixed_batches(
    evaluator: SampleEvaluator,
    keys: np.ndarray,
    eta: np.ndarray,
    xi: np.ndarray,
    log_depth: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate identically shaped padded batches and discard padding."""
    n_samples = eta.shape[0]
    outputs = tuple(np.empty(n_samples, dtype=np.float64) for _ in range(4))
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        count = stop - start
        key_batch = np.empty((batch_size, 2), dtype=np.uint32)
        eta_batch = np.empty((batch_size, eta.shape[-1]), dtype=np.float32)
        xi_batch = np.empty_like(eta_batch)
        depth_batch = np.empty((batch_size,), dtype=np.float64)
        key_batch[:count] = keys[start:stop]
        eta_batch[:count] = eta[start:stop]
        xi_batch[:count] = xi[start:stop]
        depth_batch[:count] = log_depth[start:stop]
        if count < batch_size:
            key_batch[count:] = key_batch[count - 1]
            eta_batch[count:] = eta_batch[count - 1]
            xi_batch[count:] = xi_batch[count - 1]
            depth_batch[count:] = depth_batch[count - 1]
        batch_outputs = evaluator(
            jnp.asarray(key_batch),
            jnp.asarray(eta_batch),
            jnp.asarray(xi_batch),
            jnp.asarray(depth_batch),
        )
        host_outputs = tuple(
            np.asarray(jax.device_get(values), dtype=np.float64)[:count]
            for values in batch_outputs
        )
        for destination, values in zip(outputs, host_outputs, strict=True):
            destination[start:stop] = values
    return outputs


def predict_selected_truth_states(
    predictor: Predictor,
    case: CaseTrajectory,
    frame_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluate production q on selected truth states with fixed padded batches."""
    eta = case.truth_eta[frame_indices]
    xi = case.truth_xi[frame_indices]
    prediction = np.empty_like(eta)
    log_depth = np.float32(np.log(case.depth))
    for start in range(0, frame_indices.size, batch_size):
        stop = min(start + batch_size, frame_indices.size)
        count = stop - start
        eta_batch = np.empty((batch_size, eta.shape[-1]), dtype=np.float32)
        xi_batch = np.empty_like(eta_batch)
        eta_batch[:count] = eta[start:stop]
        xi_batch[:count] = xi[start:stop]
        if count < batch_size:
            eta_batch[count:] = eta_batch[count - 1]
            xi_batch[count:] = xi_batch[count - 1]
        values = predictor(
            jnp.asarray(eta_batch),
            jnp.asarray(xi_batch),
            jnp.full((batch_size,), log_depth, dtype=jnp.float32),
        )
        prediction[start:stop] = np.asarray(jax.device_get(values))[:count]
    return prediction


def relative_h1_series(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match the trainer's per-sample relative H1 Fourier loss."""
    prediction_hat = np.fft.rfft(prediction, axis=-1)
    target_hat = np.fft.rfft(target, axis=-1)
    wave = np.arange(prediction_hat.shape[-1], dtype=np.float64)
    weight_squared = 1.0 + wave**2
    numerator = np.sqrt(
        np.sum(weight_squared[None, :] * np.abs(prediction_hat - target_hat) ** 2, axis=-1)
    )
    denominator = np.sqrt(
        np.maximum(
            np.sum(weight_squared[None, :] * np.abs(target_hat) ** 2, axis=-1),
            1e-12,
        )
    )
    return numerator / denominator


def scalar_path_summary(frame_values: np.ndarray) -> dict[str, float]:
    """Summarize a finite scalar over sampled trajectory frames."""
    return {
        "mean": float(np.mean(frame_values)),
        "median": float(np.median(frame_values)),
        "p95": float(np.percentile(frame_values, 95)),
        "maximum": float(np.max(frame_values)),
        "initial": float(frame_values[0]),
        "terminal": float(frame_values[-1]),
    }


def monte_carlo_standard_error(values: np.ndarray) -> float:
    """Estimate probe Monte Carlo error after averaging over frames."""
    probe_means = np.mean(values, axis=0)
    return float(np.std(probe_means, ddof=1) / np.sqrt(probe_means.size))


def evaluate_case(
    evaluator: SampleEvaluator,
    predictor: Predictor,
    case: CaseTrajectory,
    frame_indices: np.ndarray,
    keys: np.ndarray,
    probes: int,
    batch_size: int,
) -> CaseHadamardResult:
    """Evaluate fixed probes on evenly sampled truth states for one case."""
    eta = np.repeat(case.truth_eta[frame_indices], probes, axis=0)
    xi = np.repeat(case.truth_xi[frame_indices], probes, axis=0)
    log_depth = np.full(eta.shape[0], np.log(case.depth), dtype=np.float64)
    loss, defect, residual, forcing = evaluate_in_fixed_batches(
        evaluator,
        keys,
        eta,
        xi,
        log_depth,
        batch_size,
    )
    shape = (frame_indices.size, probes)
    loss = loss.reshape(shape)
    defect = defect.reshape(shape)
    residual = residual.reshape(shape)
    forcing = forcing.reshape(shape)
    mean_loss = np.mean(loss, axis=1)
    mean_defect = np.mean(defect, axis=1)
    mean_residual = np.mean(residual, axis=1)
    mean_forcing = np.mean(forcing, axis=1)
    teacher_prediction = predict_selected_truth_states(
        predictor, case, frame_indices, batch_size
    )
    teacher_q_h1 = relative_h1_series(
        teacher_prediction,
        np.asarray(case.truth_gxi[frame_indices], dtype=np.float64),
    )
    teacher_truth = np.asarray(case.truth_gxi[frame_indices], dtype=np.float64)
    teacher_error = np.asarray(teacher_prediction, dtype=np.float64) - teacher_truth
    teacher_q_l2 = relative_l2_series(teacher_error, teacher_truth)
    teacher_q_p8_l2 = relative_l2_series(
        low_pass(teacher_error, 2.0 * np.pi, 8.0),
        low_pass(teacher_truth, 2.0 * np.pi, 8.0),
    )
    summary = {
        "regime": case.regime,
        "case_index": case.case_index,
        "case_id": case.case_id,
        "depth": case.depth,
        "n_path_frames": int(frame_indices.size),
        "n_probes_per_frame": probes,
        "hadamard_loss": scalar_path_summary(mean_loss),
        "hadamard_defect": scalar_path_summary(mean_defect),
        "residual_h1": scalar_path_summary(mean_residual),
        "forcing_h1": scalar_path_summary(mean_forcing),
        "teacher_q_relative_h1": scalar_path_summary(teacher_q_h1),
        "teacher_q_relative_l2": scalar_path_summary(teacher_q_l2),
        "teacher_q_p8_relative_l2": scalar_path_summary(teacher_q_p8_l2),
        "path_defect_probe_standard_error": monte_carlo_standard_error(defect),
        "terminal_defect_probe_standard_error": float(
            np.std(defect[-1], ddof=1) / np.sqrt(probes)
        ),
    }
    return CaseHadamardResult(
        summary=summary,
        frame_indices=frame_indices,
        times=case.times[frame_indices],
        loss=loss,
        defect=defect,
        residual_h1=residual,
        forcing_h1=forcing,
        teacher_q_h1=teacher_q_h1,
        teacher_q_l2=teacher_q_l2,
        teacher_q_p8_l2=teacher_q_p8_l2,
    )


def spearman_summary(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return a one-sided rank-correlation screen for increasing association."""
    statistic = spearmanr(score, target, alternative="greater")
    return {"rho": float(statistic.statistic), "p_greater": float(statistic.pvalue)}


def top_k_recall(
    score: np.ndarray,
    tail: np.ndarray,
    budgets: Sequence[int],
) -> dict[str, dict[str, float | int]]:
    """Return tail recall and chance enrichment at fixed review budgets."""
    order = np.argsort(-score, kind="stable")
    population = score.size
    n_tail = int(np.count_nonzero(tail))
    result: dict[str, dict[str, float | int]] = {}
    for requested in budgets:
        budget = min(requested, population)
        hits = int(np.count_nonzero(tail[order[:budget]]))
        result[str(budget)] = {
            "hits": hits,
            "tail_count": n_tail,
            "recall": float(hits / n_tail),
            "chance_p_at_least_hits": float(
                hypergeom.sf(hits - 1, population, n_tail, budget)
            ),
        }
    return result


def tail_rank_test(
    score: np.ndarray,
    tail: np.ndarray,
    case_labels: Sequence[str],
) -> dict[str, Any]:
    """Test whether raw-error tails receive unusually high defect scores."""
    n_tail = int(np.count_nonzero(tail))
    if n_tail == 0 or n_tail == score.size:
        raise ValueError("tail test requires both tail and non-tail cases")
    ranks = rankdata(-score, method="average")
    test = mannwhitneyu(
        score[tail], score[~tail], alternative="greater", method="exact"
    )
    tail_indices = np.flatnonzero(tail)
    ordered_tail = tail_indices[np.argsort(ranks[tail_indices])]
    budgets = sorted({n_tail, 10, 16, score.size // 2})
    return {
        "tail_count": n_tail,
        "non_tail_count": int(score.size - n_tail),
        "mann_whitney_u": float(test.statistic),
        "roc_auc": float(test.statistic / (n_tail * (score.size - n_tail))),
        "p_greater": float(test.pvalue),
        "median_tail_rank": float(np.median(ranks[tail])),
        "median_non_tail_rank": float(np.median(ranks[~tail])),
        "top_k_recall": top_k_recall(score, tail, budgets),
        "tail_cases": [
            {
                "case": case_labels[index],
                "rank_high_to_low": float(ranks[index]),
                "score": float(score[index]),
            }
            for index in ordered_tail
        ],
    }


def split_half_stability(values: np.ndarray) -> dict[str, float]:
    """Measure case-ranking stability between the two probe halves."""
    probes = values.shape[-1]
    midpoint = probes // 2
    if midpoint == 0 or midpoint == probes:
        raise ValueError("split-half stability requires at least two probes")
    first = np.mean(values[..., :midpoint], axis=(1, 2))
    second = np.mean(values[..., midpoint:], axis=(1, 2))
    statistic = spearmanr(first, second)
    return {"rho": float(statistic.statistic), "p_two_sided": float(statistic.pvalue)}


def partial_rank_summary(
    score: np.ndarray,
    target: np.ndarray,
    covariates: Sequence[np.ndarray],
) -> dict[str, float]:
    """Residualize ranks on fixed covariates, then correlate the residuals."""
    design = np.column_stack(
        [np.ones(score.size)]
        + [rankdata(np.asarray(values)) for values in covariates]
    )

    def residual(values: np.ndarray) -> np.ndarray:
        ranked = rankdata(values)
        coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
        return ranked - design @ coefficients

    statistic = pearsonr(
        residual(score), residual(target), alternative="greater"
    )
    return {
        "partial_rho": float(statistic.statistic),
        "p_greater": float(statistic.pvalue),
    }


def stratified_partial_rank_summary(
    score: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Remove a rank control separately in each family before correlation."""
    score_residuals: list[np.ndarray] = []
    target_residuals: list[np.ndarray] = []
    for group in np.unique(groups):
        selected = groups == group
        design = np.column_stack(
            (np.ones(np.count_nonzero(selected)), rankdata(control[selected]))
        )
        for values, destination in (
            (score, score_residuals),
            (target, target_residuals),
        ):
            ranked = rankdata(values[selected])
            coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
            destination.append(ranked - design @ coefficients)
    statistic = pearsonr(
        np.concatenate(score_residuals),
        np.concatenate(target_residuals),
        alternative="greater",
    )
    return {
        "partial_rho": float(statistic.statistic),
        "p_greater": float(statistic.pvalue),
    }


def terminal_metrics(
    eval_dir: Path,
    regimes: Sequence[str],
    length: float,
) -> dict[tuple[str, int], dict[str, float | bool | int]]:
    """Load final raw, aligned, translation, and displacement metrics."""
    result: dict[tuple[str, int], dict[str, float | bool | int]] = {}
    for regime in regimes:
        archive = archive_metrics(
            eval_dir / regime / f"{regime}_trajs.npz", length
        )
        for index, case_id in enumerate(archive["case_ids"]):
            result[(regime, index)] = {
                "case_id": int(case_id),
                "valid": bool(archive["valid"][index]),
                "raw": float(archive["raw"][index]),
                "aligned": float(archive["aligned"][index]),
                "translation": float(archive["translation"][index]),
                "absolute_displacement": float(abs(archive["displacement"][index])),
            }
    return result


def correlation_table(
    scores: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Cross every Hadamard score with every terminal rollout target."""
    return {
        score_name: {
            target_name: spearman_summary(score, target)
            for target_name, target in targets.items()
        }
        for score_name, score in scores.items()
    }


def series_payload(
    cases: Sequence[CaseTrajectory],
    results: Sequence[CaseHadamardResult],
    terminal: Sequence[dict[str, float | bool | int]],
) -> dict[str, np.ndarray]:
    """Pack the fixed-size per-case probe arrays for independent reanalysis."""
    return {
        "case_regimes": np.asarray([case.regime for case in cases]),
        "case_indices": np.asarray([case.case_index for case in cases], np.int64),
        "case_ids": np.asarray([case.case_id for case in cases], np.int64),
        "depths": np.asarray([case.depth for case in cases], np.float64),
        "frame_indices": np.stack([item.frame_indices for item in results]),
        "times": np.stack([item.times for item in results]),
        "hadamard_loss": np.stack([item.loss for item in results]),
        "hadamard_defect": np.stack([item.defect for item in results]),
        "residual_h1": np.stack([item.residual_h1 for item in results]),
        "forcing_h1": np.stack([item.forcing_h1 for item in results]),
        "teacher_q_relative_h1": np.stack(
            [item.teacher_q_h1 for item in results]
        ),
        "teacher_q_relative_l2": np.stack(
            [item.teacher_q_l2 for item in results]
        ),
        "teacher_q_p8_relative_l2": np.stack(
            [item.teacher_q_p8_l2 for item in results]
        ),
        "terminal_raw": np.asarray([item["raw"] for item in terminal]),
        "terminal_aligned": np.asarray([item["aligned"] for item in terminal]),
        "terminal_translation": np.asarray(
            [item["translation"] for item in terminal]
        ),
        "terminal_absolute_displacement": np.asarray(
            [item["absolute_displacement"] for item in terminal]
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    eval_dir = args.eval_dir.resolve()
    cases = load_cases(eval_dir, discover_all_requests(eval_dir))
    frame_counts = {case.times.size for case in cases}
    if len(frame_counts) != 1:
        raise ValueError(f"all cases must have one frame count, got {frame_counts}")
    frame_indices = select_frame_indices(frame_counts.pop(), args.path_frames)
    nx = cases[0].truth_eta.shape[-1]

    print(f"loading {args.checkpoint} checkpoint from {args.run_dir.resolve()}")
    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    evaluator, hadamard_config = build_sample_evaluator(loaded, nx, args.length)
    predictor = build_predict_gxi_batched(loaded)
    keys = common_probe_keys(args.seed, frame_indices.size, args.probes)
    print(
        f"epoch={loaded.epoch}; backend={jax.default_backend()}; "
        f"devices={[str(device) for device in jax.devices()]}; "
        f"cases={len(cases)}, frames={frame_indices.size}, probes={args.probes}",
        flush=True,
    )

    results: list[CaseHadamardResult] = []
    for position, case in enumerate(cases, start=1):
        print(
            f"[{position:02d}/{len(cases)}] {case.regime}:{case.case_index} "
            f"h={case.depth:.8g}",
            flush=True,
        )
        results.append(
            evaluate_case(
                evaluator,
                predictor,
                case,
                frame_indices,
                keys,
                args.probes,
                args.batch_size,
            )
        )

    regimes = tuple(dict.fromkeys(case.regime for case in cases))
    terminal_by_case = terminal_metrics(eval_dir, regimes, args.length)
    terminal = [terminal_by_case[(case.regime, case.case_index)] for case in cases]
    if not all(bool(item["valid"]) for item in terminal):
        invalid = [
            f"{case.regime}:{case.case_index}"
            for case, item in zip(cases, terminal, strict=True)
            if not bool(item["valid"])
        ]
        raise ValueError(f"fixed-panel audit contains invalid cases: {invalid}")

    defect_cube = np.stack([item.defect for item in results])
    loss_cube = np.stack([item.loss for item in results])
    scores = {
        "path_mean_defect": np.mean(defect_cube, axis=(1, 2)),
        "path_p95_frame_defect": np.percentile(
            np.mean(defect_cube, axis=2), 95, axis=1
        ),
        "terminal_mean_defect": np.mean(defect_cube[:, -1, :], axis=1),
        "path_mean_loss": np.mean(loss_cube, axis=(1, 2)),
        "path_mean_teacher_q_h1": np.asarray(
            [np.mean(item.teacher_q_h1) for item in results]
        ),
        "terminal_teacher_q_h1": np.asarray(
            [item.teacher_q_h1[-1] for item in results]
        ),
        "path_mean_teacher_q_l2": np.asarray(
            [np.mean(item.teacher_q_l2) for item in results]
        ),
        "path_mean_teacher_q_p8_l2": np.asarray(
            [np.mean(item.teacher_q_p8_l2) for item in results]
        ),
    }
    targets = {
        "raw": np.asarray([float(item["raw"]) for item in terminal]),
        "aligned": np.asarray([float(item["aligned"]) for item in terminal]),
        "translation": np.asarray(
            [float(item["translation"]) for item in terminal]
        ),
        "absolute_displacement": np.asarray(
            [float(item["absolute_displacement"]) for item in terminal]
        ),
    }
    tail = targets["raw"] > args.tail_threshold
    labels = [f"{case.regime}:{case.case_index}" for case in cases]
    family_indicator = np.asarray(
        [int(case.regime == regimes[-1]) for case in cases], dtype=np.int64
    )
    depths = np.asarray([case.depth for case in cases], dtype=np.float64)
    primary_tail_test = tail_rank_test(scores["path_mean_defect"], tail, labels)
    terminal_tail_test = tail_rank_test(
        scores["terminal_mean_defect"], tail, labels
    )
    teacher_h1_tail_test = tail_rank_test(
        scores["path_mean_teacher_q_h1"], tail, labels
    )
    teacher_l2_tail_test = tail_rank_test(
        scores["path_mean_teacher_q_l2"], tail, labels
    )
    teacher_p8_tail_test = tail_rank_test(
        scores["path_mean_teacher_q_p8_l2"], tail, labels
    )
    family_correlations = {
        regime: correlation_table(
            {name: values[family_indicator == family_index] for name, values in scores.items()},
            {name: values[family_indicator == family_index] for name, values in targets.items()},
        )
        for family_index, regime in enumerate(regimes)
    }
    family_tail_tests = {
        regime: tail_rank_test(
            scores["path_mean_defect"][family_indicator == family_index],
            tail[family_indicator == family_index],
            np.asarray(labels)[family_indicator == family_index].tolist(),
        )
        for family_index, regime in enumerate(regimes)
    }
    partial_correlations = {
        target_name: {
            "controlling_family_and_depth": partial_rank_summary(
                scores["path_mean_defect"],
                target,
                (family_indicator, depths),
            ),
            "controlling_family_depth_and_teacher_q_h1": partial_rank_summary(
                scores["path_mean_defect"],
                target,
                (
                    family_indicator,
                    depths,
                    scores["path_mean_teacher_q_h1"],
                ),
            ),
            "controlling_within_family_teacher_q_l2": (
                stratified_partial_rank_summary(
                    scores["path_mean_defect"],
                    target,
                    scores["path_mean_teacher_q_l2"],
                    family_indicator,
                )
            ),
            "controlling_within_family_teacher_q_p8_l2": (
                stratified_partial_rank_summary(
                    scores["path_mean_defect"],
                    target,
                    scores["path_mean_teacher_q_p8_l2"],
                    family_indicator,
                )
            ),
        }
        for target_name, target in targets.items()
    }

    for case, item, rollout in zip(cases, results, terminal, strict=True):
        item.summary["terminal_rollout"] = rollout
        item.summary["raw_tail"] = bool(
            float(rollout["raw"]) > args.tail_threshold
        )

    output = args.output.resolve()
    series_output = (
        args.series_output.resolve()
        if args.series_output is not None
        else output.with_suffix(".npz")
    )
    checkpoint_dir = args.run_dir.resolve() / (
        "best_val_ckpt" if args.checkpoint == "best" else "final_ckpt"
    )
    result = {
        "provenance": {
            "schema_version": 1,
            "run_dir": str(args.run_dir.resolve()),
            "checkpoint_selection": args.checkpoint,
            "checkpoint_epoch": loaded.epoch,
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_sha256": directory_sha256(checkpoint_dir),
            "eval_dir": str(eval_dir),
            "series_output": str(series_output),
            "argv": sys.argv,
            "execution_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "JAX_PLATFORMS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                )
            },
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "seed": args.seed,
            "path_frame_indices": frame_indices.tolist(),
            "probes_per_frame": args.probes,
            "batch_size": args.batch_size,
            "common_random_numbers": (
                "probe key depends only on sampled-frame position and probe index"
            ),
            "hadamard_config": {
                "k_max": hadamard_config.k_max,
                "sobolev_order": hadamard_config.sobolev_order,
                "relative_eps_min": hadamard_config.relative_eps_min,
                "relative_eps_max": hadamard_config.relative_eps_max,
                "eta_scale_floor": hadamard_config.eta_scale_floor,
                "denominator_floor": hadamard_config.denominator_floor,
            },
            "score_definition": (
                "mean over fixed probes and evenly sampled truth-path states of "
                "sqrt(Hadamard residual H1 energy / forcing H1 energy)"
            ),
            "teacher_q_h1_definition": (
                "mean over the same sampled truth states of the trainer's relative "
                "H1 Fourier error ||sqrt(1+k^2)(q_theta-q_6)|| / "
                "||sqrt(1+k^2)q_6||"
            ),
            "teacher_q_l2_definitions": {
                "full": (
                    "mean over the same 17 sampled truth states of "
                    "||q_theta-q_6||_2 / ||q_6||_2"
                ),
                "p8": (
                    "the same relative L2 error after projection onto nonzero "
                    "Fourier modes |k|<=8"
                ),
            },
            "translation_definition": "sqrt(max(raw^2 - aligned^2, 0))",
        },
        "population": {
            "n_cases": len(cases),
            "tail_threshold_raw": args.tail_threshold,
            "n_raw_tails": int(np.count_nonzero(tail)),
            "probe_split_rank_stability": split_half_stability(defect_cube),
            "correlations": correlation_table(scores, targets),
            "family_correlations": family_correlations,
            "path_mean_defect_correlation_with_negative_depth": spearman_summary(
                scores["path_mean_defect"], -depths
            ),
            "partial_rank_correlations": partial_correlations,
            "path_mean_defect_tail_test": primary_tail_test,
            "terminal_mean_defect_tail_test": terminal_tail_test,
            "path_mean_teacher_q_h1_tail_test": teacher_h1_tail_test,
            "path_mean_teacher_q_l2_tail_test": teacher_l2_tail_test,
            "path_mean_teacher_q_p8_l2_tail_test": teacher_p8_tail_test,
            "family_path_mean_defect_tail_tests": family_tail_tests,
            "score_summaries": {
                name: {
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "maximum": float(np.max(values)),
                }
                for name, values in scores.items()
            },
        },
        "cases": [item.summary for item in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    series_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(series_output, **series_payload(cases, results, terminal))
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["population"], indent=2, allow_nan=False))
    print(f"summary -> {output}")
    print(f"series  -> {series_output}")


if __name__ == "__main__":
    main()
