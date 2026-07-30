"""Calibrate ordinary and dispersion-weighted mode losses on fixed batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
for directory in (
    REPO_ROOT,
    REPO_ROOT / "models" / "fno-jax",
    REPO_ROOT / "models" / "dno-net",
    REPO_ROOT / "train-jax-10m",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from losses import build_loss  # noqa: E402
from mode_balanced_regularizer import (  # noqa: E402
    ModeBalancedConfig,
    compute_mode_balanced_loss,
)
from solver.evals.model_rollout import LoadedRun, load_run  # noqa: E402
from solver.solvers.dno_series_jax import build_grid  # noqa: E402
from util import (  # noqa: E402
    NormStats,
    build_dataset_split_indices,
    load_dataset_arrays,
    make_normalizers,
    require_jax_devices,
)


ArrayTree = Any
GradientStep = Callable[
    [ArrayTree, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, ArrayTree, jax.Array],
]


@dataclass(frozen=True)
class BatchSpec:
    """One reproducible global calibration batch."""

    label: str
    indices: np.ndarray
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/c16_fixed_replay_ep16_to32_20260711_014635"),
    )
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--dataset", default="combined_dataset_v9.npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--natural-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stress-seed", type=int, default=20260717)
    parser.add_argument("--mode-weight", type=float, default=6.0)
    parser.add_argument("--target-gradient-fraction", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def directory_sha256(path: Path) -> str:
    """Hash file names and payloads in a checkpoint directory."""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def balanced_source_indices(
    train_indices: np.ndarray,
    source: np.ndarray,
    source_ids: tuple[int, ...],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a fixed, as-even-as-possible source-balanced batch."""
    base, remainder = divmod(batch_size, len(source_ids))
    counts = [base + int(index < remainder) for index in range(len(source_ids))]
    selected = [
        rng.choice(
            train_indices[source[train_indices] == source_id],
            size=count,
            replace=False,
        )
        for source_id, count in zip(source_ids, counts)
    ]
    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def build_batch_specs(
    train_indices: np.ndarray,
    source: np.ndarray,
    batch_size: int,
    natural_batches: int,
    seed: int,
    stress_seed: int,
) -> list[BatchSpec]:
    """Reproduce epoch-one batches and append fixed source stress batches."""
    epoch_seed = np.random.SeedSequence(seed).spawn(1)[0]
    ordered_indices = np.array(train_indices, copy=True)
    np.random.default_rng(epoch_seed).shuffle(ordered_indices)
    natural = [
        BatchSpec(
            label=f"natural_{index:02d}",
            indices=ordered_indices[index * batch_size : (index + 1) * batch_size],
            kind="natural",
        )
        for index in range(natural_batches)
    ]
    rng = np.random.default_rng(stress_seed)
    stress_groups = (
        ("tanaka", (5, 6, 14)),
        ("benjamin_feir", (7, 8, 9)),
        ("background", (0, 1, 2, 3, 4, 11, 12, 13)),
    )
    stress = [
        BatchSpec(
            label=f"stress_{label}",
            indices=balanced_source_indices(
                train_indices, source, source_ids, batch_size, rng
            ),
            kind="stress",
        )
        for label, source_ids in stress_groups
    ]
    return natural + stress


def tree_inner_product(left: ArrayTree, right: ArrayTree) -> float:
    """Return a float64 Euclidean inner product over parameter pytrees."""
    terms = [
        jnp.vdot(
            left_leaf.astype(jnp.float64), right_leaf.astype(jnp.float64)
        ).real
        for left_leaf, right_leaf in zip(
            jax.tree.leaves(left), jax.tree.leaves(right)
        )
    ]
    return float(jax.device_get(sum(terms, start=jnp.asarray(0.0, jnp.float64))))


def tree_norm(tree: ArrayTree) -> float:
    """Return the float64 Euclidean norm of a parameter pytree."""
    return float(np.sqrt(max(tree_inner_product(tree, tree), 0.0)))


def tree_cosine(left: ArrayTree, right: ArrayTree) -> float:
    """Return the cosine between two parameter-gradient pytrees."""
    denominator = tree_norm(left) * tree_norm(right)
    return tree_inner_product(left, right) / max(denominator, 1e-300)


def make_gradient_steps(
    loaded: LoadedRun,
    n_devices: int,
) -> tuple[GradientStep, GradientStep, GradientStep]:
    """Build replicated gradient steps for H1, mode, and dynamic mode losses."""
    stats = NormStats.from_dict(loaded.stats, mode=loaded.norm_mode)
    norm_inputs, norm_targets, denorm_targets = make_normalizers(stats)
    loss_h1 = build_loss(sobolev_k=int(loaded.config.get("sobolev_k", 1)))
    domain_length = float(loaded.config.get("domain_length", 2.0 * np.pi))
    nx = int(loaded.stats.get("nx", 1024))
    _, k = build_grid(nx, domain_length)
    k_rfft = jnp.asarray(np.abs(k[: nx // 2 + 1]), dtype=jnp.float32)
    gravity = float(loaded.config.get("pushforward_gravity", 1.0))
    depth_clip = float(loaded.config.get("pushforward_h_clip_max", 5.0))
    base_config = ModeBalancedConfig(
        k_max=128.0,
        gravity=gravity,
        active_scale_relative=1e-4,
        denominator_floor_relative=1e-6,
    )

    def predictions_and_targets(
        params: ArrayTree,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        log_depth: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        inputs = norm_inputs(eta, xi)
        targets = norm_targets(gxi)
        predictions = loaded.model.apply({"params": params}, inputs, log_depth)
        return predictions, targets, jnp.exp(
            jnp.minimum(log_depth[:, 0], jnp.log(depth_clip))
        )

    def h1_objective(
        params: ArrayTree,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        log_depth: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        predictions, targets, _ = predictions_and_targets(
            params, eta, xi, gxi, log_depth
        )
        return loss_h1(predictions, targets), jnp.zeros((4,), predictions.dtype)

    def mode_objective(
        params: ArrayTree,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        log_depth: jax.Array,
        *,
        dispersion_weighting: bool,
    ) -> tuple[jax.Array, jax.Array]:
        predictions, targets, depth = predictions_and_targets(
            params, eta, xi, gxi, log_depth
        )
        loss, diagnostics = compute_mode_balanced_loss(
            eta=eta,
            gxi_prediction=denorm_targets(predictions)[..., 0],
            gxi_target=denorm_targets(targets)[..., 0],
            depth=depth,
            k_rfft=k_rfft,
            config=ModeBalancedConfig(
                **{
                    **base_config.__dict__,
                    "dispersion_weighting": dispersion_weighting,
                }
            ),
        )
        auxiliary = jnp.stack(
            (
                diagnostics["unweighted_loss"],
                diagnostics["effective_frequency_squared_mean"],
                diagnostics["clipped_mode_fraction"],
                diagnostics["active_modes"],
            )
        )
        return loss, auxiliary

    def make_step(
        objective: Callable[..., tuple[jax.Array, jax.Array]],
    ) -> GradientStep:
        def local_step(
            params: ArrayTree,
            eta: jax.Array,
            xi: jax.Array,
            gxi: jax.Array,
            log_depth: jax.Array,
        ) -> tuple[jax.Array, ArrayTree, jax.Array]:
            (loss, auxiliary), gradient = jax.value_and_grad(
                objective, has_aux=True
            )(params, eta, xi, gxi, log_depth)
            return (
                jax.lax.pmean(loss, "devices"),
                jax.lax.pmean(gradient, "devices"),
                jax.lax.pmean(auxiliary, "devices"),
            )

        return jax.pmap(
            local_step,
            axis_name="devices",
            in_axes=(None, 0, 0, 0, 0),
            devices=jax.local_devices()[:n_devices],
        )

    return (
        make_step(h1_objective),
        make_step(
            lambda *values: mode_objective(
                *values, dispersion_weighting=False
            )
        ),
        make_step(
            lambda *values: mode_objective(
                *values, dispersion_weighting=True
            )
        ),
    )


def first_replica(tree: ArrayTree) -> ArrayTree:
    """Select one copy from a pmean-replicated pytree."""
    return jax.tree.map(lambda value: value[0], tree)


def evaluate_batch(
    spec: BatchSpec,
    dataset: dict[str, np.ndarray],
    loaded: LoadedRun,
    steps: tuple[GradientStep, GradientStep, GradientStep],
    n_devices: int,
    microbatch_size: int,
) -> dict[str, object]:
    """Compute loss and parameter-gradient diagnostics for one global batch."""
    gradient_parts: list[list[ArrayTree]] = [[] for _ in steps]
    loss_parts: list[list[float]] = [[] for _ in steps]
    diagnostic_parts: list[list[np.ndarray]] = [[] for _ in steps]
    part_weights: list[float] = []
    for start in range(0, spec.indices.size, microbatch_size):
        part_indices = spec.indices[start : start + microbatch_size]
        local_batch = part_indices.size // n_devices

        def shard(values: np.ndarray) -> jax.Array:
            selected = np.asarray(values[part_indices])
            return jnp.asarray(
                selected.reshape((n_devices, local_batch, *selected.shape[1:]))
            )

        eta = shard(dataset["eta"])
        xi = shard(dataset["xi"])
        gxi = shard(dataset["gxi"])
        depth = np.asarray(dataset["depth"][part_indices], dtype=np.float32)
        log_depth = jnp.asarray(np.log(np.clip(depth, 1e-12, None))).reshape(
            (n_devices, local_batch, 1)
        )
        outputs = [
            step(loaded.params, eta, xi, gxi, log_depth) for step in steps
        ]
        for index, output in enumerate(outputs):
            loss_parts[index].append(float(jax.device_get(output[0][0])))
            gradient_parts[index].append(first_replica(output[1]))
            diagnostic_parts[index].append(
                np.asarray(jax.device_get(output[2][0]))
            )
        part_weights.append(part_indices.size / spec.indices.size)

    losses = [
        float(np.sum(np.asarray(values) * np.asarray(part_weights)))
        for values in loss_parts
    ]
    gradients = [
        jax.tree.map(
            lambda *leaves: sum(
                (
                    weight * leaf
                    for weight, leaf in zip(part_weights, leaves)
                ),
                start=jnp.zeros_like(leaves[0]),
            ),
            *parts,
        )
        for parts in gradient_parts
    ]
    diagnostics = [
        np.sum(
            np.stack(values) * np.asarray(part_weights)[:, None], axis=0
        )
        for values in diagnostic_parts
    ]
    norms = [tree_norm(gradient) for gradient in gradients]
    if not all(np.isfinite((*losses, *norms))):
        raise FloatingPointError(f"nonfinite calibration value in {spec.label}")
    source_ids, source_counts = np.unique(
        dataset["source"][spec.indices], return_counts=True
    )
    return {
        "label": spec.label,
        "kind": spec.kind,
        "indices": spec.indices.tolist(),
        "source_counts": {
            str(int(source_id)): int(count)
            for source_id, count in zip(source_ids, source_counts)
        },
        "h1_loss": losses[0],
        "mode_loss": losses[1],
        "dynamic_mode_loss": losses[2],
        "h1_gradient_norm": norms[0],
        "mode_gradient_norm": norms[1],
        "dynamic_mode_gradient_norm": norms[2],
        "mode_to_h1_gradient_ratio": norms[1] / max(norms[0], 1e-300),
        "dynamic_to_h1_gradient_ratio": norms[2] / max(norms[0], 1e-300),
        "h1_mode_gradient_cosine": tree_cosine(gradients[0], gradients[1]),
        "h1_dynamic_gradient_cosine": tree_cosine(gradients[0], gradients[2]),
        "mode_dynamic_gradient_cosine": tree_cosine(gradients[1], gradients[2]),
        "mode_unweighted_loss": float(diagnostics[1][0]),
        "dynamic_unweighted_loss": float(diagnostics[2][0]),
        "effective_frequency_squared_mean": float(diagnostics[2][1]),
        "mode_clipped_fraction": float(diagnostics[1][2]),
        "dynamic_clipped_fraction": float(diagnostics[2][2]),
        "active_modes": float(diagnostics[1][3]),
    }


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.microbatch_size < 1
        or args.natural_batches < 1
    ):
        raise ValueError(
            "batch-size, microbatch-size, and natural-batches must be positive"
        )
    _, devices = require_jax_devices(min_device_count=1)
    n_devices = len(devices)
    if args.batch_size % args.microbatch_size:
        raise ValueError("batch-size must be divisible by microbatch-size")
    if args.microbatch_size % n_devices:
        raise ValueError("microbatch-size must be divisible by the device count")
    run_dir = args.run_dir.resolve()
    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    dataset_path = (REPO_ROOT / "data" / args.dataset).resolve()
    dataset = load_dataset_arrays(dataset_path)
    train_indices, _, _ = build_dataset_split_indices(dataset, args.seed)
    specs = build_batch_specs(
        train_indices=train_indices,
        source=np.asarray(dataset["source"]),
        batch_size=args.batch_size,
        natural_batches=args.natural_batches,
        seed=args.seed,
        stress_seed=args.stress_seed,
    )
    steps = make_gradient_steps(loaded, n_devices)
    records = []
    for spec in specs:
        print(f"calibrating {spec.label}", flush=True)
        records.append(
            evaluate_batch(
                spec,
                dataset,
                loaded,
                steps,
                n_devices,
                args.microbatch_size,
            )
        )

    natural = [record for record in records if record["kind"] == "natural"]
    mode_ratios = np.asarray(
        [record["mode_to_h1_gradient_ratio"] for record in natural]
    )
    dynamic_ratios = np.asarray(
        [record["dynamic_to_h1_gradient_ratio"] for record in natural]
    )
    historical_budget = float(args.mode_weight * np.median(mode_ratios))
    if 0.02 <= historical_budget <= 0.15:
        strategy = "historical_mode_weight"
        mode_weight = float(args.mode_weight)
        target_budget = historical_budget
    else:
        strategy = "fixed_gradient_fraction_fallback"
        target_budget = float(args.target_gradient_fraction)
        mode_weight = target_budget / float(np.median(mode_ratios))
    dynamic_weight = target_budget / float(np.median(dynamic_ratios))

    for record in records:
        record["weighted_mode_to_h1_gradient_ratio"] = float(
            mode_weight * record["mode_to_h1_gradient_ratio"]
        )
        record["weighted_dynamic_to_h1_gradient_ratio"] = float(
            dynamic_weight * record["dynamic_to_h1_gradient_ratio"]
        )

    natural_mode_weighted = np.asarray(
        [record["weighted_mode_to_h1_gradient_ratio"] for record in natural]
    )
    natural_dynamic_weighted = np.asarray(
        [record["weighted_dynamic_to_h1_gradient_ratio"] for record in natural]
    )
    stress = [record for record in records if record["kind"] == "stress"]
    maximum_stress_ratio = max(
        max(
            record["weighted_mode_to_h1_gradient_ratio"],
            record["weighted_dynamic_to_h1_gradient_ratio"],
        )
        for record in stress
    )
    maximum_clipped_fraction = max(
        max(record["mode_clipped_fraction"], record["dynamic_clipped_fraction"])
        for record in records
    )
    median_agreement = abs(
        np.median(natural_mode_weighted)
        - np.median(natural_dynamic_weighted)
    ) / max(target_budget, 1e-300)
    gates = {
        "historical_budget_or_fallback_valid": bool(
            strategy == "historical_mode_weight"
            or np.isclose(target_budget, args.target_gradient_fraction)
        ),
        "weighted_median_budget_agreement_within_10pct": bool(
            median_agreement <= 0.1
        ),
        "natural_q90_below_0p15": bool(
            max(
                np.quantile(natural_mode_weighted, 0.9),
                np.quantile(natural_dynamic_weighted, 0.9),
            )
            < 0.15
        ),
        "stress_max_below_0p25": bool(maximum_stress_ratio < 0.25),
        "mode_clipping_below_1e5": bool(maximum_clipped_fraction < 1e-5),
    }
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": loaded.epoch,
        "checkpoint_sha256": directory_sha256(
            run_dir
            / ("best_val_ckpt" if args.checkpoint == "best" else "final_ckpt")
        ),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "stress_seed": args.stress_seed,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "natural_batches": args.natural_batches,
        "device_count": n_devices,
        "mode_weight": mode_weight,
        "dynamic_mode_weight": dynamic_weight,
        "target_gradient_fraction": target_budget,
        "calibration_strategy": strategy,
        "historical_mode_gradient_fraction": historical_budget,
        "natural_weighted_mode_gradient_ratio_median": float(
            np.median(natural_mode_weighted)
        ),
        "natural_weighted_dynamic_gradient_ratio_median": float(
            np.median(natural_dynamic_weighted)
        ),
        "natural_weighted_mode_gradient_ratio_q90": float(
            np.quantile(natural_mode_weighted, 0.9)
        ),
        "natural_weighted_dynamic_gradient_ratio_q90": float(
            np.quantile(natural_dynamic_weighted, 0.9)
        ),
        "maximum_stress_gradient_ratio": float(maximum_stress_ratio),
        "maximum_clipped_mode_fraction": float(maximum_clipped_fraction),
        "gates": gates,
        "accepted": bool(all(gates.values())),
        "batches": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in payload if key != "batches"}, indent=2))
    if not payload["accepted"]:
        raise SystemExit("gradient calibration failed a preregistered gate")


if __name__ == "__main__":
    main()
