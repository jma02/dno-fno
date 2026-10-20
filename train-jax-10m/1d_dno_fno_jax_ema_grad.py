from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from util import (  # noqa: E402
    FlatParams,
    assert_pytree_replicated,
    build_dataset_split_indices,
    device_prefetch,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    make_normalizers,
    replicate_pytree_from_host,
)
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from solver.gen_data.benjamin_feir_sampling import BENJAMIN_FEIR_PARAMETER_GROUPS  # noqa: E402
from solver.evals.eval_suite import FAMILY_CONFIGS, _write_json, compute_macro_summary, run_family  # noqa: E402
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched, rollout_surrogate  # noqa: E402
from hadamard_shape_regularizer import compute_hadamard_loss, compute_local_response_loss  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_coherent_rollout_loss,
    compute_modal_phase_loss,
    compute_phase_mechanism_loss,
    compute_rhs_relative_error,
    compute_translation_tangent_loss,
)
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402

from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, CraigSulemTermsResidual, DepthAwareMultiplier  # noqa: E402
from fno1d import FNO1d  # noqa: E402
from checkpoint_util import (  # noqa: E402
    CheckpointMetadata,
    WeightEMATrainState,
    save_checkpoint,
    training_counter_values,
)


def subsample_family_indices(
    indices: np.ndarray, family_ids: np.ndarray, fraction: float, seed: int,
) -> np.ndarray:
    if fraction == 1.0:
        return indices
    rng = np.random.default_rng(seed)
    split_families = family_ids[indices]
    groups = (indices[split_families == family] for family in PhysicalFamilyId)
    return np.sort(np.concatenate([
        rng.choice(group, size=int(fraction * group.size), replace=False) for group in groups
    ]))


def main() -> None:
    # Model, dataset, and optimizer options.
    parser = argparse.ArgumentParser(description="Train a 1D JAX neural DNO surrogate with gradient EMA.")
    parser.add_argument("--model", choices=("fno", "cs_dno"), default="fno")
    parser.add_argument("--cs_terms_residual", action="store_true",
                        help="Decode CS fields through baseline order 1 or 3 with the surface encoder, at true depth.")
    parser.add_argument("--cs_terms_cpu_prefetch", action="store_true",
                        help="Prepare the next ordinary batch's CS fields on CPU while the GPU trains; use JAX_PLATFORMS=cuda,cpu.")
    parser.add_argument(
        "--cs_n_polys",
        type=int,
        default=3,
        help="Highest power of η used in the CS-DNO spatial features. "
        "n_polys=3 includes eta, eta^2, eta^3.",
    )
    for feature in ("first_deriv", "second_deriv", "half_deriv", "hilbert"):
        parser.add_argument(
            f"--cs_use_{feature}", action=argparse.BooleanOptionalAction, default=True
        )
    parser.add_argument(
        "--cs_use_relative_elevation", action="store_true",
        help="Add physical eta/h to the surface features.",
    )
    parser.add_argument(
        "--cs_use_learned_eta_filters", action="store_true",
        help="Learn depth-conditioned modulations of the second- and half-derivative surface filters.",
    )
    parser.add_argument("--cs_normalize_surface_features", action="store_true",
                        help="Fit fixed per-feature RMS scales on up to 65536 training rows.")
    parser.add_argument("--cs_surface_amplitude_normalization", action="store_true",
                        help="Normalize ordinary surface features per sample, condition on amplitude, and restore amplitude squared.")
    parser.add_argument("--cs_curvature_cutoff", type=float, default=0.0,
                        help="Physical-wavenumber cutoff for softened curvature; 0 keeps the second derivative.")
    parser.add_argument("--cs_whiten_surface_features", action="store_true",
                        help="Decorrelate RMS-scaled features with a fixed, uncentered training-only transform.")
    parser.add_argument("--cs_surface_context_width", type=int, default=0,
                        help="Channels in the depth-conditioned Fourier surface encoder; 0 disables it.")
    parser.add_argument("--cs_surface_context_gate", action="store_true",
                        help="Also modulate the quadratic surface lift by bounded Fourier context.")
    parser.add_argument("--cs_surface_depth_gate", action="store_true",
                        help="Modulate lifted surface features by a bounded, depth-only learned gain.")
    parser.add_argument("--cs_surface_polynomial_skip", action="store_true",
                        help="Append normalized eta squared/cubed directly to the final surface features.")
    parser.add_argument("--cs_surface_parity_products", action="store_true",
                        help="Use reflection-even derivative products and append them after the quadratic lift.")
    parser.add_argument("--cs_surface_parity_streams", action="store_true",
                        help="Retain odd derivative channels in a separate reflection-equivariant surface stream.")
    parser.add_argument("--cs_surface_cubic_channels", type=int, default=0,
                        help="Give this many final surface channels cubic rather than quadratic onset.")
    parser.add_argument("--cs_multiplier_skip", action="store_true",
                        help="Add a learned linear skip from normalized 1, k*tanh(kh), k squared to each multiplier.")
    parser.add_argument("--cs_log_multiplier_inputs", action="store_true",
                        help="Use log1p(k), log(h), tanh(kh), and log1p(k*tanh(kh)) as multiplier inputs.")
    parser.add_argument("--cs_nested_blocks", type=int, default=0,
                        help="Replace the final N branch groups with learned nested surface/filter compositions.")
    parser.add_argument("--cs_pair_coupling", action="store_true",
                        help="Add symmetric learned coupling within adjacent ordinary branch pairs.")
    parser.add_argument("--cs_nested_zero_inner", action="store_true",
                        help="Initialize nested gates at one and inner output weights at zero.")
    parser.add_argument("--cs_nested_surface_modulation", action="store_true",
                        help="Apply a shared bounded eta/h modulation to both nested surface factors.")
    parser.add_argument("--cs_nested_surface_features", action="store_true",
                        help="Learn per-branch mixtures of eta and its even linear features for nested factors.")
    parser.add_argument("--cs_phase_rank_one", action="store_true",
                        help="Add a learned signed rank-one correction along the half derivative of eta_x.")
    parser.add_argument("--cs_phase_wavenumber_hidden", type=int, default=0,
                        help="Hidden width of a kh-conditioned symmetric phase correction; 0 disables it.")
    parser.add_argument(
        "--cs_baseline_order", type=int, choices=(1, 2, 3), default=1,
        help="Highest analytic CS term; order 3 is available only for the CS-field decoder.",
    )
    parser.add_argument("--cs_baseline_true_depth", action="store_true",
                        help="Use the actual depth for the analytic baseline, without clipping at h=5.")
    parser.add_argument(
        "--cs_mult_hidden",
        type=int,
        default=32,
        help="Hidden size of the depth-aware Fourier-multiplier MLP.",
    )
    parser.add_argument("--norm", choices=("minmax", "scale"), default="minmax")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory containing the saved .npy arrays.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fixed seeded fraction of training rows retained per family.")
    parser.add_argument("--validation_fraction", type=float, default=1.0,
                        help="Fixed seeded fraction of validation rows retained per family; test rollouts are unchanged.")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--datum_shift_rms", type=float, default=0.0,
                        help="Training-only vertical-datum shifts on half the rows, relative to eta RMS; "
                        "shift eta and depth together, capped at one quarter of depth, keeping targets unchanged.")
    parser.add_argument("--reflection_augmentation", action="store_true",
                        help="Reflect eta, xi, and target together for half of training rows.")
    parser.add_argument("--input_noise_rms", type=float, default=0.0,
                        help="Training-only eta/xi noise relative to each centered field's RMS; "
                        "smooth spectrum 1/(1+k^2), 0<|k|<=128, unchanged DNO targets.")
    parser.add_argument(
        "--dirichlet_probe_mode", choices=("off", "control", "scrambled"), default="off",
        help="Replace the batch tail with paired copies of prefix surfaces; scrambled randomizes their potential phases.",
    )
    parser.add_argument(
        "--dirichlet_probe_fraction", type=float, default=0.25,
        help="Fraction of each batch reserved for paired DNO-data-only rows, at most one half.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Final epoch number; also the default schedule horizon. Completed epochs are not repeated.",
    )
    parser.add_argument("--schedule_epochs", type=int, default=0,
                        help="Initial LR schedule and weight-EMA start horizon, independent of final epoch; 0 uses --epochs.")
    parser.add_argument(
        "--stop_after_epoch", type=int, default=1,
        help="Stop after this epoch without shortening the learning-rate schedule; 0 runs all epochs.",
    )
    parser.add_argument("--rollout_epochs", type=int, nargs="+", default=[],
                        help="Evaluate weight-EMA parameters on 32 stratified test trajectories per family at these epochs.")
    parser.add_argument("--rollout_truth_run",
                        help="Run directory containing the two cached-M6 reference evaluation groups.")
    parser.add_argument("--rollout_training_weight", type=float, default=0.0,
                        help="Weight on the selected loss at detached M6-labelled rollout states; applied only on scheduled updates.")
    parser.add_argument("--rollout_training_loss", choices=("data", "full"), default="full",
                        help="Generated targets use data loss alone or data plus mode-balancing and tangent losses; original-data losses are unchanged.")
    parser.add_argument("--rollout_training_full_spectrum", action="store_true",
                        help="Use full M6 targets for valid generated rows, including their auxiliary losses; stored and fallback rows remain projected.")
    parser.add_argument("--rollout_training_rhs", action="store_true",
                        help="Replace generated-row DNO data fit by relative retained two-equation RHS error.")
    parser.add_argument("--rollout_coherent_weight", type=float, default=0.0,
                        help="Additional accumulated full-RHS error at detached GL2 stages, normalized by initial wave energy; uses the rollout schedule and ramp.")
    parser.add_argument("--rollout_training_start_epoch", type=int, default=60)
    parser.add_argument("--rollout_training_ramp_epochs", type=float, default=5.0,
                        help="Linear weight ramp duration in epochs; fractional values are allowed. 0 enables full weight immediately. Start 60 and ramp 5 reach full weight at the start of epoch 65.")
    parser.add_argument("--rollout_training_interval", type=int, default=16)
    parser.add_argument("--rollout_training_microbatch", type=int, default=8)
    parser.add_argument("--rollout_training_steps", type=int, default=10,
                        help="Number of GL2 steps at dt=0.01 used to generate each training endpoint.")
    parser.add_argument("--amplitude_probe_scale", type=float, default=0.0,
                        help="Scale original-source eta for extra M6 data supervision on rollout-teacher updates; 0 disables, otherwise must be in (0, 1).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_continuation", type=float, default=0.0,
                        help="LR after the initial schedule: ramp from zero over --lr_warmup_steps, then hold constant.")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Linear learning-rate warmup from zero before cosine decay. "
        "Useful when a zero-initialized residual sits on top of a strong "
        "analytic baseline. 0 preserves the original schedule.",
    )
    parser.add_argument("--lr_hold_fraction", type=float, default=0.0,
                        help="Fraction of planned updates before cosine decay starts; hold peak LR after warmup until this point. 0 starts decay immediately after warmup.")
    parser.add_argument("--surface_head_lr_multiplier", type=float, default=1.0,
                        help="Scale post-AdamW updates only for the spatial branch projection kernels.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--gradient_ema_decay", type=float, default=0.9,
        help="Bias-corrected gradient EMA decay before AdamW; must be in [0, 1).",
    )
    parser.add_argument("--adam_b1", type=float, default=0.9,
                        help="AdamW first-moment decay after gradient EMA; must be in [0, 1).")
    parser.add_argument("--adam_b2", type=float, default=0.999,
                        help="AdamW second-moment decay after gradient EMA; must be in [0, 1).")
    parser.add_argument("--weight_ema_decay", type=float, default=0.0,
                        help="Inference-only EMA of post-update parameters; 0 disables it.")
    parser.add_argument("--weight_ema_secondary_decay", type=float, default=0.0,
                        help="Optional second inference-only EMA sharing the primary start; 0 disables it.")
    parser.add_argument("--weight_ema_start_fraction", type=float, default=0.8,
                        help="Fraction of the initial schedule completed before weight averaging starts.")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--init_checkpoint", default=None,
                        help="Initialize weights from this checkpoint, with fresh optimizer state.")
    parser.add_argument("--nonlinear_reference", type=float, default=0.002,
                        help="Relative correction beyond G0+G1 at which the nonlinear weight is 1.5.")
    parser.add_argument("--nonlinear_weight_mean", type=float, default=1.463313,
                        help="Fixed training mean of the selected row weight; default is for nonlinear weighting.")
    parser.add_argument("--subtract_before_fft", action="store_true",
                        help="Transform the physical prediction error directly in data and mode-balancing losses.")
    parser.add_argument("--fp64_residual", action="store_true",
                        help="Fit the direct FP32 correction against the physical target minus FP64 M1 in data and MB losses.")
    parser.add_argument("--subtract_before_centering", action="store_true",
                        help="Subtract prediction and target before centering in tangent and phase losses.")
    parser.add_argument("--data_normalization", choices=("target", "bounded", "nonlinear", "residual"), default="nonlinear",
                        help="Target-relative loss, bounded inverse-residual weighting (raw weights 1..2), "
                        "nonlinear weighting, or normalization by the model's baseline residual.")
    parser.add_argument("--data_k_max", type=float, default=None,
                        help="Optional physical-wavenumber cutoff for data loss and its normalizers; full L2 metrics stay unfiltered.")
    parser.add_argument("--data_xi_projection_weight", type=float, default=0.0,
                        help="Extra weight inside the data-error norm along xi on 0<k<=128; ordinary L2 metrics stay unchanged.")
    parser.add_argument("--data_phase_projection_weight", type=float, default=0.0,
                        help="Extra weight inside the data-error norm perpendicular to each active eta Fourier mode.")
    parser.add_argument("--residual_floor", type=float, default=1e-6,
                        help="Residual denominator floor as a fraction of the target norm.")
    parser.add_argument("--data_modal_residual_floor", type=float, default=0.0,
                        help="Positive fractional peak-residual floor enables within-mode data weighting, "
                        "preserving the analytic-baseline loss scale; zero disables it.")
    parser.add_argument("--metric_interval", type=int, default=16,
                        help="Copy metrics to the host every N training or validation batches.")
    # Additional losses; a zero weight disables each one.
    parser.add_argument("--phase_mechanism", choices=("none", "stokes_carrier", "tanaka_momentum", "all_momentum", "bf_sideband"),
                        default="none", help="Coupled phase or momentum-production target; all_momentum uses every family uniformly.")
    parser.add_argument("--phase_mechanism_weight", type=float, default=0.0)
    parser.add_argument("--phase_mechanism_ramp_epochs", type=float, default=0.0,
                        help="Ramp the phase/momentum coefficient from zero over this many epochs; 0 uses full weight immediately.")
    parser.add_argument("--jonswap_mode_balanced_scale", type=float, default=1.0,
                        help="Multiply only JONSWAP/TMA rows' mode-balancing penalty by this factor.")
    parser.add_argument(
        "--translation_tangent_weight",
        type=float,
        default=0.0,
        help="Weight of the localized translation-tangent error on selected rows. Projects "
        "the DNO error onto eta_x in periodic windows and normalizes "
        "by sqrt(g*h). 0 disables.",
    )
    parser.add_argument(
        "--translation_tangent_scope", choices=("tanaka", "all"), default="tanaka",
        help="Wave families contributing to the translation-tangent loss.",
    )
    parser.add_argument(
        "--translation_tangent_geometry", choices=("local", "coupled_modal"), default="local",
        help="Local translation projection, or directional Fourier phase-rate errors of the coupled eta/xi dynamics.",
    )
    parser.add_argument("--translation_tangent_k_max", type=int, default=128,
                        help="Highest Fourier mode in the coupled-modal phase loss; match the rollout cutoff.")
    parser.add_argument(
        "--stokes_relative_tangent", action="store_true",
        help="Replace Stokes local tangent loss with target-normalized global translation error.",
    )
    parser.add_argument(
        "--translation_tangent_smoothing_scale",
        type=float,
        default=1.0,
        help="Gaussian smoothing standard deviation = this value * water depth.",
    )
    parser.add_argument(
        "--translation_tangent_denominator_eps",
        type=float,
        default=1e-3,
        help="Add eps * max_x(smoothed eta_x^2) to the speed-error denominator "
        "for each sample, preventing large estimates in nearly flat regions.",
    )
    parser.add_argument(
        "--translation_tangent_loss_norm",
        choices=("squared", "charbonnier"),
        default="squared",
        help="Per-row tangent error norm: squared phase pressure, or Charbonnier "
        "sqrt(row_loss + eps^2) - eps for absolute global / RMS local phase error.",
    )
    parser.add_argument(
        "--translation_tangent_charbonnier_eps",
        type=float,
        default=1e-7,
        help="Charbonnier smoothing epsilon; must be finite and positive.",
    )
    parser.add_argument(
        "--mode_balanced_weight",
        type=float,
        default=0.0,
        help="Weight of the universal mode-balanced complex spectral loss. 0 disables.",
    )
    parser.add_argument(
        "--mode_balanced_warmup_steps",
        type=int,
        default=500,
        help="Linear ramp of mode_balanced_weight over this many optimizer steps.",
    )
    parser.add_argument(
        "--mode_balanced_k_max",
        type=float,
        default=128.0,
        help="Largest positive physical wavenumber included in the mode-balanced loss.",
    )
    parser.add_argument(
        "--mode_balanced_activity_threshold",
        type=float,
        default=1e-4,
        help="Downweight weak Fourier modes. At 1e-4, a mode with 0.01%% of the "
        "strongest mode's strength in the same sample gets about half weight.",
    )
    parser.add_argument(
        "--mode_balanced_denominator_eps",
        type=float,
        default=1e-6,
        help="Add eps times the strongest mode's strength in each sample to every "
        "mode's error denominator.",
    )
    parser.add_argument(
        "--hadamard_weight",
        type=float,
        default=0.0,
        help="Weight of the randomized finite-secant DNO Hadamard shape-identity "
        "loss. 0 disables the regularizer.",
    )
    parser.add_argument(
        "--hadamard_mode", choices=("self", "secant", "teacher", "targeted"), default="self",
        help="Local regularizer: learned identity, M6 joint secant, M6 Hadamard derivative, "
        "or M6 secant with phase/flow-defect directions.",
    )
    parser.add_argument(
        "--hadamard_flow_probes", action="store_true",
        help="Mix stored-target surface-flow directions with random self-Hadamard "
        "training probes; validation keeps random probes.",
    )
    parser.add_argument(
        "--hadamard_interval",
        type=int,
        default=4,
        help="Evaluate the Hadamard regularizer every this-many optimizer steps.",
    )
    parser.add_argument(
        "--val_hadamard_interval", type=int, default=16,
        help="Evaluate the sampled validation Hadamard diagnostic every N batches; 1 evaluates all.",
    )
    parser.add_argument(
        "--hadamard_microbatch",
        type=int,
        default=8,
        help="Global Hadamard microbatch size, split evenly across devices.",
    )
    parser.add_argument(
        "--hadamard_warmup_steps",
        type=int,
        default=500,
        help="Linear ramp of hadamard_weight from zero to its configured value.",
    )
    parser.add_argument(
        "--hadamard_k_max",
        type=float,
        default=128.0,
        help="Maximum retained |k| in the Hadamard defect and probe.",
    )
    parser.add_argument(
        "--hadamard_sobolev_order",
        type=int,
        choices=(0, 1),
        default=1,
        help="Hadamard norm: 0 = L2, 1 = H1 (includes the first derivative).",
    )
    parser.add_argument(
        "--hadamard_fd_step_min",
        type=float,
        default=1e-3,
        help="Minimum relative finite-difference step (fraction of surface RMS).",
    )
    parser.add_argument(
        "--hadamard_fd_step_max",
        type=float,
        default=3e-3,
        help="Maximum relative finite-difference step (fraction of surface RMS).",
    )
    parser.add_argument(
        "--hadamard_min_surface_rms",
        type=float,
        default=1e-3,
        help="Minimum surface RMS used to set the perturbation's scale.",
    )
    parser.add_argument(
        "--hadamard_denominator_eps",
        type=float,
        default=1e-12,
        help="Epsilon added to the squared norm in the Hadamard loss denominator.",
    )
    args = parser.parse_args()
    if args.cs_phase_wavenumber_hidden < 0:
        parser.error("--cs_phase_wavenumber_hidden must be nonnegative")
    if args.cs_phase_wavenumber_hidden and (
        args.model != "cs_dno" or args.cs_terms_residual or args.cs_baseline_order != 1 or args.cs_phase_rank_one
    ):
        parser.error("--cs_phase_wavenumber_hidden requires tied-filter cs_dno with G0+G1 and no rank-one head")
    if args.cs_phase_rank_one and (
        args.model != "cs_dno" or args.cs_terms_residual or args.cs_baseline_order != 1
    ):
        parser.error("--cs_phase_rank_one requires the tied-filter cs_dno model with a G0+G1 baseline")
    if args.cs_terms_residual:
        if args.model != "cs_dno" or args.norm != "scale" or not args.cs_normalize_surface_features:
            parser.error("--cs_terms_residual requires cs_dno, scale normalization, and RMS-scaled surface features")
        if args.cs_baseline_order not in (1, 3):
            parser.error("--cs_terms_residual requires --cs_baseline_order 1 or 3")
        if (args.cs_nested_blocks or args.cs_nested_zero_inner or args.cs_nested_surface_modulation
                or args.cs_nested_surface_features or args.cs_surface_context_gate or args.cs_surface_depth_gate
                or args.cs_surface_polynomial_skip or args.cs_multiplier_skip or args.cs_log_multiplier_inputs
                or args.surface_head_lr_multiplier != 1.0):
            parser.error("--cs_terms_residual does not use nested branches, post-trunk gates/skips, or block multipliers")
        args.cs_baseline_true_depth = True
    elif args.cs_baseline_order == 3:
        parser.error("--cs_baseline_order 3 requires --cs_terms_residual")
    if args.cs_terms_cpu_prefetch and (
        not args.cs_terms_residual or args.datum_shift_rms > 0 or args.reflection_augmentation
        or args.input_noise_rms > 0 or args.dirichlet_probe_mode != "off"
    ):
        parser.error("--cs_terms_cpu_prefetch requires --cs_terms_residual and unmodified ordinary inputs")
    if args.schedule_epochs < 0 or args.epochs < 1:
        parser.error("--schedule_epochs must be nonnegative and --epochs must be positive")
    schedule_epochs = args.schedule_epochs or args.epochs
    if not np.isfinite(args.lr_continuation) or args.lr_continuation < 0:
        parser.error("--lr_continuation must be finite and nonnegative")
    if args.hadamard_flow_probes and args.hadamard_mode != "self":
        parser.error("--hadamard_flow_probes requires --hadamard_mode self")
    if not np.isfinite(args.phase_mechanism_weight) or args.phase_mechanism_weight < 0:
        parser.error("phase_mechanism_weight must be finite and nonnegative")
    if not np.isfinite(args.phase_mechanism_ramp_epochs) or args.phase_mechanism_ramp_epochs < 0:
        parser.error("phase_mechanism_ramp_epochs must be finite and nonnegative")
    if not np.isfinite(args.jonswap_mode_balanced_scale) or args.jonswap_mode_balanced_scale < 0:
        parser.error("jonswap_mode_balanced_scale must be finite and nonnegative")
    if (args.phase_mechanism == "none") != (args.phase_mechanism_weight == 0):
        parser.error("A phase mechanism requires a positive weight; none requires zero weight")
    if not all(0.0 < fraction <= 1.0 for fraction in (args.train_fraction, args.validation_fraction)):
        parser.error("--train_fraction and --validation_fraction must be in (0, 1]")
    if args.rollout_epochs and (args.rollout_truth_run is None or
                               any(epoch < 1 or epoch > args.epochs for epoch in args.rollout_epochs)):
        parser.error("--rollout_epochs requires --rollout_truth_run and epochs within the training budget")
    if not 0.0 <= args.gradient_ema_decay < 1.0:
        parser.error("--gradient_ema_decay must be in [0, 1)")
    if not 0.0 <= args.adam_b1 < 1.0:
        parser.error("--adam_b1 must be in [0, 1)")
    if not 0.0 <= args.adam_b2 < 1.0:
        parser.error("--adam_b2 must be in [0, 1)")
    if not np.isfinite(args.lr_hold_fraction) or not 0.0 <= args.lr_hold_fraction < 1.0:
        parser.error("--lr_hold_fraction must be finite and in [0, 1)")
    if not np.isfinite(args.surface_head_lr_multiplier) or args.surface_head_lr_multiplier <= 0:
        parser.error("--surface_head_lr_multiplier must be finite and positive")
    if not 0.0 <= args.weight_ema_decay < 1.0:
        parser.error("--weight_ema_decay must be in [0, 1)")
    if not 0.0 <= args.weight_ema_secondary_decay < 1.0:
        parser.error("--weight_ema_secondary_decay must be in [0, 1)")
    if args.weight_ema_secondary_decay > 0.0 and args.weight_ema_decay == 0.0:
        parser.error("--weight_ema_secondary_decay requires --weight_ema_decay > 0")
    if args.rollout_epochs and args.weight_ema_decay == 0.0:
        parser.error("--rollout_epochs requires --weight_ema_decay > 0")
    if not 0.0 <= args.weight_ema_start_fraction < 1.0:
        parser.error("--weight_ema_start_fraction must be in [0, 1)")
    if args.rollout_epochs and min(args.rollout_epochs) <= schedule_epochs * args.weight_ema_start_fraction:
        parser.error("--rollout_epochs must follow the weight-EMA start; choose later milestones or an earlier --weight_ema_start_fraction")
    if args.metric_interval < 1:
        parser.error("--metric_interval must be positive")
    if not np.isfinite(args.datum_shift_rms) or args.datum_shift_rms < 0:
        parser.error("--datum_shift_rms must be finite and nonnegative")
    if not np.isfinite(args.input_noise_rms) or args.input_noise_rms < 0:
        parser.error("--input_noise_rms must be finite and nonnegative")
    if not np.isfinite(args.rollout_training_weight) or args.rollout_training_weight < 0:
        parser.error("--rollout_training_weight must be finite and nonnegative")
    if args.rollout_training_full_spectrum and args.rollout_training_weight == 0:
        parser.error("--rollout_training_full_spectrum requires --rollout_training_weight > 0")
    if args.rollout_training_rhs and not args.rollout_training_full_spectrum:
        parser.error("--rollout_training_rhs requires --rollout_training_full_spectrum")
    if not np.isfinite(args.rollout_coherent_weight) or args.rollout_coherent_weight < 0:
        parser.error("--rollout_coherent_weight must be finite and nonnegative")
    if args.rollout_coherent_weight > 0 and args.rollout_training_weight == 0:
        parser.error("--rollout_coherent_weight requires --rollout_training_weight > 0")
    if not np.isfinite(args.rollout_training_ramp_epochs) or args.rollout_training_ramp_epochs < 0:
        parser.error("rollout-training ramp must be finite and nonnegative")
    if min(args.rollout_training_start_epoch, args.rollout_training_interval,
           args.rollout_training_microbatch, args.rollout_training_steps) < 1:
        parser.error("rollout-training epoch, interval, microbatch and steps must be positive")
    if not np.isfinite(args.amplitude_probe_scale) or not 0.0 <= args.amplitude_probe_scale < 1.0:
        parser.error("--amplitude_probe_scale must be 0 (disabled) or in (0, 1)")
    if args.amplitude_probe_scale > 0 and args.rollout_training_weight == 0:
        parser.error("--amplitude_probe_scale requires --rollout_training_weight > 0")
    if args.val_hadamard_interval < 1:
        parser.error("--val_hadamard_interval must be positive")
    if args.stokes_relative_tangent and args.translation_tangent_scope != "all":
        parser.error("--stokes_relative_tangent requires --translation_tangent_scope all")
    if args.stokes_relative_tangent and args.translation_tangent_geometry != "local":
        parser.error("--stokes_relative_tangent requires local tangent geometry")
    if args.translation_tangent_k_max < 1:
        parser.error("--translation_tangent_k_max must be positive")
    if (
        not np.isfinite(args.translation_tangent_charbonnier_eps)
        or args.translation_tangent_charbonnier_eps <= 0.0
    ):
        parser.error("--translation_tangent_charbonnier_eps must be finite and positive")
    if not np.isfinite(args.nonlinear_reference) or args.nonlinear_reference <= 0.0:
        parser.error("--nonlinear_reference must be finite and positive")
    if not 1.0 <= args.nonlinear_weight_mean <= 2.0:
        parser.error("--nonlinear_weight_mean must be in [1, 2]")
    if not np.isfinite(args.residual_floor) or args.residual_floor <= 0.0:
        parser.error("--residual_floor must be finite and positive")
    if not np.isfinite(args.data_modal_residual_floor) or args.data_modal_residual_floor < 0.0:
        parser.error("--data_modal_residual_floor must be finite and nonnegative")
    if args.data_modal_residual_floor and args.model != "cs_dno":
        parser.error("--data_modal_residual_floor requires --model cs_dno")
    if args.data_k_max is not None and (not np.isfinite(args.data_k_max) or args.data_k_max <= 0.0):
        parser.error("--data_k_max must be finite and positive")
    if not np.isfinite(args.data_xi_projection_weight) or args.data_xi_projection_weight < 0.0:
        parser.error("--data_xi_projection_weight must be finite and nonnegative")
    if not np.isfinite(args.data_phase_projection_weight) or args.data_phase_projection_weight < 0.0:
        parser.error("--data_phase_projection_weight must be finite and nonnegative")
    if args.data_normalization in ("bounded", "residual") and args.model != "cs_dno":
        parser.error("--data_normalization bounded/residual requires --model cs_dno")
    if not 0.0 < args.dirichlet_probe_fraction <= 0.5:
        parser.error("--dirichlet_probe_fraction must be in (0, 0.5]")
    if args.model == "cs_dno" and args.norm != "scale":
        parser.error("CS residual normalization requires --norm scale")
    if args.cs_surface_context_width < 0:
        parser.error("--cs_surface_context_width must be nonnegative")
    if not np.isfinite(args.cs_curvature_cutoff) or args.cs_curvature_cutoff < 0:
        parser.error("--cs_curvature_cutoff must be finite and nonnegative")
    if args.cs_curvature_cutoff > 0 and not args.cs_use_second_deriv:
        parser.error("--cs_curvature_cutoff requires --cs_use_second_deriv")
    if args.cs_surface_context_gate and args.cs_surface_context_width == 0:
        parser.error("--cs_surface_context_gate requires --cs_surface_context_width > 0")
    if args.cs_surface_amplitude_normalization and (
        args.model != "cs_dno" or not args.cs_normalize_surface_features or args.cs_baseline_order != 1
        or args.cs_terms_residual or args.cs_nested_surface_features or args.cs_nested_surface_modulation
    ):
        parser.error("--cs_surface_amplitude_normalization requires RMS-scaled tied-filter M1 features "
                     "without nested surface features/modulation")
    if args.cs_surface_polynomial_skip and (args.model != "cs_dno" or args.cs_n_polys < 3 or args.cs_baseline_order != 1):
        parser.error("Polynomial skip requires cs_dno, at least three surface powers, and the G0+G1 baseline")
    if not 0 <= args.cs_surface_cubic_channels <= args.width // 2:
        parser.error("--cs_surface_cubic_channels must lie between zero and half the width")
    if args.cs_surface_cubic_channels and (
        args.model != "cs_dno" or args.cs_baseline_order != 1 or args.cs_terms_residual
        or args.cs_surface_parity_streams or args.cs_surface_amplitude_normalization
    ):
        parser.error("Cubic surface channels require tied-filter M1 without parity streams or amplitude normalization")
    if args.cs_surface_parity_products and (
        args.model != "cs_dno" or args.cs_terms_residual or args.cs_baseline_order != 1
        or args.cs_n_polys != 3 or not args.cs_normalize_surface_features
        or not all((args.cs_use_first_deriv, args.cs_use_second_deriv, args.cs_use_half_deriv, args.cs_use_hilbert))
        or args.cs_use_relative_elevation or args.cs_use_learned_eta_filters
        or args.cs_surface_amplitude_normalization or args.cs_whiten_surface_features
    ):
        parser.error("Parity products require RMS-scaled tied-filter M1, three surface powers and all four "
                     "derivative features, without extra eta channels, amplitude normalization or whitening")
    if args.cs_surface_parity_streams and (
        not args.cs_surface_parity_products or args.width % 4
        or args.cs_surface_context_width <= 0 or args.cs_surface_context_width % 2
        or args.cs_surface_context_gate or args.cs_surface_polynomial_skip or args.cs_surface_depth_gate
        or args.cs_nested_surface_features or args.cs_nested_surface_modulation
        or args.cs_phase_rank_one or args.cs_phase_wavenumber_hidden
    ):
        parser.error("Parity streams require parity products, width divisible by four, positive even context "
                     "width, and no context/depth gates, polynomial skip, nested surface changes or phase heads")
    if not 0 <= args.cs_nested_blocks <= args.n_blocks:
        parser.error("--cs_nested_blocks must be between zero and --n_blocks")
    if args.cs_pair_coupling and (
        args.model != "cs_dno" or args.cs_terms_residual or args.latent < 2 or args.latent % 2
        or args.cs_nested_blocks >= args.n_blocks
    ):
        parser.error("--cs_pair_coupling requires tied-filter cs_dno, positive even latent, and an ordinary block")
    if args.cs_nested_blocks and (args.model != "cs_dno" or args.cs_baseline_order != 1):
        parser.error("Nested branches require --model cs_dno with the G0+G1 baseline")
    if args.fp64_residual and (
        args.model != "cs_dno" or args.norm != "scale" or args.cs_baseline_order != 1 or not args.cs_baseline_true_depth
    ):
        parser.error("--fp64_residual requires cs_dno, scale normalization, and true-depth G0+G1")
    if args.cs_nested_zero_inner and not args.cs_nested_blocks:
        parser.error("--cs_nested_zero_inner requires --cs_nested_blocks > 0")
    if args.cs_nested_surface_modulation and not args.cs_nested_blocks:
        parser.error("--cs_nested_surface_modulation requires --cs_nested_blocks > 0")
    if args.cs_nested_surface_features and (
        not args.cs_nested_blocks or not args.cs_normalize_surface_features
        or not args.cs_use_half_deriv or not args.cs_use_second_deriv
        or args.cs_use_learned_eta_filters or args.cs_nested_surface_modulation
    ):
        parser.error("--cs_nested_surface_features requires nested blocks, RMS-scaled half/second derivatives, "
                     "and no learned eta filters or nested surface modulation")
    if args.model != "cs_dno" and (
        args.cs_normalize_surface_features or args.cs_surface_context_width or args.cs_curvature_cutoff > 0
    ):
        parser.error("Surface-feature experiments require --model cs_dno")
    if args.cs_whiten_surface_features and not args.cs_normalize_surface_features:
        parser.error("--cs_whiten_surface_features requires --cs_normalize_surface_features")
    start_time = perf_counter()

    training_dtype = jnp.float32
    physical_dtype = jnp.float64 if args.cs_terms_residual else training_dtype

    # Split batches across GPUs, but keep a full model/optimizer copy on each GPU.
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"JAX GPU backend is required. Found {backend!r}.")
    cpu_sharding = SingleDeviceSharding(jax.devices("cpu")[0]) if args.cs_terms_cpu_prefetch else None
    devices = jax.local_devices()
    n_devices = len(devices)
    if args.rollout_training_weight > 0 and args.rollout_training_microbatch % n_devices:
        parser.error("--rollout_training_microbatch must be divisible by the device count")
    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by device count {n_devices}"
        )
    if args.hadamard_weight > 0.0 and args.hadamard_microbatch % n_devices != 0:
        raise ValueError(
            f"hadamard_microbatch {args.hadamard_microbatch} must be divisible by "
            f"device count {n_devices}"
        )
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())

    dataset_path = Path(args.dataset).expanduser().resolve()
    outputs_dir = REPO_ROOT / args.output_root
    run_name = args.run_name or datetime.now().strftime("fno_jax_10m_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Use the saved simulation splits and fit normalization on training samples only.
    dataset = load_dataset_arrays(dataset_path)
    family_ids = np.load(dataset_path / "family_id.npy", mmap_mode="r")
    nx = int(dataset["x"].shape[0])
    train_indices, val_indices, _ = build_dataset_split_indices(dataset)
    train_indices = subsample_family_indices(train_indices, family_ids, args.train_fraction, args.seed)
    val_indices = subsample_family_indices(val_indices, family_ids, args.validation_fraction, args.seed + 1)
    if not train_indices.size or not val_indices.size:
        parser.error("The selected fractions leave an empty training or validation split")
    stats = load_or_compute_stats(dataset_path, dataset, indices=train_indices)
    phase_modes = None
    if args.phase_mechanism == "bf_sideband":
        phase_modes = np.zeros((family_ids.size, 2), dtype=np.int16)
        selected_rows = np.concatenate((train_indices, val_indices))
        bf_rows = selected_rows[family_ids[selected_rows] == PhysicalFamilyId.BENJAMIN_FEIR]
        group_ids = np.load(dataset_path / "parameter_group_id.npy", mmap_mode="r")
        groups, inverse = np.unique(group_ids[bf_rows], return_inverse=True)
        phase_modes[bf_rows] = np.asarray([BENJAMIN_FEIR_PARAMETER_GROUPS[group] for group in groups])[inverse]
    full_batches, remainder = divmod(train_indices.size, args.batch_size)
    train_steps_per_epoch = (
        full_batches + bool(remainder // n_devices) + bool(remainder % n_devices)
    )

    domain_length = float(dataset["domain_length"])
    # Physical-unit scaling for both models' analytic DNO terms, matched to norm=scale.
    feature_scale = np.asarray(
        cast(list[float], stats["feature_absmax"]), dtype=physical_dtype
    )
    eta_scale, xi_scale = map(float, np.where(feature_scale > 0, feature_scale, 1.0))
    target_absmax = float(cast(float, stats["target_absmax"]))
    target_scale = target_absmax if target_absmax > 0 else 1.0
    args.cs_surface_feature_scales = None
    args.cs_surface_feature_transform = None
    args.cs_surface_amplitude_reference = 0.0
    if args.cs_normalize_surface_features:
        # Independent RNG: fitting training-only scales must not advance the batch shuffle.
        feature_rows = np.random.default_rng(0).choice(
            train_indices, size=min(65536, train_indices.size), replace=False,
        )
        if args.cs_surface_amplitude_normalization:
            surface_energy = 0.0
            for start in range(0, feature_rows.size, 256):
                eta = np.asarray(dataset["eta"][feature_rows[start:start + 256]], dtype=np.float32) / np.float32(eta_scale)
                surface_energy += float(np.sum(np.square(eta, dtype=np.float64)))
            args.cs_surface_amplitude_reference = float(np.sqrt(surface_energy / (feature_rows.size * nx)))
            if not np.isfinite(args.cs_surface_amplitude_reference) or args.cs_surface_amplitude_reference <= 0:
                raise FloatingPointError("Non-positive or non-finite training surface RMS")
            print(f"Surface-amplitude reference RMS: {args.cs_surface_amplitude_reference}")
        feature_k = (2.0 * np.pi / domain_length) * np.arange(nx // 2 + 1)
        feature_second_symbol = -feature_k**2
        if args.cs_curvature_cutoff > 0:
            feature_second_symbol = feature_second_symbol / np.sqrt(
                1.0 + (feature_k / args.cs_curvature_cutoff) ** 2
            )
        feature_symbols = [symbol for enabled, symbol in (
            (args.cs_use_first_deriv, 1j * feature_k),
            (args.cs_use_second_deriv, feature_second_symbol),
            (args.cs_use_half_deriv, np.sqrt(feature_k)),
            (args.cs_use_hilbert, -1j * np.sign(feature_k)),
        ) if enabled]
        feature_energy = np.zeros(args.cs_n_polys + len(feature_symbols)
                                  + args.cs_use_relative_elevation + args.cs_surface_parity_products
                                  + 2 * args.cs_surface_parity_streams)
        feature_cross = (np.zeros((feature_energy.size, feature_energy.size))
                         if args.cs_whiten_surface_features else None)
        for start in range(0, feature_rows.size, 256):
            rows = feature_rows[start:start + 256]
            eta_features = np.asarray(dataset["eta"][rows], dtype=np.float32) / np.float32(eta_scale)
            fields = [eta_features**p for p in range(1, args.cs_n_polys + 1)]
            eta_hat = np.fft.rfft(eta_features, axis=-1)
            fields.extend(np.fft.irfft(eta_hat * symbol, n=nx, axis=-1) for symbol in feature_symbols)
            if args.cs_surface_parity_products:
                dx, hilbert = fields[3], fields[6]
                fields[3], fields[6] = dx**2, hilbert**2
                fields.append(dx * hilbert)
                if args.cs_surface_parity_streams:
                    fields.extend((dx, hilbert))
            if args.cs_use_relative_elevation:
                fields.append(eta_features * eta_scale / dataset["depth"][rows, None])
            if args.cs_surface_amplitude_normalization:
                amplitude = np.sqrt(np.mean(
                    (eta_features / np.float32(args.cs_surface_amplitude_reference)) ** 2,
                    axis=-1, keepdims=True,
                ) + np.float32(1e-6))
                fields = [field / amplitude ** (index + 1 if index < args.cs_n_polys else 1)
                          for index, field in enumerate(fields)]
            feature_energy += np.array([
                np.sum(np.square(field, dtype=np.float64)) for field in fields
            ])
            if feature_cross is not None:
                field_matrix = np.stack(fields, axis=-1).reshape(-1, feature_energy.size).astype(np.float64)
                feature_cross += field_matrix.T @ field_matrix
        feature_rms = np.sqrt(feature_energy / (feature_rows.size * nx))
        if not np.all(np.isfinite(feature_rms)):
            raise FloatingPointError("Non-finite training surface-feature RMS")
        args.cs_surface_feature_scales = np.maximum(feature_rms, 1e-12).tolist()
        print(f"Surface-feature RMS ({feature_rows.size} training rows): {args.cs_surface_feature_scales}")
        if feature_cross is not None:
            scales = np.asarray(args.cs_surface_feature_scales)
            second_moment = feature_cross / (feature_rows.size * nx) / scales[:, None] / scales[None, :]
            eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
            eigen_floor = max(0.05 * float(np.trace(second_moment)) / feature_energy.size, 1e-12)
            transform = (eigenvectors / np.sqrt(np.maximum(eigenvalues, eigen_floor))) @ eigenvectors.T
            if not np.all(np.isfinite(transform)):
                raise FloatingPointError("Non-finite training surface-feature whitening transform")
            args.cs_surface_feature_transform = transform.tolist()
            print(f"Surface-feature second-moment eigenvalues: {eigenvalues}; whitening floor: {eigen_floor}")
    # Construct the selected model and initialize its parameters.
    if args.model == "cs_dno":
        model_class = CraigSulemTermsResidual if args.cs_terms_residual else CraigSulemDNO
        model = model_class(
            width=args.width,
            n_blocks=args.n_blocks,
            latent=args.latent,
            n_polys=args.cs_n_polys,
            use_first_deriv=args.cs_use_first_deriv,
            use_second_deriv=args.cs_use_second_deriv,
            curvature_cutoff=args.cs_curvature_cutoff,
            use_half_deriv=args.cs_use_half_deriv,
            use_hilbert=args.cs_use_hilbert,
            use_relative_elevation=args.cs_use_relative_elevation,
            use_learned_eta_filters=args.cs_use_learned_eta_filters,
            surface_feature_scales=(tuple(args.cs_surface_feature_scales)
                                    if args.cs_surface_feature_scales is not None else None),
            surface_amplitude_reference=args.cs_surface_amplitude_reference,
            surface_feature_transform=(tuple(map(tuple, args.cs_surface_feature_transform))
                                       if args.cs_surface_feature_transform is not None else None),
            surface_context_width=args.cs_surface_context_width,
            surface_context_gate=args.cs_surface_context_gate,
            surface_depth_gate=args.cs_surface_depth_gate,
            surface_polynomial_skip=args.cs_surface_polynomial_skip,
            surface_parity_products=args.cs_surface_parity_products,
            surface_parity_streams=args.cs_surface_parity_streams,
            surface_cubic_channels=args.cs_surface_cubic_channels,
            multiplier_skip=args.cs_multiplier_skip,
            log_multiplier_inputs=args.cs_log_multiplier_inputs,
            nested_blocks=args.cs_nested_blocks,
            pair_coupling=args.cs_pair_coupling,
            nested_surface_modulation=args.cs_nested_surface_modulation,
            nested_surface_features=args.cs_nested_surface_features,
            phase_rank_one=args.cs_phase_rank_one,
            phase_wavenumber_hidden=args.cs_phase_wavenumber_hidden,
            mult_hidden=args.cs_mult_hidden,
            baseline_order=args.cs_baseline_order,
            baseline_true_depth=args.cs_baseline_true_depth,
            domain_length=domain_length,
            xi_scale=xi_scale,
            eta_scale=eta_scale,
            target_scale=target_scale,
        )
    else:
        model = FNO1d(
            modes=args.modes,
            width=args.width,
            n_blocks=args.n_blocks,
            domain_length=domain_length,
            xi_scale=xi_scale,
            target_scale=target_scale,
        )
    with jax.default_device(devices[0]):
        params = model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros((1, nx, 2), dtype=training_dtype),
            jnp.zeros((1, 1), dtype=training_dtype),
        )["params"]
        if args.cs_nested_zero_inner:
            for block_idx in range(args.n_blocks - args.cs_nested_blocks, args.n_blocks):
                group = params[f"cs_block_{block_idx}"]
                group["gamma"] = jnp.ones_like(cast(jax.Array, group["gamma"]))
                group["m_inner"]["out"] = jax.tree.map(jnp.zeros_like, group["m_inner"]["out"])
        if args.init_checkpoint and not (run_dir / "latest_ckpt" / "metadata.json").exists():
            restored_weights = checkpoints.restore_checkpoint(
                ckpt_dir=Path(args.init_checkpoint).resolve(), target=None,
                prefix="ckpt_", orbax_checkpointer=ocp.PyTreeCheckpointer(),
            )
            params = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value, dtype=training_dtype), restored_weights["params"]
            )
            print(f"Initialized weights from {args.init_checkpoint}; optimizer starts fresh")

    def apply_model(variables: dict[str, Any], inputs: jax.Array, depth: jax.Array,
                    *, return_baseline: bool = False, terms_phys: jax.Array | None = None) -> Any:
        if args.model != "cs_dno":
            prediction = cast(jax.Array, model.apply(variables, inputs, depth))
            return (prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction) if return_baseline else prediction
        if args.cs_terms_residual:
            return model.apply(variables, inputs, depth, return_baseline=return_baseline, terms_phys=terms_phys)
        multiplier_model = DepthAwareMultiplier(
            args.latent, domain_length, 5.0, args.cs_mult_hidden,
            multiplier_skip=args.cs_multiplier_skip,
            log_multiplier_inputs=args.cs_log_multiplier_inputs,
        )

        def intercept(next_fun: Any, call_args: tuple[Any, ...], call_kwargs: dict[str, Any],
                      context: nn.module.InterceptorContext) -> Any:
            if isinstance(context.module, CraigSulemBlock) and context.method_name == "__call__":
                eta_features, xi_phys, log_depth = call_args
                if context.module.name != f"cs_block_{args.n_blocks - 1}":
                    return jnp.zeros_like(xi_phys)
                groups = [variables["params"][f"cs_block_{i}"] for i in range(args.n_blocks)]
                ordinary_groups = args.n_blocks - args.cs_nested_blocks
                ordinary_branches = ordinary_groups * args.latent
                head_kernels = [group["phi_proj"]["kernel"] for group in groups[:ordinary_groups]]
                if model.pair_coupling:
                    head_kernels += [group["phi_cross"]["kernel"] for group in groups[:ordinary_groups]]
                weights = (eta_features @ jnp.concatenate(head_kernels, axis=-1)
                           if ordinary_groups else jnp.zeros((*eta_features.shape[:2], 0), dtype=eta_features.dtype))
                filters = jnp.concatenate([
                    cast(jax.Array, multiplier_model.apply({"params": group["m_shared"]}, log_depth, nx // 2 + 1))
                    for group in groups
                ], axis=-1)
                xi_hat = jnp.fft.rfft(xi_phys, axis=-1, norm="forward")
                filtered = jnp.fft.irfft(filters * xi_hat[..., None], n=nx, axis=1, norm="forward")
                weighted = weights[..., :ordinary_branches] * filtered[..., :ordinary_branches]
                if model.pair_coupling:
                    weighted += jnp.repeat(weights[..., ordinary_branches:], 2, axis=-1) * filtered[
                        ..., jnp.arange(ordinary_branches) ^ 1
                    ]
                if args.cs_nested_blocks:
                    nested_eta = call_kwargs["eta_norm"][..., None]
                    if args.cs_nested_surface_features:
                        surface_mix = jnp.concatenate([
                            group["surface_mix"] for group in groups[ordinary_groups:]
                        ], axis=-1)
                        nested_eta = nested_eta + call_kwargs["nested_features"] @ surface_mix
                    inner_filters = jnp.concatenate([
                        cast(jax.Array, multiplier_model.apply({"params": group["m_inner"]}, log_depth, nx // 2 + 1))
                        for group in groups[ordinary_groups:]
                    ], axis=-1)
                    inner = jnp.fft.irfft(
                        inner_filters * jnp.fft.rfft(nested_eta * filtered[..., ordinary_branches:], axis=1, norm="forward"),
                        n=nx, axis=1, norm="forward",
                    )
                    gains = jnp.concatenate([group["gamma"] for group in groups[ordinary_groups:]])
                    weighted = jnp.concatenate((weighted, nested_eta * inner * gains), axis=-1)
                weighted_hat = jnp.fft.rfft(weighted, axis=1, norm="forward")
                correction_hat = jnp.sum(filters * weighted_hat, axis=-1)
                if call_kwargs["correction_hat_addition"] is not None:
                    correction_hat = correction_hat + call_kwargs["correction_hat_addition"]
                return jnp.fft.irfft(correction_hat, n=nx, axis=-1, norm="forward")
            return next_fun(*call_args, **call_kwargs)

        with nn.intercept_methods(intercept):
            return model.apply(variables, inputs, depth, return_baseline=return_baseline)

    # Set the learning-rate schedule and initialize AdamW's update state.
    schedule_steps = schedule_epochs * train_steps_per_epoch
    if args.lr_hold_fraction > 0:
        hold_steps = int(schedule_steps * args.lr_hold_fraction)
        if hold_steps < args.lr_warmup_steps:
            parser.error("--lr_hold_fraction must end at or after --lr_warmup_steps")
        lr_schedule = optax.join_schedules(
            [optax.warmup_constant_schedule(0.0, args.lr, args.lr_warmup_steps)
             if args.lr_warmup_steps > 0 else optax.constant_schedule(args.lr),
             optax.cosine_decay_schedule(init_value=args.lr, decay_steps=schedule_steps - hold_steps)],
            boundaries=[hold_steps],
        )
    elif args.lr_warmup_steps > 0:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=args.lr_warmup_steps,
            decay_steps=schedule_steps,
            end_value=0.0,
        )
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=schedule_steps,
        )
    if args.lr_continuation > 0:
        lr_schedule = optax.join_schedules(
            [lr_schedule,
             optax.warmup_constant_schedule(0.0, args.lr_continuation, args.lr_warmup_steps)
             if args.lr_warmup_steps > 0 else optax.constant_schedule(args.lr_continuation)],
            boundaries=[schedule_steps],
        )
        print(f"LR continuation after update {schedule_steps}: "
              f"{args.lr_warmup_steps}-update ramp to {args.lr_continuation}, then constant")
    optimizer_transforms: list[optax.GradientTransformation] = [
        optax.ema(args.gradient_ema_decay, debias=True),
        optax.adamw(learning_rate=lr_schedule, b1=args.adam_b1, b2=args.adam_b2,
                    weight_decay=args.weight_decay),
    ]
    if args.surface_head_lr_multiplier != 1.0:
        head_mask = jax.tree_util.tree_map_with_path(
            lambda path, _: len(path) == 3 and getattr(path[0], "key").startswith("cs_block_")
            and path[1:] == (jax.tree_util.DictKey("phi_proj"), jax.tree_util.DictKey("kernel")),
            params,
        )
        optimizer_transforms.append(optax.masked(optax.scale(args.surface_head_lr_multiplier), head_mask))
    optimizer = optax.chain(*optimizer_transforms)
    print(f"gradient EMA before AdamW: decay={args.gradient_ema_decay}, debias=True; "
          f"AdamW b1={args.adam_b1}, b2={args.adam_b2}")
    training_state = WeightEMATrainState.create(
        apply_fn=apply_model, params=params, tx=optimizer,
        weight_ema_params=params if args.weight_ema_decay > 0.0 else None,
        weight_ema_decay=args.weight_ema_decay,
        weight_ema_start_step=int(schedule_steps * args.weight_ema_start_fraction),
        weight_ema_secondary_params=params if args.weight_ema_secondary_decay > 0.0 else None,
        weight_ema_secondary_decay=args.weight_ema_secondary_decay,
    )
    if args.weight_ema_decay > 0.0:
        print(f"inference-only weight EMA: decay={args.weight_ema_decay}, "
              f"starts after update {training_state.weight_ema_start_step}")
    if args.weight_ema_secondary_decay > 0.0:
        print(f"secondary inference-only weight EMA: decay={args.weight_ema_secondary_decay}, "
              f"starts after update {training_state.weight_ema_start_step}")

    training_state = replicate_pytree_from_host(training_state, replicated)
    checkpoint_async_manager = checkpoints.AsyncManager(max_workers=1)

    # Save the resolved settings and dataset sizes alongside the checkpoints.
    config_payload: dict[str, object] = {
        **vars(args),
        "run_name": run_name,
        "device": backend,
        "device_count": n_devices,
        "param_count": sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)),
        "train_examples": int(train_indices.shape[0]),
        "val_examples": int(val_indices.shape[0]),
        "domain_length": domain_length,
        "xi_scale": xi_scale,
        "eta_scale": eta_scale,
        "target_scale": target_scale,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    norm_inputs_jax, norm_targets_jax, denorm_targets_jax = make_normalizers(
        stats, args.norm, dtype=np.dtype(physical_dtype),
    )

    log_h_max = float(np.log(5.0))
    hadamard_microbatch_local = args.hadamard_microbatch // n_devices
    _, _k_grid = build_grid(nx, domain_length)
    k_grid_jax = jnp.asarray(_k_grid, dtype=physical_dtype)
    k_rfft_jax = jnp.abs(k_grid_jax[: nx // 2 + 1])

    # Full-batch losses shared by training and validation.
    def compute_loss_components(
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
        *,
        collect_family_metrics: bool = False,
        n_data_only_rows: int = 0,
        terms_phys: jax.Array | None = None,
        full_spectrum_rows: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
        carrier_modes = sideband_offsets = None
        if args.phase_mechanism == "bf_sideband":
            carrier_modes, sideband_offsets = family[:, 1], family[:, 2]
            family = family[:, 0]
        target_residual64 = None
        if args.fp64_residual:
            physical_inputs64 = norm_inputs_jax(eta.astype(jnp.float64), xi.astype(jnp.float64))
            physical_depth64 = batch_depth.astype(jnp.float64)
            baseline64 = (
                cast(jax.Array, model.apply({}, physical_inputs64[..., 1], physical_depth64,
                                           method=model._linear_baseline))
                + cast(jax.Array, model.apply({}, physical_inputs64[..., 0], physical_inputs64[..., 1],
                                             physical_depth64, method=model._g1_baseline))
            )[..., None]
            target_residual64 = norm_targets_jax(gxi.astype(jnp.float64)) - baseline64
        eta, xi, gxi, batch_depth = map(physical_dtype, (eta, xi, gxi, batch_depth))
        batch_targets = norm_targets_jax(gxi)
        predictions, baseline, m1_baseline, correction = apply_model(
            {"params": current_params}, norm_inputs_jax(eta, xi), batch_depth,
            return_baseline=True,
            **({"terms_phys": terms_phys} if terms_phys is not None else {}),
        )
        pred_hat = jnp.fft.rfft(predictions, axis=1)
        target_hat = jnp.fft.rfft(batch_targets, axis=1)
        baseline_hat = jnp.fft.rfft(baseline, axis=1)
        error_hat = (jnp.fft.rfft(predictions - batch_targets, axis=1)
                     if args.subtract_before_fft else pred_hat - target_hat)
        physical_error64 = None
        if target_residual64 is not None:
            residual64 = correction.astype(jnp.float64) - target_residual64
            error_hat = jnp.fft.rfft(residual64, axis=1)
            physical_error64 = denorm_targets_jax(residual64)[..., 0]
        squared_error = jnp.sum(jnp.abs(error_hat.astype(jnp.complex128)) ** 2, axis=(1, 2))
        zero_error = squared_error == 0.0
        error_norm = jnp.where(zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error)))
        target_norm = jnp.sqrt(jnp.maximum(
            jnp.sum(jnp.abs(target_hat.astype(jnp.complex128)) ** 2, axis=(1, 2)), 1e-12,
        ))
        residual_squared = jnp.sum(
            jnp.abs((target_hat - baseline_hat).astype(jnp.complex128)) ** 2, axis=(1, 2),
        )
        residual_denominator = jnp.sqrt(residual_squared + (args.residual_floor * target_norm) ** 2)
        data_error_norm, data_target_norm, data_residual_denominator = error_norm, target_norm, residual_denominator
        data_squared_error: jax.Array = squared_error
        data_mask: jax.Array | bool = True
        if args.data_k_max is not None:
            data_mask = (k_rfft_jax <= args.data_k_max)[None, :, None]
        if full_spectrum_rows is not None:
            data_mask = data_mask | full_spectrum_rows[:, None, None]
        if args.data_k_max is not None or full_spectrum_rows is not None:
            data_squared_error = jnp.sum(jnp.abs(jnp.where(
                data_mask, error_hat.astype(jnp.complex128), 0.0,
            )) ** 2, axis=(1, 2))
            data_zero_error = data_squared_error == 0.0
            data_error_norm = jnp.where(data_zero_error, 0.0, jnp.sqrt(jnp.where(data_zero_error, 1.0, data_squared_error)))
            data_target_norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(jnp.where(
                data_mask, target_hat.astype(jnp.complex128), 0.0,
            )) ** 2, axis=(1, 2)), 1e-12))
            data_residual_denominator = jnp.sqrt(jnp.sum(jnp.abs(jnp.where(
                data_mask, (target_hat - baseline_hat).astype(jnp.complex128), 0.0,
            )) ** 2, axis=(1, 2)) + (args.residual_floor * data_target_norm) ** 2)
        if args.data_modal_residual_floor > 0.0:
            modal_mask = data_mask & (k_rfft_jax > 0.0)[None, :, None]
            modal_residual = (target_hat.astype(jnp.complex128)
                              - baseline_hat.astype(jnp.complex128))
            modal_power = jnp.where(modal_mask, jnp.abs(modal_residual) ** 2, 0.0)
            modal_floor = (args.data_modal_residual_floor**2 * jnp.max(modal_power, axis=(1, 2))
                           + (args.residual_floor * data_target_norm) ** 2)
            inverse_scale = jax.lax.rsqrt(modal_power + modal_floor[:, None, None])
            residual_power = jnp.sum(modal_power, axis=(1, 2))
            scaled_power = jnp.sum(inverse_scale * modal_power, axis=(1, 2))
            modal_weight = jnp.where(
                modal_mask & (residual_power > 0.0)[:, None, None],
                inverse_scale * (
                    residual_power / jnp.where(residual_power > 0.0, scaled_power, 1.0)
                )[:, None, None],
                1.0,
            )
            # Keep DC and the target denominator unchanged; weights are label-only.
            data_squared_error = jnp.sum(jnp.where(
                data_mask,
                jax.lax.stop_gradient(modal_weight) * jnp.abs(error_hat.astype(jnp.complex128)) ** 2,
                0.0,
            ), axis=(1, 2))
            data_zero_error = data_squared_error == 0.0
            data_error_norm = jnp.where(
                data_zero_error, 0.0,
                jnp.sqrt(jnp.where(data_zero_error, 1.0, data_squared_error)),
            )
        if args.data_phase_projection_weight > 0.0:
            eta_hat = jnp.fft.rfft(eta, axis=1).astype(jnp.complex128)
            phase_band = ((k_rfft_jax > 0.0) & (k_rfft_jax <= 128.0))[None, :, None] & data_mask
            eta_power = jnp.abs(eta_hat) ** 2
            phase_active = phase_band[..., 0] & (
                eta_power > 1e-12 * jnp.max(eta_power[:, 1:], axis=1, keepdims=True)
            )
            phase_error = jnp.where(
                phase_active,
                jnp.imag(error_hat[..., 0].astype(jnp.complex128) * jnp.conj(eta_hat))
                / jnp.sqrt(jnp.where(phase_active, eta_power, 1.0)),
                0.0,
            )
            data_squared_error = data_squared_error + args.data_phase_projection_weight * jnp.sum(
                phase_error**2, axis=1,
            )
            data_zero_error = data_squared_error == 0.0
            data_error_norm = jnp.where(data_zero_error, 0.0, jnp.sqrt(jnp.where(data_zero_error, 1.0, data_squared_error)))
        if args.data_xi_projection_weight > 0.0:
            xi_hat = jnp.fft.rfft(xi, axis=1).astype(jnp.complex128)
            projection_mask = ((k_rfft_jax > 0.0) & (k_rfft_jax <= 128.0))[None, :, None] & data_mask
            xi_direction = jnp.where(projection_mask[..., 0], xi_hat, 0.0)
            projection_error = jnp.where(
                projection_mask, error_hat.astype(jnp.complex128), 0.0,
            )[..., 0]
            direction_squared = jnp.sum(jnp.abs(xi_direction) ** 2, axis=1)
            projection_dot = jnp.real(jnp.sum(jnp.conj(xi_direction) * projection_error, axis=1))
            direction_active = direction_squared > jnp.finfo(xi.dtype).eps * jnp.sum(
                jnp.abs(xi_hat[:, 1:]) ** 2, axis=1,
            )
            projection_squared = jnp.where(
                direction_active,
                projection_dot**2 / jnp.where(direction_active, direction_squared, 1.0),
                0.0,
            )
            emphasized_squared_error: jax.Array = data_squared_error + args.data_xi_projection_weight * projection_squared
            emphasized_zero_error = emphasized_squared_error == 0.0
            data_error_norm = jnp.where(
                emphasized_zero_error, 0.0,
                jnp.sqrt(jnp.where(emphasized_zero_error, 1.0, emphasized_squared_error)),
            )
        denominator = data_target_norm
        if args.model == "cs_dno":
            if args.data_normalization == "residual":
                denominator = data_residual_denominator
            elif args.data_normalization == "bounded":
                residual_ratio = data_residual_denominator / data_target_norm
                row_weight = 1.0 + args.nonlinear_reference / (residual_ratio + args.nonlinear_reference)
                denominator = data_target_norm * args.nonlinear_weight_mean / row_weight
            elif args.data_normalization == "nonlinear":
                # Nonlinear weighting uses the M1 correction even with a G2 baseline.
                m1_error_hat = (target_hat - jnp.fft.rfft(m1_baseline, axis=1)).astype(jnp.complex128)
                if args.data_k_max is not None:
                    m1_error_hat = jnp.where(data_mask, m1_error_hat, 0.0)
                residual_ratio = jnp.sqrt(jnp.sum(
                    jnp.abs(m1_error_hat) ** 2, axis=(1, 2),
                )) / data_target_norm
                nonlinear_weight = 1.0 + residual_ratio / (residual_ratio + args.nonlinear_reference)
                denominator = data_target_norm * args.nonlinear_weight_mean / nonlinear_weight
        denominator = jax.lax.stop_gradient(denominator)
        per_row_data_loss = data_error_norm / denominator
        if full_spectrum_rows is not None and args.rollout_training_rhs:
            rhs_error = compute_rhs_relative_error(
                eta, xi, denorm_targets_jax(predictions)[..., 0], gxi,
                jnp.exp(batch_depth[:, 0].astype(jnp.float64)), k_grid_jax,
            )
            per_row_data_loss = jnp.where(full_spectrum_rows, rhs_error, per_row_data_loss)
        data_loss = jnp.mean(per_row_data_loss)
        full_l2_loss = jnp.mean(error_norm / target_norm)
        family_metrics: tuple[jax.Array, ...] = ()
        if collect_family_metrics:
            row_metrics = jnp.stack((
                jnp.ones_like(error_norm), error_norm / target_norm,
                error_norm / residual_denominator, squared_error < residual_squared,
                per_row_data_loss,
            ))
            # Masked means, not means of each shard's family means: pmean remains exact.
            family_metrics = tuple(
                value for family_id in PhysicalFamilyId
                for value in jnp.mean(jnp.where(family == family_id, row_metrics, 0.0), axis=1)
            )
        # Data fit uses normalized Gxi; the physics losses below use physical fields.
        h_phys = jnp.exp(jnp.minimum(batch_depth[:, 0], log_h_max))
        gxi_prediction = denorm_targets_jax(predictions)[..., 0]
        # Paired probe rows contribute only to data fit; auxiliary means use originals.
        full_batch_size = eta.shape[0]
        n_auxiliary_rows = full_batch_size - n_data_only_rows
        if n_auxiliary_rows == 0:
            zero = jnp.float32(0.0)
            return data_loss, zero, zero, zero, zero, full_l2_loss, family_metrics
        eta, xi, gxi, batch_depth, family, h_phys, gxi_prediction, batch_targets = (
            value[:n_auxiliary_rows]
            for value in (eta, xi, gxi, batch_depth, family, h_phys, gxi_prediction, batch_targets)
        )
        mode_loss = jnp.float32(0.0)
        if args.mode_balanced_weight > 0.0:
            mode_loss = compute_mode_balanced_loss(
                eta=eta,
                gxi_prediction=gxi_prediction,
                gxi_target=denorm_targets_jax(batch_targets)[..., 0],
                depth=h_phys,
                k_rfft=k_rfft_jax,
                k_max=args.mode_balanced_k_max,
                activity_threshold=args.mode_balanced_activity_threshold,
                denominator_eps=args.mode_balanced_denominator_eps,
                subtract_before_fft=args.subtract_before_fft,
                gxi_error=None if physical_error64 is None else physical_error64[:n_auxiliary_rows],
                sample_weight=cast(jax.Array, jnp.where(family == PhysicalFamilyId.JONSWAP_TMA,
                                                       training_dtype(args.jonswap_mode_balanced_scale), training_dtype(1.0))),
            )
        tangent_loss = jnp.float32(0.0)
        local_selected = jnp.float32(0.0)
        if args.translation_tangent_weight > 0.0:
            tangent_options: dict[str, Any] = dict(
                sample_mask=(family == PhysicalFamilyId.TANAKA) if args.translation_tangent_scope == "tanaka" else None,
                denominator_eps=args.translation_tangent_denominator_eps,
                loss_norm=args.translation_tangent_loss_norm,
                charbonnier_eps=args.translation_tangent_charbonnier_eps,
                subtract_before_centering=args.subtract_before_centering,
            )
            if args.translation_tangent_geometry == "coupled_modal":
                tangent_loss, local_selected = compute_modal_phase_loss(
                    eta, xi, gxi_prediction, gxi, jnp.exp(batch_depth[:, 0]), k_grid_jax,
                    k_max=args.translation_tangent_k_max, **tangent_options,
                )
            else:
                tangent_loss, local_selected = compute_translation_tangent_loss(
                    eta, gxi_prediction, gxi, h_phys, k_grid_jax,
                    global_relative_mask=(family == PhysicalFamilyId.STOKES) if args.stokes_relative_tangent else None,
                    smoothing_scale=args.translation_tangent_smoothing_scale, **tangent_options,
                )
            # Weight each GPU's contribution by its number of selected nonflat samples.
            global_selected = jax.lax.psum(local_selected, axis_name="batch")
            device_count = jax.lax.psum(jnp.float32(1.0), axis_name="batch")
            shard_weight = (
                device_count
                * local_selected
                / jnp.maximum(global_selected, jnp.float32(1.0))
            )
            tangent_loss = tangent_loss * shard_weight
        mechanism_loss = jnp.float32(0.0)
        if args.phase_mechanism == "stokes_carrier":
            mechanism_loss, mechanism_count = compute_phase_mechanism_loss(
                eta, xi, gxi_prediction, gxi, jnp.exp(batch_depth[:, 0]), k_grid_jax,
                mechanism="carrier", sample_mask=family == PhysicalFamilyId.STOKES, horizon=20.0,
                subtract_before_centering=args.subtract_before_centering,
            )
            mechanism_loss *= mechanism_count / full_batch_size
        elif args.phase_mechanism in ("tanaka_momentum", "all_momentum"):
            mechanism_loss, mechanism_count = compute_phase_mechanism_loss(
                eta, xi, gxi_prediction, gxi, jnp.exp(batch_depth[:, 0]), k_grid_jax,
                mechanism="momentum", horizon=200.0,
                subtract_before_centering=args.subtract_before_centering,
                sample_mask=family == PhysicalFamilyId.TANAKA if args.phase_mechanism == "tanaka_momentum" else None,
            )
            mechanism_loss *= mechanism_count / full_batch_size
        elif args.phase_mechanism == "bf_sideband":
            mechanism_loss, mechanism_count = compute_phase_mechanism_loss(
                eta, xi, gxi_prediction, gxi, jnp.exp(batch_depth[:, 0]), k_grid_jax,
                mechanism="sideband", sample_mask=family == PhysicalFamilyId.BENJAMIN_FEIR, horizon=200.0,
                subtract_before_centering=args.subtract_before_centering,
                carrier_mode=cast(jax.Array, carrier_modes)[:n_auxiliary_rows],
                sideband_offset=cast(jax.Array, sideband_offsets)[:n_auxiliary_rows],
            )
            mechanism_loss *= mechanism_count / full_batch_size
        return data_loss, mode_loss, tangent_loss, mechanism_loss, local_selected / full_batch_size, full_l2_loss, family_metrics

    def hadamard_loss_on_subset(
        rng: jax.Array,
        current_params: FlatParams,
        eta: jax.Array,
        xi: jax.Array,
        batch_depth: jax.Array,
        flow_target: jax.Array | None = None,
    ) -> jax.Array:
        rng_perm, rng_probe = jax.random.split(rng)
        sample_indices = jax.random.permutation(rng_perm, eta.shape[0])[
            :hadamard_microbatch_local
        ]
        eta_sub = eta.astype(physical_dtype)[sample_indices]
        xi_sub = xi.astype(physical_dtype)[sample_indices]
        batch_depth_sub = batch_depth.astype(physical_dtype)[sample_indices].astype(jnp.float64)
        response_loss = partial(
            compute_hadamard_loss,
            flow_target=flow_target[sample_indices] if flow_target is not None else None,
        ) if args.hadamard_mode == "self" else partial(
            compute_local_response_loss, mode=args.hadamard_mode, length=domain_length,
        )
        return response_loss(
            rng=rng_probe,
            apply_fn=apply_model,
            model_params=current_params,
            eta_phys=eta_sub,
            xi_phys=xi_sub,
            batch_depth_local=batch_depth_sub,
            norm_inputs_fn=norm_inputs_jax,
            denorm_targets_fn=denorm_targets_jax,
            k=k_grid_jax,
            dtype=jnp.float64,
            k_max=args.hadamard_k_max,
            sobolev_order=args.hadamard_sobolev_order,
            fd_step_min=args.hadamard_fd_step_min,
            fd_step_max=args.hadamard_fd_step_max,
            min_surface_rms=args.hadamard_min_surface_rms,
            denominator_eps=args.hadamard_denominator_eps,
        )

    def train_step(
        current_state: WeightEMATrainState,
        rng_key: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
        terms_phys: jax.Array | None = None,
    ) -> tuple[WeightEMATrainState, jax.Array, dict[str, jax.Array]]:
        if args.datum_shift_rms > 0:
            datum_mask_key, datum_shift_key = jax.random.split(jax.random.fold_in(
                jax.random.fold_in(rng_key, current_state.step),
                jax.lax.axis_index("batch") + 16381,
            ))
            shifted = jax.random.bernoulli(datum_mask_key, shape=(eta.shape[0], 1))
            eta = eta.astype(jnp.float64)
            depth = jnp.exp(batch_depth.astype(jnp.float64))
            datum_shift = jnp.where(shifted, jax.random.uniform(
                datum_shift_key, (eta.shape[0], 1), dtype=jnp.float64, minval=-1.0, maxval=1.0,
            ) * jnp.minimum(args.datum_shift_rms * jnp.sqrt(jnp.mean(eta**2, axis=-1, keepdims=True)),
                            0.25 * depth), 0.0)
            eta = eta - datum_shift
            batch_depth = jnp.where(shifted, jnp.log(depth + datum_shift), batch_depth.astype(jnp.float64))
        if args.reflection_augmentation:
            reflection_key = jax.random.fold_in(
                jax.random.fold_in(rng_key, current_state.step),
                jax.lax.axis_index("batch") + 12289,
            )
            reflected = jax.random.bernoulli(reflection_key, shape=(eta.shape[0], 1))
            reflection_indices = (-jnp.arange(nx)) % nx
            eta, xi, gxi = (jnp.where(reflected, field[:, reflection_indices], field)
                            for field in (eta, xi, gxi))
        if args.input_noise_rms > 0:
            noise_key = jax.random.fold_in(
                jax.random.fold_in(rng_key, current_state.step),
                jax.lax.axis_index("batch") + 8191,
            )
            fields = jnp.stack((eta, xi))
            centered = fields - jnp.mean(fields, axis=-1, keepdims=True)
            raw_noise = jax.random.normal(noise_key, fields.shape, dtype=fields.dtype)
            noise_hat = jnp.fft.rfft(raw_noise, axis=-1) * (
                ((k_rfft_jax > 0) & (k_rfft_jax <= 128)) / (1 + k_rfft_jax**2)
            )
            noise = jnp.fft.irfft(noise_hat, n=nx, axis=-1)
            noise *= args.input_noise_rms * jnp.sqrt(jnp.mean(centered**2, axis=-1, keepdims=True)) / jnp.maximum(
                jnp.sqrt(jnp.mean(noise**2, axis=-1, keepdims=True)), 1e-30,
            )
            eta_noisy, xi_noisy = fields + noise
            valid = jnp.all(eta_noisy + jnp.exp(batch_depth.astype(jnp.float64)) > 0, axis=-1)
            eta, xi = (cast(jax.Array, jnp.where(valid[:, None], noisy, original))
                       for noisy, original in ((eta_noisy, eta), (xi_noisy, xi)))
        n_data_only_rows = (
            int(args.dirichlet_probe_fraction * eta.shape[0])
            if args.dirichlet_probe_mode != "off" else 0
        )
        n_auxiliary_rows = eta.shape[0] - n_data_only_rows
        if n_data_only_rows:
            probe_xi = xi[:n_data_only_rows].astype(jnp.float64)
            if args.dirichlet_probe_mode == "scrambled":
                probe_rng = jax.random.fold_in(
                    jax.random.fold_in(rng_key, current_state.step),
                    jax.lax.axis_index("batch") + 4099,
                )
                probe_hat = jnp.fft.rfft(probe_xi, axis=-1)
                phases = jax.random.uniform(
                    probe_rng, probe_hat.shape, dtype=jnp.float64,
                    minval=0.0, maxval=2.0 * jnp.pi,
                )
                probe_hat = jnp.where(
                    (k_rfft_jax > 0.0) & (k_rfft_jax <= 128.0),
                    jnp.abs(probe_hat) * jnp.exp(1j * phases), 0.0,
                )
                probe_xi = jnp.fft.irfft(probe_hat, n=nx, axis=-1)
            _, probe_xi, probe_target = compute_dno_target(
                eta[:n_data_only_rows], probe_xi,
                jnp.exp(batch_depth[:n_data_only_rows].astype(jnp.float64)),
                nx=nx, length=domain_length, dno_order=6, pad_factor=8,
                maximum_wavenumber=128.0,
            )
            eta, xi, gxi, batch_depth, family = (
                jnp.concatenate((original[:n_auxiliary_rows], tail.astype(original.dtype)), axis=0)
                for original, tail in (
                    (eta, eta[:n_data_only_rows]), (xi, probe_xi), (gxi, probe_target),
                    (batch_depth, batch_depth[:n_data_only_rows]),
                    (family, family[:n_data_only_rows]),
                )
            )

        rollout_batch = None
        coherent_batch = None
        amplitude_batch = None
        if args.rollout_training_weight > 0:
            rollout_scheduled = (
                (current_state.step >= (args.rollout_training_start_epoch - 1) * train_steps_per_epoch)
                & (current_state.step % args.rollout_training_interval == 0)
            )
            n_rollout = min(args.rollout_training_microbatch // n_devices, n_auxiliary_rows)
            rollout_depth = batch_depth[:n_rollout]
            source_fields = (eta[:n_rollout], xi[:n_rollout], gxi[:n_rollout])

            def generate_rollout_batch(_: None) -> tuple[jax.Array, ...]:
                frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, current_state.params)
                depth = jnp.exp(rollout_depth.astype(jnp.float64))

                def predict_frozen(eta_phys: jax.Array, xi_phys: jax.Array) -> jax.Array:
                    prediction = apply_model(
                        {"params": frozen_params},
                        norm_inputs_jax(physical_dtype(eta_phys), physical_dtype(xi_phys)),
                        physical_dtype(rollout_depth),
                    )
                    physical = denorm_targets_jax(prediction)[..., 0].astype(jnp.float64)
                    return physical - jnp.mean(physical, axis=-1, keepdims=True)

                trajectory = rollout_surrogate(
                    ti.State(*map(lambda value: jax.lax.stop_gradient(value.astype(jnp.float64)), source_fields[:2])),
                    jnp.array([0.0, 0.01 * args.rollout_training_steps], dtype=jnp.float64),
                    ti.make_solver_params(
                        nx, domain_length, depth,
                        filter_fraction=min(1.0, 128.0 * domain_length / (jnp.pi * nx)),
                    ),
                    predict_frozen, substeps=args.rollout_training_steps,
                    **({"return_stages": True} if args.rollout_coherent_weight > 0 else {}),
                )
                stage_data = ()
                if args.rollout_coherent_weight > 0:
                    stage_eta, stage_xi = map(jax.lax.stop_gradient, (
                        trajectory["stage_eta"], trajectory["stage_xi"],
                    ))
                    stage_valid = jnp.all(
                        jnp.isfinite(stage_eta) & jnp.isfinite(stage_xi)
                        & jnp.isfinite(physical_dtype(stage_eta)) & jnp.isfinite(physical_dtype(stage_xi))
                        & (stage_eta + depth > 0), axis=(0, 1, 3),
                    )
                    stage_eta, stage_xi = (
                        jnp.where(stage_valid[None, None, :, None], value, 0.0)
                        for value in (stage_eta, stage_xi)
                    )
                    # Label the actual stage, without projecting its input spectrum.
                    stage_target = dno_series_eval(
                        stage_eta.reshape(-1, nx), stage_xi.reshape(-1, nx),
                        k_grid_jax.astype(jnp.float64),
                        jnp.broadcast_to(depth, stage_eta.shape[:-1] + (1,)).reshape(-1, 1),
                        6, pad_factor=8,
                    ).reshape(stage_eta.shape)
                    stage_valid &= jnp.all(jnp.isfinite(stage_target), axis=(0, 1, 3))
                    stage_data = (jax.tree.map(jax.lax.stop_gradient, (
                        *(jnp.where(stage_valid[None, None, :, None], value, 0.0)
                          for value in (stage_eta, stage_xi, stage_target)),
                        stage_valid, trajectory["stage_times"], trajectory["stage_weights"],
                    )),)
                eta_new, xi_new = map(jax.lax.stop_gradient, (trajectory["eta"][-1], trajectory["xi"][-1]))
                valid = jnp.all(jnp.isfinite(eta_new) & jnp.isfinite(xi_new) & (eta_new + depth > 0), axis=-1)
                # Invalid endpoints are sanitized before M6 and fall back to original supervised rows.
                eta_new, xi_new, target_new = compute_dno_target(
                    jnp.where(valid[:, None], eta_new, source_fields[0]),
                    jnp.where(valid[:, None], xi_new, source_fields[1]), depth,
                    nx=nx, length=domain_length, dno_order=6, pad_factor=8,
                    maximum_wavenumber=128.0,
                    project_output=not args.rollout_training_full_spectrum,
                )
                if args.rollout_training_full_spectrum:
                    target_new -= jnp.mean(target_new, axis=-1, keepdims=True)
                generated_fields = tuple(
                    generated.astype(original.dtype)
                    for generated, original in zip((eta_new, xi_new, target_new), source_fields)
                )
                valid &= jnp.all(jnp.isfinite(jnp.stack(generated_fields)), axis=(0, 2))
                valid &= jnp.all(eta_new + depth > 0, axis=-1)
                fields = tuple(
                    jax.lax.stop_gradient(cast(jax.Array, jnp.where(valid[:, None], generated, original)))
                    for generated, original in zip(generated_fields, source_fields)
                )
                return (
                    *fields, jnp.mean(valid.astype(jnp.float32)),
                    *((jax.lax.stop_gradient(valid),) if args.rollout_training_full_spectrum else ()),
                    *stage_data,
                )

            rollout_fields = jax.lax.cond(
                rollout_scheduled, generate_rollout_batch,
                lambda _: (
                    *map(jax.lax.stop_gradient, source_fields), jnp.float32(0.0),
                    *((jnp.zeros(n_rollout, dtype=bool),) if args.rollout_training_full_spectrum else ()),
                    *(((*(
                        jnp.zeros((args.rollout_training_steps, 2, n_rollout, nx), dtype=jnp.float64)
                        for _ in range(3)), jnp.zeros(n_rollout, dtype=bool),
                        jnp.zeros((args.rollout_training_steps, 2), dtype=jnp.float64),
                        jnp.zeros((args.rollout_training_steps, 2), dtype=jnp.float64)),)
                      if args.rollout_coherent_weight > 0 else ()),
                ), None,
            )
            if args.rollout_coherent_weight > 0:
                coherent_batch = (rollout_scheduled, rollout_depth, *rollout_fields[-1])
                rollout_fields = rollout_fields[:-1]
            rollout_batch = (
                rollout_scheduled, rollout_depth, family[:n_rollout], *rollout_fields,
            )
            if args.amplitude_probe_scale > 0:
                def generate_amplitude_batch(_: None) -> tuple[jax.Array, ...]:
                    depth = jnp.exp(rollout_depth.astype(jnp.float64))
                    generated = compute_dno_target(
                        source_fields[0] * args.amplitude_probe_scale, source_fields[1], depth,
                        nx=nx, length=domain_length, dno_order=6, pad_factor=8,
                        maximum_wavenumber=128.0,
                    )
                    generated_fields = tuple(
                        field.astype(original.dtype)
                        for field, original in zip(generated, source_fields)
                    )
                    valid = jnp.all(jnp.isfinite(jnp.stack(generated_fields)), axis=(0, 2))
                    valid &= jnp.all(generated_fields[0] + depth > 0, axis=-1)
                    fields = tuple(
                        jax.lax.stop_gradient(cast(jax.Array, jnp.where(valid[:, None], field, original)))
                        for field, original in zip(generated_fields, source_fields)
                    )
                    return *fields, jax.lax.stop_gradient(valid)

                amplitude_batch = jax.lax.cond(
                    rollout_scheduled, generate_amplitude_batch,
                    lambda _: (*map(jax.lax.stop_gradient, source_fields), jnp.zeros(n_rollout, dtype=bool)), None,
                )

        def compute_training_loss(
            current_params: FlatParams,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            data_loss, mode_loss, tangent_loss, mechanism_loss, tangent_fraction, full_l2_loss, _ = compute_loss_components(
                current_params, eta, xi, gxi, batch_depth, family,
                n_data_only_rows=n_data_only_rows,
                terms_phys=terms_phys,
            )
            physics_loss = jnp.asarray(0.0, dtype=training_dtype)
            mode_weight = args.mode_balanced_weight * jnp.minimum(
                jnp.asarray(current_state.step, dtype=training_dtype)
                / max(args.mode_balanced_warmup_steps, 1),
                1.0,
            )
            mechanism_weight = args.phase_mechanism_weight * (
                jnp.clip(current_state.step / (args.phase_mechanism_ramp_epochs * train_steps_per_epoch), 0.0, 1.0)
                if args.phase_mechanism_ramp_epochs > 0 else 1.0
            )
            metrics: dict[str, jax.Array] = {
                "train_l2_loss": full_l2_loss, "train_data_loss": data_loss,
            }
            if rollout_batch is not None:
                scheduled, depth_sub, family_sub, eta_sub, xi_sub, target_sub, valid_fraction = rollout_batch[:7]
                full_spectrum_rows = rollout_batch[7] if args.rollout_training_full_spectrum else None
                rollout_loss, rollout_mode_loss, rollout_tangent_loss, rollout_mechanism_loss = jax.lax.cond(
                    scheduled,
                    lambda p: tuple(map(jnp.float64, compute_loss_components(
                        p, eta_sub, xi_sub, target_sub, depth_sub, family_sub,
                        n_data_only_rows=eta_sub.shape[0] if args.rollout_training_loss == "data" else 0,
                        full_spectrum_rows=full_spectrum_rows,
                    )[:4])),
                    lambda _: (jnp.float64(0.0),) * 4, current_params,
                )
                rollout_weight = args.rollout_training_weight * (
                    jnp.clip(
                        (current_state.step - (args.rollout_training_start_epoch - 1) * train_steps_per_epoch)
                        / (args.rollout_training_ramp_epochs * train_steps_per_epoch), 0.0, 1.0,
                    ) if args.rollout_training_ramp_epochs > 0 else 1.0
                )
                data_loss += rollout_weight * rollout_loss
                physics_loss += rollout_weight * (
                    mode_weight * rollout_mode_loss
                    + args.translation_tangent_weight * rollout_tangent_loss
                    + mechanism_weight * rollout_mechanism_loss
                )
                metrics.update(
                    rollout_data_loss=rollout_loss, rollout_valid_fraction=valid_fraction,
                    rollout_mode_balanced_loss=rollout_mode_loss,
                    rollout_translation_tangent_loss=rollout_tangent_loss,
                    rollout_phase_mechanism_loss=rollout_mechanism_loss,
                    rollout_active=jnp.float32(scheduled), rollout_weight=rollout_weight * scheduled,
                )
                if amplitude_batch is not None:
                    probe_eta, probe_xi, probe_target, valid = amplitude_batch
                    def compute_amplitude_loss(p: FlatParams) -> jax.Array:
                        prediction = apply_model(
                            {"params": p},
                            norm_inputs_jax(physical_dtype(probe_eta), physical_dtype(probe_xi)),
                            physical_dtype(depth_sub),
                        )
                        error_hat = (
                            jnp.fft.rfft(prediction, axis=1)
                            - jnp.fft.rfft(norm_targets_jax(physical_dtype(probe_target)), axis=1)
                        ).astype(jnp.complex128)
                        # M6 labels stop at 128; G1's out-of-band tail scales as lambda, not lambda squared.
                        error_hat = jnp.where((k_rfft_jax <= 128.0)[None, :, None], error_hat, 0.0)
                        squared_error = jnp.sum(jnp.abs(error_hat) ** 2, axis=(1, 2))
                        zero_error = squared_error == 0.0
                        error_norm = jnp.where(zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error)))
                        source_hat = jnp.fft.rfft(norm_targets_jax(physical_dtype(gxi[:probe_eta.shape[0]])), axis=1)
                        denominator = jax.lax.stop_gradient(args.amplitude_probe_scale**2 * jnp.sqrt(jnp.maximum(
                            jnp.sum(jnp.abs(source_hat.astype(jnp.complex128)) ** 2, axis=(1, 2)), 1e-12,
                        )))
                        return jnp.mean(cast(jax.Array, jnp.where(valid, error_norm / denominator, 0.0)))

                    amplitude_loss = jax.lax.cond(
                        scheduled, compute_amplitude_loss, lambda _: jnp.float64(0.0), current_params,
                    )
                    data_loss += rollout_weight * amplitude_loss
                    metrics["amplitude_data_loss"] = amplitude_loss
            if args.mode_balanced_weight > 0.0:
                physics_loss += mode_weight * mode_loss
                metrics["mode_balanced_loss"] = mode_loss
            if args.translation_tangent_weight > 0.0:
                physics_loss += args.translation_tangent_weight * tangent_loss
                metrics["translation_tangent_loss"] = tangent_loss
                metrics["translation_tangent_fraction"] = tangent_fraction
            if args.phase_mechanism_weight > 0.0:
                physics_loss += mechanism_weight * mechanism_loss
                metrics["phase_mechanism_loss"] = mechanism_loss
            if args.hadamard_weight > 0.0:
                # Evaluate Hadamard on a small random subset, only on scheduled steps.
                hadamard_scheduled = current_state.step % args.hadamard_interval == 0

                rng_hadamard = jax.random.fold_in(
                    jax.random.fold_in(rng_key, jax.lax.axis_index("batch") * 53 + 127),
                    current_state.step,
                )
                hadamard_loss = jax.lax.cond(
                    hadamard_scheduled,
                    lambda rng: hadamard_loss_on_subset(
                        rng, current_params, eta[:n_auxiliary_rows],
                        xi[:n_auxiliary_rows], batch_depth[:n_auxiliary_rows],
                        flow_target=gxi[:n_auxiliary_rows] if args.hadamard_flow_probes else None,
                    ),
                    lambda _: jnp.float64(0.0),
                    rng_hadamard,
                )
                hadamard_weight = args.hadamard_weight * jnp.minimum(
                    jnp.asarray(current_state.step, dtype=jnp.float64)
                    / max(args.hadamard_warmup_steps, 1),
                    1.0,
                )
                physics_loss += training_dtype(hadamard_weight * hadamard_loss)
                metrics.update(
                    hadamard_active=training_dtype(hadamard_scheduled),
                    hadamard_loss=training_dtype(hadamard_loss),
                )

            return data_loss + physics_loss, metrics

        # Differentiate the total loss, average gradients across GPUs, then update weights.
        (loss_value, metrics), grads = jax.value_and_grad(
            compute_training_loss, has_aux=True
        )(current_state.params)
        if coherent_batch is not None:
            scheduled, depth_sub, stage_eta, stage_xi, stage_target, stage_valid, stage_times, stage_weights = coherent_batch
            stage_depth = jnp.broadcast_to(depth_sub, stage_eta.shape[:-1] + (1,))
            # Differentiate inside the branch: skipped updates must not allocate
            # the large stage-prediction backward tape.
            accumulated_loss, accumulated_grads = jax.lax.cond(
                scheduled,
                lambda p: jax.value_and_grad(lambda student: compute_coherent_rollout_loss(
                    stage_eta, stage_xi,
                    denorm_targets_jax(apply_model(
                        {"params": student},
                        norm_inputs_jax(physical_dtype(stage_eta.reshape(-1, nx)),
                                        physical_dtype(stage_xi.reshape(-1, nx))),
                        physical_dtype(stage_depth.reshape(-1, 1)),
                    ))[..., 0].reshape(stage_eta.shape),
                    stage_target, jnp.exp(depth_sub.astype(jnp.float64)).reshape(-1),
                    k_grid_jax, stage_times, stage_weights,
                    eta[:stage_eta.shape[2]], xi[:stage_eta.shape[2]], stage_valid,
                ))(p),
                lambda p: (jnp.float64(0.0), jax.tree.map(jnp.zeros_like, p)), current_state.params,
            )
            coherent_weight = args.rollout_coherent_weight * metrics["rollout_weight"]
            loss_value += coherent_weight * accumulated_loss
            grads = jax.tree.map(lambda g, extra: g + (coherent_weight * extra).astype(g.dtype),
                                 grads, accumulated_grads)
            metrics.update(rollout_coherent_loss=accumulated_loss,
                           rollout_coherent_valid_fraction=jnp.mean(stage_valid.astype(jnp.float32)))
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_value = jax.lax.pmean(loss_value, axis_name="batch")
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        next_state = current_state.apply_gradients(grads=grads)
        return next_state, loss_value, metrics

    # Split divisible batches across GPUs; replicate tiny remainder batches instead.
    # Creating these callables does not train the model; the epoch loop calls them.
    train_steps = {
        partition: jax.jit(
            shard_map(
                train_step,
                mesh=mesh,
                in_specs=(P(), P(), partition, partition, partition, partition, partition,
                          *((partition,) if args.cs_terms_cpu_prefetch else ())),
                out_specs=(P(), P(), P()),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
    }

    # Validation uses fixed Hadamard samples/probes and takes no parameter gradients.
    def eval_step(
        current_params: FlatParams,
        rng_key: jax.Array,
        batch_index: jax.Array,
        eta: jax.Array,
        xi: jax.Array,
        gxi: jax.Array,
        batch_depth: jax.Array,
        family: jax.Array,
        terms_phys: jax.Array | None = None,
        *,
        run_hadamard: bool,
    ) -> tuple[jax.Array, ...]:
        hadamard_loss = jnp.float32(0.0)
        if args.hadamard_weight > 0.0 and run_hadamard:
            hadamard_loss = training_dtype(hadamard_loss_on_subset(
                jax.random.fold_in(jax.random.fold_in(rng_key, batch_index), jax.lax.axis_index("batch")),
                current_params, eta, xi, batch_depth,
            ))
        data_loss, mode_loss, tangent_loss, mechanism_loss, tangent_fraction, full_l2_loss, family_metrics = compute_loss_components(
            current_params, eta, xi, gxi, batch_depth, family, collect_family_metrics=True,
            terms_phys=terms_phys,
        )
        return jax.lax.pmean(
            (data_loss, mode_loss, tangent_loss, hadamard_loss, tangent_fraction, full_l2_loss, mechanism_loss, *family_metrics),
            axis_name="batch",
        )

    eval_steps = {
        (partition, run_hadamard): jax.jit(
            shard_map(
                partial(eval_step, run_hadamard=run_hadamard),
                mesh=mesh,
                in_specs=(P(), P(), P(), partition, partition, partition, partition, partition,
                          *((partition,) if args.cs_terms_cpu_prefetch else ())),
                out_specs=(P(),) * (7 + 5 * len(PhysicalFamilyId)),
                check_rep=False,
            )
        )
        for partition in (P("batch"), P())
        for run_hadamard in (False, True)
    }

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    train_loss = float("inf")
    start_epoch = 1

    # Resume the last saved epoch, including optimizer state and metric history.
    latest_ckpt_dir = run_dir / "latest_ckpt"
    metadata_path = latest_ckpt_dir / "metadata.json"
    if metadata_path.exists():
        metadata: CheckpointMetadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        resume_epoch = metadata["epoch"]
        restore_target = {
            "params": training_state.params,
            "opt_state": training_state.opt_state,
            "step": int(training_state.step),
        }
        if args.weight_ema_decay > 0.0:
            restore_target["weight_ema_params"] = training_state.weight_ema_params
        if args.weight_ema_secondary_decay > 0.0:
            restore_target["weight_ema_secondary_params"] = training_state.weight_ema_secondary_params
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=latest_ckpt_dir,
            target=restore_target,
            step=resume_epoch,
            prefix="ckpt_",
            orbax_checkpointer=ocp.PyTreeCheckpointer(),
        )
        restored["step"] = jnp.asarray(int(restored["step"]), dtype=jnp.int32)
        training_state = training_state.replace(**restored)
        training_state = replicate_pytree_from_host(training_state, replicated)
        history = metadata["history"]
        train_loss = metadata["train_loss"]
        best_val_loss = metadata["best_val_loss"]
        best_epoch = metadata["best_epoch"]
        start_epoch = resume_epoch + 1
        print(
            f"auto-resume: restored from {latest_ckpt_dir} at epoch {resume_epoch}, "
            f"resuming at epoch {start_epoch}"
        )

    # Verify all GPU copies agree before training, including after checkpoint restore.
    # The counter helper reads AdamW; checkpoints retain the full EMA + AdamW state.
    training_counter_values(
        training_state.replace(opt_state=training_state.opt_state[1]),
        context="training startup",
        announce=True,
    )
    assert_pytree_replicated(training_state, name="training startup state")

    cpu_terms = None
    if args.cs_terms_cpu_prefetch:
        cpu_k = jax.device_put(np.abs(_k_grid[:nx // 2 + 1]), cpu_sharding)
        cpu_terms = jax.jit(
            lambda eta, xi, log_depth: dno_series_eval(
                eta, xi, cpu_k, jnp.exp(log_depth), order=args.cs_baseline_order, pad_factor=2, return_terms=True,
            ), in_shardings=(cpu_sharding,) * 3, out_shardings=cpu_sharding,
        )

    # The producer optionally prepares CPU terms while the current GPU update runs.
    def prefetch_batches(
        indices: np.ndarray, rng: np.random.Generator | None
    ) -> Iterator[tuple[jax.Array, ...]]:
        batches = get_batches(
            dataset["eta"],
            dataset["xi"],
            dataset["gxi"],
            dataset["depth"],
            indices,
            args.batch_size,
            rng,
            device_count=n_devices,
            physical_dtype=np.dtype(physical_dtype) if args.cs_terms_residual else None,
        )
        return device_prefetch(
            ((*batch[:4], family_ids[batch[4]] if phase_modes is None else
              np.column_stack((family_ids[batch[4]], phase_modes[batch[4]])),
              *((cpu_terms(batch[0], batch[1], batch[3]),) if cpu_terms is not None else ()))
             for batch in batches),
            sharding=data_sharding,
        )

    seed_seq = np.random.SeedSequence(args.seed)
    epoch_seeds = seed_seq.spawn(args.epochs)
    # Hadamard folds this fixed key with the GPU index and restored optimizer step.
    train_rng = replicate_pytree_from_host(jax.random.PRNGKey(args.seed + 1), replicated)
    validation_rng = replicate_pytree_from_host(jax.random.PRNGKey(args.seed + 2), replicated)

    def evaluate_validation(current_params: FlatParams) -> dict[str, float]:
        # Full validation, fixed probes, and the same buffered metric reads as training.
        val_batch_losses: list[tuple[float, ...]] = []
        val_batch_sizes: list[int] = []
        pending_val_losses: list[tuple[jax.Array, ...]] = []
        for batch_index, (eta_b, xi_b, gxi_b, depth_b, family_b, *terms_b) in enumerate(
            prefetch_batches(val_indices, None)
        ):
            val_batch_sizes.append(int(eta_b.shape[0]))
            compiled_eval_step = eval_steps[
                P("batch") if eta_b.shape[0] % n_devices == 0 else P(),
                args.hadamard_weight > 0.0 and batch_index % args.val_hadamard_interval == 0,
            ]
            batch_val_losses = compiled_eval_step(
                current_params, validation_rng, batch_index,
                eta_b, xi_b, gxi_b, depth_b, family_b, *terms_b,
            )
            pending_val_losses.append(batch_val_losses)
            if len(pending_val_losses) == args.metric_interval:
                val_batch_losses.extend(tuple(map(float, losses)) for losses in jax.device_get(pending_val_losses))
                pending_val_losses.clear()
        if pending_val_losses:
            val_batch_losses.extend(tuple(map(float, losses)) for losses in jax.device_get(pending_val_losses))

        losses = np.asarray(val_batch_losses)
        means = np.average(losses[:, :4], axis=0, weights=val_batch_sizes)
        tangent_counts = losses[:, 4] * val_batch_sizes
        means[2] = np.dot(losses[:, 2], tangent_counts) / max(tangent_counts.sum(), 1.0)
        hadamard_selected = np.arange(len(losses)) % args.val_hadamard_interval == 0
        means[3] = np.average(losses[hadamard_selected, 3], weights=np.asarray(val_batch_sizes)[hadamard_selected])
        data, mode, tangent, hadamard = map(float, means)
        mechanism = float(np.average(losses[:, 6], weights=val_batch_sizes))
        score = data + args.mode_balanced_weight * mode + args.translation_tangent_weight * tangent + args.phase_mechanism_weight * mechanism
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation selection score")
        metrics = {
            "val_loss": score,
            "val_l2_loss": float(np.average(losses[:, 5], weights=val_batch_sizes)),
            "val_data_loss": data,
            "val_mode_balanced_loss": mode,
            "val_translation_tangent_loss": tangent,
            "val_hadamard_loss": hadamard,
            "val_phase_mechanism_loss": mechanism,
        }
        family_totals = np.sum(losses[:, 7:] * np.asarray(val_batch_sizes)[:, None], axis=0).reshape(-1, 5)
        for family_id, (count, relative, residual, wins, data_sum) in zip(PhysicalFamilyId, family_totals):
            prefix = f"val_{family_id.name.lower()}"
            metrics[f"{prefix}_l2_loss"] = float(relative / count)
            metrics[f"{prefix}_baseline_ratio_floored"] = float(residual / count)
            metrics[f"{prefix}_baseline_win_fraction"] = float(wins / count)
            metrics[f"{prefix}_data_loss_share"] = float(data_sum / family_totals[:, 4].sum())
        return metrics

    # A raw-scored checkpoint must be re-scored before EMA-based early stopping.
    # Subsequent resumes retain their EMA score and patience history unchanged.
    if (history and args.weight_ema_decay > 0.0
            and int(training_state.step) > training_state.weight_ema_start_step
            and not history[-1].get("val_weight_ema", 0.0)):
        print(f"Revalidating epoch {start_epoch - 1} weight EMA before continuation", flush=True)
        history[-1].update(evaluate_validation(training_state.weight_ema_params))
        history[-1]["val_weight_ema"] = 1.0
        best_val_loss = history[-1]["val_loss"]
        best_epoch = start_epoch - 1
        # Only scores changed; do not overwrite the restored checkpoint payload.
        metadata_tmp = latest_ckpt_dir / "metadata.json.tmp"
        metadata_tmp.write_text(json.dumps({
            "epoch": start_epoch - 1, "train_loss": train_loss,
            "val_loss": best_val_loss, "history": history,
            "best_val_loss": best_val_loss, "best_epoch": best_epoch, "stats": stats,
        }, indent=2), encoding="utf-8")
        metadata_tmp.replace(metadata_path)
        print(f"EMA validation baseline: epoch {best_epoch}, loss {best_val_loss:.8g}", flush=True)

    # Retain the full training state and EMA shadow of the selected best epoch.
    if history and history[-1]["epoch"] == best_epoch and np.isfinite(best_val_loss):
        save_checkpoint(
            run_dir / "best_val_ckpt", state=training_state, history=history,
            best_val_loss=best_val_loss, best_epoch=best_epoch, stats=stats,
            async_manager=checkpoint_async_manager,
        )
    with open(run_dir / "train_log.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in history)

    # Train on every sample once per epoch, in a newly shuffled order.
    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Epochs", leave=False)
    for epoch in epoch_bar:
        if args.stop_after_epoch and epoch > args.stop_after_epoch:
            break
        if 0 < args.early_stopping_patience <= epoch - 1 - best_epoch:
            print(f"Early stopping after epoch {epoch - 1}: no validation improvement for "
                  f"{args.early_stopping_patience} epochs; best epoch {best_epoch}, "
                  f"validation loss {best_val_loss:.8g}", flush=True)
            break
        epoch_start_step, epoch_start_optimizer_count = training_counter_values(
            training_state.replace(opt_state=training_state.opt_state[1]),
            context=f"epoch {epoch} start",
        )
        epoch_rng = np.random.default_rng(epoch_seeds[epoch - 1])
        batch_losses: list[float] = []
        train_batch_sizes: list[int] = []
        loss_sums: dict[str, float] = {}
        tangent_loss_sum = tangent_sample_count = 0.0
        hadamard_loss_sum = 0.0
        hadamard_batch_evaluation_count = 0.0
        pending_train_metrics: list[tuple[jax.Array, dict[str, jax.Array], int]] = []
        train_bar = tqdm(
            prefetch_batches(train_indices, epoch_rng),
            total=train_steps_per_epoch,
            desc=f"Train {epoch:03d}",
            leave=False,
        )
        for batch_index, (eta_b, xi_b, gxi_b, depth_b, family_b, *terms_b) in enumerate(train_bar, 1):
            batch_size = int(eta_b.shape[0])
            compiled_train_step = train_steps[P("batch") if batch_size % n_devices == 0 else P()]
            # Apply one optimizer update; the returned metrics are only for logging.
            training_state, batch_loss, batch_metrics = compiled_train_step(
                training_state, train_rng, eta_b, xi_b, gxi_b, depth_b, family_b, *terms_b,
            )
            pending_train_metrics.append((batch_loss, batch_metrics, batch_size))
            if len(pending_train_metrics) < args.metric_interval and batch_index < train_steps_per_epoch:
                continue
            for batch_loss, batch_metrics, buffered_size in jax.device_get(pending_train_metrics):
                batch_loss_value = float(batch_loss)
                if not np.isfinite(batch_loss_value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
                batch_losses.append(batch_loss_value)
                train_batch_sizes.append(buffered_size)
                # Skipped Hadamard batches add zero loss and zero to the evaluation count.
                hadamard_loss_sum += float(batch_metrics.pop("hadamard_loss", 0.0))
                hadamard_batch_evaluation_count += float(batch_metrics.pop("hadamard_active", 0.0))
                # Translation loss is a mean over selected nonflat rows.
                selected_count = float(batch_metrics.pop("translation_tangent_fraction", 0.0)) * buffered_size
                tangent_sample_count += selected_count
                tangent_loss_sum += float(batch_metrics.pop("translation_tangent_loss", 0.0)) * selected_count
                # Other losses are batch means; convert them to sums over samples.
                for name, batch_mean in batch_metrics.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(batch_mean) * buffered_size
            pending_train_metrics.clear()
            train_bar.set_postfix(loss=batch_losses[-1], refresh=False)

        # Require one optimizer update per batch and matching counters across GPUs.
        epoch_end_step, epoch_end_optimizer_count = training_counter_values(
            training_state.replace(opt_state=training_state.opt_state[1]),
            context=f"epoch {epoch} end",
        )
        state_updates = epoch_end_step - epoch_start_step
        optimizer_updates = epoch_end_optimizer_count - epoch_start_optimizer_count
        expected_updates = len(batch_losses)
        if state_updates != expected_updates or optimizer_updates != expected_updates:
            raise RuntimeError(
                f"epoch {epoch}: expected {expected_updates} updates; "
                f"TrainState={state_updates}, optimizer={optimizer_updates}"
            )
        train_loss = float(np.average(batch_losses, weights=train_batch_sizes))

        ema_validation = args.weight_ema_decay > 0.0 and epoch_end_step > training_state.weight_ema_start_step
        validation = evaluate_validation(training_state.weight_ema_params if ema_validation else training_state.params)
        if ema_validation and history and not history[-1].get("val_weight_ema", 0.0):
            best_val_loss = float("inf")
        val_loss = validation["val_loss"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation,
            "val_weight_ema": float(ema_validation),
        }
        # Ordinary losses: total / samples. Hadamard: total / evaluated batches.
        for name, total in loss_sums.items():
            epoch_record[name] = total / sum(train_batch_sizes)
        if loss_sums.get("rollout_active", 0.0) > 0:
            for name in ("rollout_data_loss", "rollout_mode_balanced_loss",
                         "rollout_translation_tangent_loss", "rollout_phase_mechanism_loss",
                         "rollout_valid_fraction", "rollout_weight"):
                epoch_record[name] = loss_sums[name] / loss_sums["rollout_active"]
        if args.translation_tangent_weight > 0.0:
            epoch_record["translation_tangent_loss"] = tangent_loss_sum / max(tangent_sample_count, 1.0)
        if hadamard_batch_evaluation_count:
            epoch_record["hadamard_loss"] = hadamard_loss_sum / hadamard_batch_evaluation_count
        history.append(epoch_record)
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")

        # Per-epoch checkpoint for preemption-safe resume. Overwrites prior latest_ckpt.
        save_checkpoint(
            run_dir / "latest_ckpt",
            state=training_state,
            history=history,
            best_val_loss=best_val_loss if val_loss >= best_val_loss else val_loss,
            best_epoch=best_epoch if val_loss >= best_val_loss else epoch,
            stats=stats,
            async_manager=checkpoint_async_manager,
        )
        if args.weight_ema_decay > 0.0 and epoch_end_step > training_state.weight_ema_start_step:
            save_checkpoint(
                run_dir / "weight_ema_ckpt", state=training_state, history=history,
                best_val_loss=best_val_loss, best_epoch=best_epoch, stats=stats,
                async_manager=checkpoint_async_manager, weight_ema=True,
            )
        if args.weight_ema_secondary_decay > 0.0 and epoch_end_step > training_state.weight_ema_start_step:
            save_checkpoint(
                run_dir / "weight_ema_secondary_ckpt",
                state=training_state.replace(weight_ema_params=training_state.weight_ema_secondary_params,
                                             weight_ema_decay=args.weight_ema_secondary_decay),
                history=history, best_val_loss=best_val_loss, best_epoch=best_epoch, stats=stats,
                async_manager=checkpoint_async_manager, weight_ema=True,
            )

        # Save a separate copy whenever validation improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best_val_ckpt",
                state=training_state,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                stats=stats,
                async_manager=checkpoint_async_manager,
            )

        if epoch in args.rollout_epochs:
            print(f"Epoch {epoch}: starting 128 weight-EMA milestone rollouts", flush=True)
            rollout_loaded = LoadedRun(model, training_state.weight_ema_params, config_payload, stats, args.norm, epoch)
            rollout_predict = build_predict_gxi_batched(rollout_loaded)
            for group, families in enumerate((("stokes", "tanaka"), ("jonswap_tma", "benjamin_feir"))):
                rollout_dir = run_dir / f"eval_epoch{epoch}_weight_ema_test_stratified_n32_gpu{group}"
                rollout_dir.mkdir(parents=True, exist_ok=True)
                rollout_summaries = {
                    family: run_family(
                        family, FAMILY_CONFIGS[family], rollout_loaded, rollout_predict,
                        rollout_dir, dataset_path, 32,
                        Path(args.rollout_truth_run) / f"eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{group}",
                        {"run_dir": str(run_dir), "selection": "weight_ema",
                         "path": str(run_dir / "weight_ema_ckpt"), "epoch": epoch},
                        32, ic_selection="stratified",
                    ) for family in families
                }
                for filename, payload in (
                    ("all_summaries.json", rollout_summaries),
                    ("macro_summary.json", compute_macro_summary(rollout_summaries)),
                ):
                    _write_json(rollout_dir / filename, payload)
            del rollout_loaded, rollout_predict
            print(f"Epoch {epoch}: milestone rollouts complete; continuing training", flush=True)

    summary_payload = {
        "dataset": args.dataset,
        "device": backend,
        "run_dir": str(run_dir),
        "hparams": config_payload,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"],
        "runtime_seconds": perf_counter() - start_time,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
