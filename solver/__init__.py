from .reference_solutions.solitary_wave import load_soliton_dataset, load_soliton_file, list_soliton_files
from .reference_solutions.stokes_wave import stokes_eta_xi, stokes_truth_trajectory
from .evals.compare_rollout_jax import compare_rollout_to_truth, relative_l2
from .solvers.dno_series_jax import batched_dno_series_eval, build_grid, dno_series_eval, make_linear_dno_symbol, multiply
from .solvers.time_integrator import (
    RolloutSettings,
    SolverParams,
    State,
    batched_rollout,
    gauss_legendre_2_if_step,
    implicit_midpoint_if_step,
    make_normalized_rollout_settings,
    make_solver_params,
    rk4_if_step,
    rollout,
    take_step,
)

__all__ = [
    "SolverParams",
    "State",
    "RolloutSettings",
    "batched_dno_series_eval",
    "batched_rollout",
    "build_grid",
    "compare_rollout_to_truth",
    "dno_series_eval",
    "gauss_legendre_2_if_step",
    "implicit_midpoint_if_step",
    "list_soliton_files",
    "load_soliton_dataset",
    "load_soliton_file",
    "make_normalized_rollout_settings",
    "make_linear_dno_symbol",
    "make_solver_params",
    "multiply",
    "relative_l2",
    "rk4_if_step",
    "rollout",
    "stokes_eta_xi",
    "stokes_truth_trajectory",
    "take_step",
]
