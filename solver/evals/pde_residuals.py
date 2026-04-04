from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.solitary_loader_jax import list_soliton_files, load_soliton_file


def build_wavenumbers(nx: int, length: float) -> np.ndarray:
    dk = 2.0 * np.pi / length
    return dk * np.concatenate((np.arange(0, nx // 2 + 1), np.arange(1 - nx // 2, 0)))


def spectral_dx(field: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(1j * k * np.fft.fft(field, axis=-1), axis=-1))


def time_derivative(field: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.gradient(field, times, axis=0, edge_order=2)


def summarize_residuals(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    times: np.ndarray,
    length: float,
    gravity: float = 1.0,
) -> dict[str, float]:
    nx = eta.shape[-1]
    k = build_wavenumbers(nx, length)

    eta_t = time_derivative(eta, times)
    xi_t = time_derivative(xi, times)
    eta_x = spectral_dx(eta, k)
    xi_x = spectral_dx(xi, k)

    rhs_eta = gxi
    numerator = gxi + eta_x * xi_x
    rhs_xi = -gravity * eta - 0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)

    residual_eta = eta_t - rhs_eta
    residual_xi = xi_t - rhs_xi

    eta_abs_rms = float(np.sqrt(np.mean(residual_eta**2)))
    xi_abs_rms = float(np.sqrt(np.mean(residual_xi**2)))
    eta_rel_l2 = float(np.linalg.norm(residual_eta) / np.linalg.norm(rhs_eta))
    xi_rel_l2 = float(np.linalg.norm(residual_xi) / np.linalg.norm(rhs_xi))

    return {
        "eta_abs_rms": eta_abs_rms,
        "xi_abs_rms": xi_abs_rms,
        "eta_rel_l2": eta_rel_l2,
        "xi_rel_l2": xi_rel_l2,
    }


def load_rollout_payload(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path).expanduser().resolve())
    return {
        "x": np.asarray(data["x"]),
        "t": np.asarray(data["t"]),
        "eta": np.asarray(data["pred_eta"]),
        "xi": np.asarray(data["pred_xi"]),
        "gxi": np.asarray(data["pred_gxi"]),
    }


def analyze_rollout_paths(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = load_rollout_payload(path)
        length = float((payload["x"][1] - payload["x"][0]) * payload["x"].shape[0])
        metrics = summarize_residuals(
            payload["eta"],
            payload["xi"],
            payload["gxi"],
            payload["t"],
            length=length,
        )
        rows.append({"name": path.stem, **metrics})
    return rows


def analyze_soliton_data() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in list_soliton_files():
        data = load_soliton_file(path)
        metrics = summarize_residuals(
            np.asarray(data["eta"]),
            np.asarray(data["xi"]),
            np.asarray(data["gxi"]),
            np.asarray(data["t"]),
            length=float(data["length"]),
        )
        rows.append({"name": data["name"], **metrics})
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    eta_rel = np.array([row["eta_rel_l2"] for row in rows], dtype=np.float64)
    xi_rel = np.array([row["xi_rel_l2"] for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "eta_rel_l2_mean": float(np.mean(eta_rel)),
        "eta_rel_l2_median": float(np.median(eta_rel)),
        "eta_rel_l2_max": float(np.max(eta_rel)),
        "xi_rel_l2_mean": float(np.mean(xi_rel)),
        "xi_rel_l2_median": float(np.median(xi_rel)),
        "xi_rel_l2_max": float(np.max(xi_rel)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PDE residual summaries for saved rollout payloads and provided soliton data.")
    parser.add_argument("--output", default="outputs/pde_residuals_summary.json")
    parser.add_argument(
        "--fp32_rollout_glob",
        default="outputs/tanaka_batch_e2e_fp32_sep24/gifs/*.npz",
    )
    parser.add_argument(
        "--fp64_rollout_glob",
        default="outputs/tanaka_batch_e2e_fp64_sep24/gifs/*.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp32_paths = sorted(Path().glob(args.fp32_rollout_glob))
    fp64_paths = sorted(Path().glob(args.fp64_rollout_glob))

    fp32_rows = analyze_rollout_paths(fp32_paths)
    fp64_rows = analyze_rollout_paths(fp64_paths)
    soliton_rows = analyze_soliton_data()

    summary = {
        "fp32_rollouts": {
            "paths": [str(path) for path in fp32_paths],
            "aggregate": aggregate(fp32_rows),
            "rows": fp32_rows,
        },
        "fp64_rollouts": {
            "paths": [str(path) for path in fp64_paths],
            "aggregate": aggregate(fp64_rows),
            "rows": fp64_rows,
        },
        "soliton_data": {
            "aggregate": aggregate(soliton_rows),
            "rows": soliton_rows,
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
