from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from models.fno import FNO1d

fno_train = importlib.import_module("train.1d_dno_fno")
load_processed_dataset = fno_train.load_processed_dataset
build_feature_target_tensors = fno_train.build_feature_target_tensors
normalize_to_range = fno_train.normalize_to_range
denormalize_from_range = fno_train.denormalize_from_range


plt.rcParams["font.family"] = "DejaVu Serif"
TITLE_FONT = {"family": "DejaVu Serif", "weight": "bold", "size": 10}

TEST_FRACTION = 0.1
RUN_RE = re.compile(r"Nx(?P<Nx>\d+)_M(?P<M>\d+)_ichoi(?P<ichoi>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a0=max most-oscillatory test diagnostics for grid-search FNO runs"
    )
    parser.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--grid-data-dir", default=str(REPO_ROOT / "grid_data"))
    parser.add_argument("--name-contains", default="gridfno_")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--save-name", default="a0max_most_oscillatory_test.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loss", default="")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--filter-scope", choices=["test", "all"], default="test")
    return parser.parse_args()


def iter_runs(outputs_dir: Path, name_contains: str) -> list[Path]:
    runs: list[Path] = []
    for run_dir in sorted(outputs_dir.glob("fno_*")):
        if not run_dir.is_dir():
            continue
        if name_contains and name_contains not in run_dir.name:
            continue
        runs.append(run_dir)
    return runs


def parse_run_grid_params(run_dir: Path) -> dict[str, int] | None:
    match = RUN_RE.search(run_dir.name)
    if not match:
        return None
    return {k: int(v) for k, v in match.groupdict().items()}


def dataset_paths(data_dir: Path, grid_data_dir: Path, params: dict[str, int]) -> tuple[Path, Path]:
    stem = f"stokes_dno_training_Nx{params['Nx']}_M{params['M']}_ichoi{params['ichoi']}"
    return data_dir / f"{stem}.npz", grid_data_dir / f"{stem}.mat"


def resolve_mat_path(grid_data_dir: Path, params: dict[str, int]) -> Path:
    stem = f"stokes_dno_training_Nx{params['Nx']}_M{params['M']}_ichoi{params['ichoi']}.mat"
    candidates = [
        grid_data_dir / stem,
        grid_data_dir / f"M{params['M']}" / stem,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(grid_data_dir.glob(f"**/{stem}"))
    if matches:
        return matches[0]
    return candidates[0]


def load_run_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    cfg_path = run_dir / "config.json"
    state_path = run_dir / "best_model.pt"
    if not cfg_path.exists() or not state_path.exists():
        raise FileNotFoundError("Missing config.json or best_model.pt")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config, state_path


def load_mat_params(mat_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(mat_path, "r") as f:
        def read_vec(name: str) -> np.ndarray:
            return np.array(f[name]).T.reshape(-1)

        return {
            "a0": read_vec("params/a0"),
            "n0": read_vec("params/n0"),
            "k0": read_vec("params/k0"),
            "ichoi": read_vec("params/ichoi"),
        }


def samplewise_eta_tv(eta_samples: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(eta_samples, axis=1)), axis=1)


def predict_in_batches(
    model: torch.nn.Module,
    features_norm: torch.Tensor,
    target_stats: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    preds: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features_norm.size(0), batch_size):
            batch = features_norm[start : start + batch_size].to(device)
            pred_norm = model(batch)
            pred = denormalize_from_range(pred_norm, target_stats)
            preds.append(pred.cpu())
    return torch.cat(preds, dim=0)


def rel_errors(preds: torch.Tensor, targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    flat_pred = preds.view(preds.size(0), -1)
    flat_true = targets.view(targets.size(0), -1)
    l2 = torch.norm(flat_pred - flat_true, dim=1) / (torch.norm(flat_true, dim=1) + 1e-12)
    l1 = torch.sum(torch.abs(flat_pred - flat_true), dim=1) / (
        torch.sum(torch.abs(flat_true), dim=1) + 1e-12
    )
    return l2.numpy(), l1.numpy()


def plot_selected_samples(
    run_dir: Path,
    save_path: Path,
    x_plot: np.ndarray,
    eta_test: np.ndarray,
    xi_test: np.ndarray,
    gxi_test: np.ndarray,
    preds_test: np.ndarray,
    selected_local_idx: np.ndarray,
    global_sample_idx: np.ndarray,
    traj_idx_for_selected: np.ndarray,
    params_by_traj: dict[str, np.ndarray],
    scores_test: np.ndarray,
    err_l2: np.ndarray,
    err_l1: np.ndarray,
    grid_params: dict[str, int],
    a0_max: float,
    scope_label: str,
) -> None:
    rows = len(selected_local_idx)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.2 * rows), sharex=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, local_idx in enumerate(selected_local_idx):
        traj_idx = int(traj_idx_for_selected[row])
        global_idx = int(global_sample_idx[row])
        a0_val = float(params_by_traj["a0"][traj_idx])
        n0_val = float(params_by_traj["n0"][traj_idx])
        k0_val = float(params_by_traj["k0"][traj_idx])

        axes[row, 0].plot(x_plot, eta_test[local_idx], color="tab:blue", linewidth=1.0)
        axes[row, 0].set_title("η(x)", fontdict=TITLE_FONT)
        axes[row, 0].set_ylabel(
            "sample {sample}\ntraj {traj}\nscore={score:.2f}\na0={a0:.3f}\n"
            "n0={n0:.3f}\nk0={k0:.3f}".format(
                sample=global_idx,
                traj=traj_idx,
                score=float(scores_test[local_idx]),
                a0=a0_val,
                n0=n0_val,
                k0=k0_val,
            ),
            fontsize=9,
        )
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(x_plot, xi_test[local_idx], color="tab:green", linewidth=1.0)
        axes[row, 1].set_title("ξ(x)", fontdict=TITLE_FONT)
        axes[row, 1].grid(True, alpha=0.3)

        axes[row, 2].plot(x_plot, gxi_test[local_idx], color="black", linewidth=1.0, label="Ground Truth")
        axes[row, 2].plot(x_plot, preds_test[local_idx], color="tab:red", linewidth=1.0, label="Prediction")
        axes[row, 2].set_title(
            f"G(η,ξ)  L2={float(err_l2[local_idx]):.2e}  L1={float(err_l1[local_idx]):.2e}",
            fontdict=TITLE_FONT,
        )
        axes[row, 2].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 2].legend(fontsize=8)

    for col in range(3):
        axes[-1, col].set_xlabel("x")

    fig.suptitle(
        f"{run_dir.name} | Nx={grid_params['Nx']} | M={grid_params['M']} | ichoi={grid_params['ichoi']} | "
        f"a0=max={a0_max:.3f} | n={rows} | Most oscillatory solutions ({scope_label} scope)",
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def safe_slug(text: str) -> str:
    if not text:
        return "gridfno"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "gridfno"


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_dir",
        "Nx",
        "M",
        "ichoi",
        "status",
        "reason",
        "output_plot",
        "a0_max",
        "filtered_test_count",
        "filter_scope",
        "selected_count",
        "selected_global_samples",
        "selected_traj_indices",
        "selected_scores",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def process_run(
    run_dir: Path,
    data_dir: Path,
    grid_data_dir: Path,
    device: torch.device,
    batch_size: int,
    num_samples: int,
    save_name: str,
    overwrite: bool,
    loss_override: str,
    filter_scope: str,
) -> dict[str, Any]:
    base_result: dict[str, Any] = {
        "run_dir": run_dir.name,
        "Nx": "",
        "M": "",
        "ichoi": "",
        "status": "",
        "reason": "",
        "output_plot": "",
        "a0_max": "",
        "filtered_test_count": "",
        "filter_scope": "",
        "selected_count": "",
        "selected_global_samples": "",
        "selected_traj_indices": "",
        "selected_scores": "",
    }

    plot_path = run_dir / save_name
    if plot_path.exists() and not overwrite:
        base_result["status"] = "skip_exists"
        base_result["reason"] = "plot_exists"
        base_result["output_plot"] = str(plot_path)
        return base_result

    params = parse_run_grid_params(run_dir)
    if params is None:
        base_result["status"] = "skip_parse_run_name"
        base_result["reason"] = "unparseable_run_name"
        return base_result
    base_result.update({k: params[k] for k in ("Nx", "M", "ichoi")})

    try:
        config, state_path = load_run_config(run_dir)
    except FileNotFoundError:
        base_result["status"] = "skip_missing_checkpoint"
        base_result["reason"] = "missing config.json or best_model.pt"
        return base_result

    npz_path, _ = dataset_paths(data_dir, grid_data_dir, params)
    mat_path = resolve_mat_path(grid_data_dir, params)
    if not npz_path.exists():
        base_result["status"] = "skip_missing_npz"
        base_result["reason"] = str(npz_path)
        return base_result
    if not mat_path.exists():
        base_result["status"] = "skip_missing_mat"
        base_result["reason"] = str(mat_path)
        return base_result

    try:
        dataset_np = load_processed_dataset(npz_path)
        features_raw, targets_raw = build_feature_target_tensors(dataset_np)
        features_norm, _ = normalize_to_range(features_raw)
        _, target_stats = normalize_to_range(targets_raw)

        n_samples = int(features_norm.size(0))
        test_size = int(n_samples * TEST_FRACTION)
        test_idx = np.arange(n_samples, dtype=np.int64)[:test_size]
        if test_size <= 0:
            raise RuntimeError("Empty test split from TEST_FRACTION")
        if filter_scope == "all":
            pool_idx = np.arange(n_samples, dtype=np.int64)
            scope_label = "all"
        else:
            pool_idx = test_idx
            scope_label = "test"
        base_result["filter_scope"] = scope_label

        params_by_traj = load_mat_params(mat_path)
        a0_by_traj = params_by_traj["a0"]
        n_time = int(np.asarray(dataset_np["t"]).shape[0])
        if n_time <= 0:
            raise RuntimeError("Processed dataset has empty t")
        n_traj = int(a0_by_traj.shape[0])
        if n_samples != n_traj * n_time:
            base_result["status"] = "skip_shape_mismatch"
            base_result["reason"] = f"n_samples={n_samples}, n_traj={n_traj}, n_time={n_time}"
            return base_result

        traj_idx_all = np.arange(n_samples, dtype=np.int64) // n_time
        a0_all = a0_by_traj[traj_idx_all]
        a0_max = float(np.max(a0_by_traj))
        a0_pool = a0_all[pool_idx]

        eta_np = np.asarray(dataset_np["eta"])
        xi_np = np.asarray(dataset_np["xi"])
        gxi_np = np.asarray(dataset_np["Gxi"])
        eta_pool = eta_np[pool_idx]
        xi_pool = xi_np[pool_idx]
        gxi_pool = gxi_np[pool_idx]

        scores_pool = samplewise_eta_tv(eta_pool)
        filtered_local = np.where(np.isclose(a0_pool, a0_max))[0]
        base_result["a0_max"] = f"{a0_max:.8g}"
        base_result["filtered_test_count"] = int(filtered_local.size)
        if filtered_local.size == 0:
            base_result["status"] = "skip_no_filtered_samples"
            base_result["reason"] = f"no a0=max samples in {scope_label} split"
            return base_result

        ranked = filtered_local[np.argsort(scores_pool[filtered_local])[::-1]]
        selected_local = ranked[: max(1, min(num_samples, ranked.size))]

        model = FNO1d(
            int(config["modes"]),
            int(config["width"]),
            n_blocks=int(config.get("n_blocks", 10)),
            mode_sampling=config.get("mode_sampling", "low"),
            high_mode_frac=float(config.get("high_mode_frac", 0.5)),
        ).to(device)
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)

        pool_features_norm = features_norm[pool_idx]
        preds_pool_t = predict_in_batches(model, pool_features_norm, target_stats, batch_size, device)
        targets_pool_t = targets_raw[pool_idx].cpu()
        err_l2, err_l1 = rel_errors(preds_pool_t, targets_pool_t)

        preds_test = preds_pool_t.squeeze(-1).numpy()
        global_sample_idx = pool_idx[selected_local]
        selected_traj_idx = traj_idx_all[global_sample_idx]

        x_arr = np.asarray(dataset_np.get("x", np.arange(gxi_pool.shape[1])))
        if x_arr.ndim != 1 or x_arr.shape[0] != gxi_pool.shape[1]:
            x_arr = np.arange(gxi_pool.shape[1])

        plot_selected_samples(
            run_dir=run_dir,
            save_path=plot_path,
            x_plot=x_arr,
            eta_test=eta_pool,
            xi_test=xi_pool,
            gxi_test=gxi_pool,
            preds_test=preds_test,
            selected_local_idx=selected_local,
            global_sample_idx=global_sample_idx,
            traj_idx_for_selected=selected_traj_idx,
            params_by_traj=params_by_traj,
            scores_test=scores_pool,
            err_l2=err_l2,
            err_l1=err_l1,
            grid_params=params,
            a0_max=a0_max,
            scope_label=scope_label,
        )

        base_result["status"] = "ok"
        base_result["reason"] = ""
        base_result["output_plot"] = str(plot_path)
        base_result["selected_count"] = int(selected_local.size)
        base_result["selected_global_samples"] = ";".join(str(int(i)) for i in global_sample_idx)
        base_result["selected_traj_indices"] = ";".join(str(int(i)) for i in selected_traj_idx)
        base_result["selected_scores"] = ";".join(f"{float(scores_pool[i]):.6g}" for i in selected_local)
        if loss_override:
            _ = loss_override  # accepted for interface compatibility; plotting does not need scalar loss eval
        return base_result
    except Exception as exc:
        base_result["status"] = "error_eval"
        base_result["reason"] = f"{type(exc).__name__}: {exc}"
        return base_result


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    grid_data_dir = Path(args.grid_data_dir).resolve()

    runs = iter_runs(outputs_dir, args.name_contains)
    if not runs:
        print("No matching grid-search FNO runs found.")
        return

    device = torch.device(args.device)
    summary_rows: list[dict[str, Any]] = []
    total = len(runs)

    for idx, run_dir in enumerate(runs, start=1):
        row = process_run(
            run_dir=run_dir,
            data_dir=data_dir,
            grid_data_dir=grid_data_dir,
            device=device,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            save_name=args.save_name,
            overwrite=args.overwrite,
            loss_override=args.loss,
            filter_scope=args.filter_scope,
        )
        summary_rows.append(row)

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == total):
            status = row.get("status", "")
            if status == "ok":
                print(
                    f"[{idx}/{total}] OK {run_dir.name} -> {args.save_name} "
                    f"(n={row.get('selected_count', 0)})",
                    flush=True,
                )
            else:
                print(
                    f"[{idx}/{total}] {status.upper()} {run_dir.name} "
                    f"reason={row.get('reason', '')}",
                    flush=True,
                )

    summary_name = f"{safe_slug(args.name_contains)}_a0max_oscillatory_plots.csv"
    summary_path = outputs_dir / summary_name
    write_summary_csv(summary_rows, summary_path)

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    skip_count = sum(1 for r in summary_rows if str(r.get("status", "")).startswith("skip"))
    err_count = sum(1 for r in summary_rows if r.get("status") == "error_eval")
    print(
        f"Done. total={len(summary_rows)} ok={ok_count} skipped={skip_count} errors={err_count} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
