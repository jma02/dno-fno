"""Export saved C27 rollout surfaces and original errors for a static gallery.

The gzip payload is little-endian Float32: truth and predicted initial eta
surfaces, then truth and predicted Fourier coefficients for subsequent frames.
Only display values are downcast; error curves come from full-grid FP64 archives.
This reads saved arrays only and never loads a model or initializes JAX.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = (
    ("tanaka", "Tanaka", 0),
    ("stokes", "Stokes", 0),
    ("benjamin_feir", "Benjamin–Feir", 1),
    ("jonswap_tma", "JONSWAP–TMA", 1),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text())
    index: dict[str, Any] = {
        "model": {
            "name": "C27 hard-P128",
            "epoch": 40,
            "branches": config["latent"],
            "parameters": config["param_count"],
        },
        "nx": 1024,
        "length": config["domain_length"],
        "binary": {
            "dtype": "float32-le",
            "layout": "truth_eta0[N],pred_eta0[N],truth_coeffs[T-1,K,2],pred_coeffs[T-1,K,2]",
            "compression": "gzip",
            "coefficientOrder": "real,imaginary",
            "fftNormalization": "forward",
            "nModes": 129,
        },
        "metricNote": "Original full-grid FP64 archive errors; display surfaces "
        "are reconstructed on all 1024 spatial points at all 251 saved frames "
        "from Float32 Fourier coefficients. The unfiltered initial frame is "
        "stored in full. Subsequent frames have modes 0–128; omitted spectral "
        "tails are measured and must be below 1e-12 relative L2.",
        "spectralTailMaxRelativeL2": 0.0,
        "displayMaxRelativeL2": 0.0,
        "families": [],
        "cases": [],
    }
    for family_id, family_label, gpu in FAMILIES:
        source_dir = (
            args.run_dir
            / f"eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{gpu}"
        )
        summary = json.loads((source_dir / f"{family_id}_summary.json").read_text())
        with np.load(source_dir / f"{family_id}_trajs.npz") as archive:
            times = archive["times"]
            ids = archive["simulation_ids"]
            depths = archive["depths"]
            truth = archive["truth_eta"]
            prediction = archive["pred_eta"]
            errors = archive["rel_l2_eta"]
            truth_valid = archive["truth_valid"]
            model_finite = ~archive["model_nonfinite_any"]
            # Check external archive integrity before assigning case labels.
            if truth.shape != prediction.shape or truth.shape != (
                len(times),
                len(ids),
                summary["nx"],
            ):
                raise ValueError(f"Unexpected surface shapes in {source_dir}")
            if ids.tolist() != summary["simulation_ids"]:
                raise ValueError(f"Simulation IDs disagree in {source_dir}")
            if not np.array_equal(depths, summary["depths"]):
                raise ValueError(f"Depths disagree in {source_dir}")
            truth_finite = np.isfinite(truth).all(axis=(0, 2))
            prediction_finite = np.isfinite(prediction).all(axis=(0, 2))
            for field in ("xi", "gxi"):
                truth_finite &= np.isfinite(archive[f"truth_{field}"]).all(
                    axis=(0, 2)
                )
                prediction_finite &= np.isfinite(archive[f"pred_{field}"]).all(
                    axis=(0, 2)
                )
            model_finite &= prediction_finite
            index["families"].append(
                {
                    "id": family_id,
                    "label": family_label,
                    "times": times.tolist(),
                }
            )
            groups = summary["evaluation_source"].get(
                "selected_parameter_group_ids", [None] * len(ids)
            )
            for case_index, simulation_id in enumerate(ids):
                case_id = f"current-fp32-{family_id}-{simulation_id}"
                filename = f"{case_id}.bin.gz"
                surfaces = (truth[:, case_index], prediction[:, case_index])
                tail_error = 0.0
                display_error = 0.0
                coefficients = []
                for surface in surfaces:
                    spectrum = np.fft.rfft(surface[1:], axis=-1, norm="forward")
                    weights = np.full(spectrum.shape[-1], 2.0)
                    weights[[0, -1]] = 1.0
                    tail_energy = np.sum(
                        np.abs(spectrum[:, 129:]) ** 2 * weights[129:], axis=-1
                    )
                    surface_energy = np.mean(surface[1:] ** 2, axis=-1)
                    tail_error = max(
                        tail_error,
                        float(np.sqrt(tail_energy / surface_energy).max()),
                    )
                    packed = spectrum[:, :129].astype("<c8")
                    reconstructed = np.fft.irfft(
                        packed.astype(np.complex128),
                        n=summary["nx"],
                        axis=-1,
                        norm="forward",
                    )
                    display_error = max(
                        display_error,
                        float(
                            np.sqrt(
                                np.mean((reconstructed - surface[1:]) ** 2, axis=-1)
                                / surface_energy
                            ).max()
                        ),
                        float(
                            np.linalg.norm(surface[0].astype("<f4") - surface[0])
                            / np.linalg.norm(surface[0])
                        ),
                    )
                    coefficients.append(packed)
                if tail_error > 1e-12:
                    raise ValueError(
                        f"Non-bandlimited eta in {case_id}: {tail_error}"
                    )
                index["spectralTailMaxRelativeL2"] = max(
                    index["spectralTailMaxRelativeL2"], tail_error
                )
                index["displayMaxRelativeL2"] = max(
                    index["displayMaxRelativeL2"], display_error
                )
                with gzip.open(
                    args.output_dir / filename, "wb", compresslevel=6
                ) as stream:
                    for surface in surfaces:
                        stream.write(surface[0].astype("<f4").tobytes(order="C"))
                    for packed in coefficients:
                        stream.write(packed.tobytes(order="C"))
                curve = [
                    float(value) if np.isfinite(value) else None
                    for value in errors[:, case_index]
                ]
                extrema = np.concatenate(
                    [surface[np.isfinite(surface)] for surface in surfaces]
                )
                index["cases"].append(
                    {
                        "id": case_id,
                        "family": family_id,
                        "simulationId": int(simulation_id),
                        "caseIndex": case_index,
                        "depth": float(depths[case_index]),
                        "file": f"data/{filename}",
                        "finite": bool(model_finite[case_index]),
                        "truthFinite": bool(truth_finite[case_index]),
                        "truthValid": bool(truth_valid[case_index]),
                        "terminalRelL2": curve[-1],
                        "etaRelL2": curve,
                        "range": [float(extrema.min()), float(extrema.max())]
                        if extrema.size
                        else [-1.0, 1.0],
                        "parameterGroup": groups[case_index],
                        "tmax": float(times[-1]),
                        "nFrames": len(times),
                        "nModes": 129,
                        "nx": summary["nx"],
                        "length": summary["length"],
                    }
                )
            print(
                f"{family_id}: {len(ids)} cases, "
                f"truth finite {int(truth_finite.sum())}, "
                f"prediction finite {int(model_finite.sum())}",
                flush=True,
            )
    (args.output_dir / "index.json").write_text(
        json.dumps(index, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(f"Exported {len(index['cases'])} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
