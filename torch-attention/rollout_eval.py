# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "numpy>=2"]
# ///
"""Run a spectral PyTorch checkpoint from saved truth or physical initial fields."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from rollout import rollout
from spectral import SpectralDNO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="eval-suite truth cache/trajs NPZ, or eta/xi (batch,n), times, depths NPZ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--substeps", type=int, help="Default: saved truth protocol, otherwise 8")
    parser.add_argument("--filter-fraction", type=float, help="Default: saved truth protocol, otherwise 0.25")
    parser.add_argument("--frames", type=int, help="Evaluate only this prefix of saved times")
    args = parser.parse_args()
    if args.out.resolve() in (args.input.resolve(), args.checkpoint.resolve()) or args.out.exists():
        parser.error("--out must be a new file, separate from input/checkpoint")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "spectral":
        parser.error("This runner supports the spectral PyTorch checkpoint")
    config = checkpoint["args"]
    model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"],
                        width=config["width"], depth=config["depth"], heads=config.get("heads", 4),
                        branches=config.get("branches", 32), bf16=config["bf16"],
                        max_mode=config.get("max_mode"),
                        feature_scales=checkpoint["model"]["feature_scales"])
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    with np.load(args.input, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    protocol = json.loads(str(data["truth_protocol_json"])) if "truth_protocol_json" in data else {}
    if protocol and (protocol.get("method") != "gl2_if" or not protocol.get("zero_mean_xi")):
        parser.error("Saved truth protocol must use gl2_if with zero_mean_xi")
    if "length" in data and not np.isclose(float(data["length"]), model.length):
        parser.error("Input length differs from checkpoint length")
    if "length" in protocol and not np.isclose(protocol["length"], model.length):
        parser.error("Truth protocol length differs from checkpoint length")
    times = np.asarray(data["times"], dtype=np.float64)
    if args.frames is not None:
        if not 1 <= args.frames <= len(times):
            parser.error("--frames must be between 1 and the input time count")
        times = times[:args.frames]
    has_truth = "truth_eta" in data
    if has_truth:
        truth = {field: np.asarray(data[f"truth_{field}"], dtype=np.float64)[:len(times)]
                 for field in ("eta", "xi", "gxi")}
        eta, xi = truth["eta"][0], truth["xi"][0]
        if any(v.shape != (len(times), *eta.shape) for v in truth.values()):
            parser.error("Truth arrays must have matching (time,batch,n) shapes")
    else:
        eta, xi = data["eta"], data["xi"]
    if eta.ndim != 2 or eta.shape[-1] != model.n:
        parser.error("Initial fields must be (batch, checkpoint grid size)")
    depths = np.asarray(data["depths"], dtype=np.float64).reshape(-1)
    if len(depths) != len(eta):
        parser.error("One depth is required per initial condition")
    ids = np.asarray(data.get("simulation_ids", np.arange(len(eta))), dtype=np.int64)
    if ids.shape != (len(eta),):
        parser.error("simulation_ids must have one entry per initial condition")
    substeps = args.substeps if args.substeps is not None else int(protocol.get("substeps", 8))
    fraction = args.filter_fraction if args.filter_fraction is not None else float(protocol.get("filter_fraction", .25))
    iterations = int(protocol.get("implicit_iterations", 4))
    tensors = [torch.as_tensor(v, dtype=torch.float64, device=args.device) for v in (eta, xi, times, depths)]
    model_depths = tensors[3].float().reshape(-1, 1)

    def predict(a: Tensor, b: Tensor) -> Tensor:
        result = model(a.float(), b.float(), model_depths).double()
        return result - result.mean(-1, keepdim=True)

    started = perf_counter()
    prediction = rollout(*tensors[:3], tensors[3], model.length, predict,
                         substeps=substeps, iterations=iterations, filter_fraction=fraction)
    pred = {k: v.cpu().numpy() for k, v in prediction.items()}
    seconds = perf_counter() - started
    output = {"times": pred["times"], "depths": depths, "simulation_ids": ids,
              "length": np.asarray(model.length), **{f"pred_{k}": pred[k] for k in ("eta", "xi", "gxi")}}
    if has_truth:
        for field, values in truth.items():
            output[f"truth_{field}"] = values
            output[f"rel_l2_{field}"] = (np.linalg.norm(pred[field] - values, axis=-1)
                                          / (np.linalg.norm(values, axis=-1) + 1e-12))
    if "truth_protocol_json" in data:
        output["truth_protocol_json"] = data["truth_protocol_json"]
    nonfinite = ~np.isfinite(np.stack([pred[k] for k in ("eta", "xi", "gxi")])).all(axis=(0, 1, 3))
    output["model_nonfinite_any"] = nonfinite
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint.get("epoch"),
                "device": args.device, "integration_dtype": "float64", "model_bf16": model.bf16,
                "substeps": substeps, "implicit_iterations": iterations, "filter_fraction": fraction,
                "length": model.length, "seconds": seconds, "frames": len(times), "initial_conditions": len(eta),
                "nonfinite_trajectories": int(nonfinite.sum()), "input": str(args.input.resolve())}
    output["rollout_metadata_json"] = np.asarray(json.dumps(metadata))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as handle:
        np.savez_compressed(handle, **output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
