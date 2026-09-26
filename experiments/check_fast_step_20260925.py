"""GPU correctness checks for capture warmup, validation and partial-batch updates."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-fast-step-check")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def check(tune_matrices: bool = False) -> dict:
    import copy
    import os
    import sys

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/repo/torch-attention")
    from spectral import SpectralDNO
    from train import CudaStep, Waves, build_optimizers, relative_l2

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    checkpoint = torch.load("/data/experiments/torch_spectral_features_h100_b256_20260925/"
                            "torch_spectral_features_h100_b256_20260925_bf16_spectral_d2.pt",
                            map_location="cuda", weights_only=True)
    batch = next(iter(DataLoader(Waves(Path("/subset"), "train"), batch_size=256)))
    report = {}
    for resumed in (False, True):
        torch.manual_seed(0)
        model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True,
                            feature_scales=checkpoint["model"]["feature_scales"]).cuda()
        if resumed:
            model.load_state_dict(checkpoint["model"])
        reference = copy.deepcopy(model)
        optimizers = build_optimizers(model, "muon-grouped" if tune_matrices else "muon", 1e-5)
        reference_optimizers = build_optimizers(reference, "muon", 1e-5)
        if resumed:
            for group in (optimizers, reference_optimizers):
                for optimizer, saved in zip(group, checkpoint["optimizers"], strict=True):
                    optimizer.load_state_dict(copy.deepcopy(saved))
        averages = [a.clone() for a in checkpoint["gradient_ema"]] if resumed else [torch.zeros_like(p) for p in model.parameters()]
        reference_averages = [a.clone() for a in averages]
        start = checkpoint["step"] if resumed else 0
        fast = CudaStep(model, optimizers, averages, batch, .8, start, autotune=tune_matrices)
        assert fast.counter.item() == start
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for a, b in zip(averages, reference_averages, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for optimizer, reference_optimizer in zip(optimizers, reference_optimizers, strict=True):
            for state, p in zip(optimizer.state.values(), reference_optimizer.param_groups[0]["params"], strict=True):
                original = reference_optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        expected = original[key].to(value.device) if resumed else torch.zeros_like(value)
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        losses = []
        # A full batch after the tail checks that captured gradient addresses survive.
        for step, size in enumerate((256, 256, 13, 256, 256), start=start + 1):
            current = tuple(value[:size] for value in batch)
            device_batch = tuple(value.cuda() for value in current)
            model.train()
            reference.train()
            fast_loss = fast(current).item()
            reference.zero_grad(set_to_none=True)
            loss = relative_l2(reference(*device_batch[:3]), device_batch[3])
            loss.backward()
            with torch.no_grad():
                for p, avg in zip(reference.parameters(), reference_averages, strict=True):
                    avg.mul_(.8).add_(p.grad, alpha=.2)
                    p.grad.copy_(avg / (1 - .8**step))
            for optimizer in reference_optimizers:
                optimizer.step()
            losses.append(abs(fast_loss - loss.item()))
            model.eval()
            reference.eval()
            with torch.no_grad():
                output = model(*device_batch[:3])
                expected = reference(*device_batch[:3])
                error = ((output - expected).norm() / expected.norm()).item()
                assert error < 2e-5, error
        assert fast.counter.item() == start + 5
        assert all(state["step"].item() == start + 5 for state in optimizers[-1].state.values())
        assert max(losses) < 3e-7, losses
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.isfinite(a).all() for a in averages)
        report["resumed" if resumed else "fresh"] = {"max_loss_difference": max(losses),
                "final_prediction_relative_difference": error, "final_step": int(fast.counter.item()),
                "warmup_state_unchanged": True, "full_tail_full_and_validation_passed": True}
        print(report, flush=True)
    volume.commit()
    return report


@app.local_entrypoint()
def main(run_name: str = "check_fast_step_20260925", tune_matrices: bool = False) -> None:
    result = check.remote(tune_matrices)
    (ROOT / "experiments" / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit("Run with modal run")
