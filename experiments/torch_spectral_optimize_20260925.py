"""Profile eager H100 training and test identical training with CUDA graph replay."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dno-spectral-step-optimization")
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run(run_name: str, compile_model: bool) -> dict:
    import copy
    import gzip
    import os
    from statistics import median
    import sys
    from time import perf_counter

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler

    sys.path.insert(0, "/repo/torch-attention")
    from spectral import SpectralDNO
    from train import Waves, build_optimizers, relative_l2

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/torch-inductor-cache")
    assert "H100" in torch.cuda.get_device_name()
    checkpoint = torch.load("/data/experiments/torch_spectral_features_h100_b256_20260925/"
                            "torch_spectral_features_h100_b256_20260925_bf16_spectral_d2.pt",
                            map_location="cuda", weights_only=True)
    result = {"gpu": torch.cuda.get_device_name(), "batch_size": 256, "variants": {}}
    destination = Path("/data/experiments") / run_name
    destination.mkdir(exist_ok=False)

    class BulkWaves(Waves):
        def __getitem__(self, indices: list[int]) -> tuple[torch.Tensor, ...]:
            values = tuple(torch.as_tensor(np.array(a[self.rows[indices]], copy=True), dtype=torch.float32)
                           .reshape(len(indices), -1) for a in self.arrays)
            eta, xi, depth, target = values
            return eta, xi - xi.mean(-1, keepdim=True), depth, target

    reference = None
    variants = ("eager", "graph_bulk", "compiled_graph_bulk") if compile_model else ("eager", "graph", "graph_bulk")
    for variant in variants:
        model = SpectralDNO(n=checkpoint["n"], length=checkpoint["length"], depth=2, bf16=True).cuda()
        model.load_state_dict(checkpoint["model"])
        model.train()
        forward = torch.compile(model, fullgraph=True, options={"triton.cudagraphs": False}) if variant == "compiled_graph_bulk" else model
        optimizers = build_optimizers(model, "muon", 1e-5)
        for optimizer, state in zip(optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(copy.deepcopy(state))
        averages = [a.clone() for a in checkpoint["gradient_ema"]]
        parameters = list(model.parameters())
        step_number = checkpoint["step"]
        dataset = BulkWaves(Path("/subset"), "train") if variant.endswith("bulk") else Waves(Path("/subset"), "train")
        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(dataset, batch_size=None, generator=generator,
                            sampler=BatchSampler(RandomSampler(dataset, generator=generator), 256, False)) if variant.endswith("bulk") else DataLoader(
                                dataset, batch_size=256, shuffle=True, generator=generator)
        iterator = iter(loader)
        example = next(iter(DataLoader(Waves(Path("/subset"), "train"), batch_size=256)))
        static = tuple(x.cuda() for x in example)

        def eager_step(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            nonlocal step_number
            with record_function("phase/transfer"):
                eta, xi, depth, target = (x.cuda() for x in batch)
            model.zero_grad(set_to_none=True)
            with record_function("phase/forward"):
                loss = relative_l2(model(eta, xi, depth), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
            with record_function("phase/backward"):
                loss.backward()
            step_number += 1
            with record_function("phase/ema"), torch.no_grad():
                for p, average in zip(parameters, averages, strict=True):
                    average.mul_(.8).add_(p.grad, alpha=.2)
                    p.grad.copy_(average / (1 - .8**step_number))
            for optimizer in optimizers:
                with record_function(f"phase/{type(optimizer).__name__}"):
                    optimizer.step()
            return loss

        setup = perf_counter()
        if variant != "eager":
            adam = optimizers[-1]
            for group in adam.param_groups:
                group["capturable"] = True
            for state in adam.state.values():
                state["step"] = state["step"].cuda()
            counter = torch.tensor(float(checkpoint["step"]), device="cuda")

            def capture_step() -> torch.Tensor:
                model.zero_grad(set_to_none=True)
                loss = relative_l2(forward(*static[:3]), static[3])
                loss.backward()
                with torch.no_grad():
                    counter.add_(1)
                    correction = 1 - .8**counter
                    gradients = [p.grad for p in parameters]
                    torch._foreach_mul_(averages, .8)
                    torch._foreach_add_(averages, gradients, alpha=.2)
                    torch._foreach_copy_(gradients, torch._foreach_div(averages, correction))
                for optimizer in optimizers:
                    optimizer.step()
                return loss

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    capture_step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_loss = capture_step()
            # Restore values in place: the graph retains pointers to these tensors.
            model.load_state_dict(checkpoint["model"])
            for optimizer, saved in zip(optimizers, checkpoint["optimizers"], strict=True):
                for current, old in zip(optimizer.state.values(), saved["state"].values(), strict=True):
                    for key, value in current.items():
                        if isinstance(value, torch.Tensor):
                            value.copy_(old[key])
            for current, old in zip(averages, checkpoint["gradient_ema"], strict=True):
                current.copy_(old)
            counter.fill_(checkpoint["step"])
        torch.cuda.synchronize()
        setup_seconds = perf_counter() - setup
        history = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(32):
            torch.cuda.synchronize()
            tick = perf_counter()
            try:
                with record_function("phase/fetch"):
                    batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if variant == "eager":
                loss = eager_step(batch)
            else:
                for buffer, value in zip(static, batch, strict=True):
                    buffer.copy_(value)
                graph.replay()
                loss = graph_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
            value = loss.item()
            torch.cuda.synchronize()
            history.append({"loss": value, "seconds": perf_counter() - tick})
        with torch.no_grad():
            prediction = model(*static[:3]).detach().clone()
        if reference is None:
            reference = {"parameters": [p.detach().clone() for p in parameters],
                         "ema": [a.clone() for a in averages], "losses": [r["loss"] for r in history]}
        errors = {"max_parameter_absolute": max((p - ref).abs().max().item() for p, ref in zip(parameters, reference["parameters"], strict=True)),
                  "max_ema_absolute": max((a - ref).abs().max().item() for a, ref in zip(averages, reference["ema"], strict=True)),
                  "max_loss_absolute": max(abs(row["loss"] - ref) for row, ref in zip(history, reference["losses"], strict=True))}
        assert all(torch.isfinite(p).all() for p in parameters)
        assert torch.isfinite(prediction).all()
        result["variants"][variant] = {"history": history, "median_ms": 1000 * median(r["seconds"] for r in history[3:]),
                                         "setup_seconds": setup_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                         "agreement": errors}
        print(variant, {k: v for k, v in result["variants"][variant].items() if k != "history"}, flush=True)
        if variant == "eager":
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                for _ in range(3):
                    batch = next(iter(loader))
                    eager_step(batch).item()
                    torch.cuda.synchronize()
            raw_trace = destination / "eager_trace.json"
            trace.export_chrome_trace(str(raw_trace))
            result["profile"] = sorted([{"name": e.key, "calls": e.count,
                                          "cpu_total_us": e.cpu_time_total,
                                          "self_device_us": e.self_device_time_total,
                                          "device_total_us": e.device_time_total}
                                         for e in trace.key_averages()], key=lambda r: r["self_device_us"], reverse=True)
            with gzip.open(destination / "eager_trace.json.gz", "wb") as compressed:
                compressed.write(raw_trace.read_bytes())
            raw_trace.unlink()
        (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
    return result


@app.local_entrypoint()
def main(run_name: str = "torch_spectral_optimize_20260925", compile_model: bool = False) -> None:
    result = run.remote(run_name, compile_model)
    (Path(__file__).parent / f"{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {key: value for key, value in v.items() if key != "history"}
                      for k, v in result["variants"].items()}, indent=2))
