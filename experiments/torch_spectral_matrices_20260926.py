"""H100 matrix-shape census and model-preserving attention microbenchmarks."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("dno-fno-train-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "numpy==2.4.2")
         .add_local_dir(ROOT / "torch-attention", "/repo/torch-attention")
         .add_local_dir(ROOT.parent / "local-data/paper_equal_subset_20260924", "/subset"))
app = modal.App("dno-spectral-matrix-shapes")


@app.function(image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4,
              memory=16384, timeout=600, retries=0, scaledown_window=2)
def run() -> dict:
    from collections.abc import Callable
    import gc
    import os
    from statistics import median
    from time import perf_counter

    import torch
    from torch import Tensor
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch-inductor-cache"
    torch.manual_seed(0)
    started = perf_counter()
    output_dir = Path("/data/experiments/torch_spectral_matrices_20260926")
    output_dir.mkdir(parents=True, exist_ok=False)

    def measure(step: Callable[[], None]) -> float:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
        times = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 10)
        return median(times)

    report = {"gpu": torch.cuda.get_device_name(), "batch_size": 256, "matrices": [], "attention": []}
    # M counts all positions across the batch. These are tall, narrow GEMMs.
    shapes = (("encoder_7_64", 262144, 7, 64, True, False, 1),
              ("encoder_64_64", 262144, 64, 64, True, True, 1),
              ("frequency_in", 131328, 132, 64, True, True, 1),
              ("qkv", 131328, 64, 192, True, True, 2),
              ("attention_out", 131328, 64, 64, True, True, 2),
              ("mlp_up", 131328, 64, 128, True, True, 2),
              ("mlp_down", 131328, 128, 64, True, True, 2),
              ("decoder", 262144, 64, 32, False, True, 1),
              ("depth_in", 131328, 4, 32, False, False, 1),
              ("depth_out", 131328, 32, 32, False, True, 1))
    shapes += (("frequency_out", 131328, 64, 128, True, True, 1),)
    for name, m, k, n, bf16, input_grad, count in shapes:
        layer = torch.nn.Linear(k, n, bias=not name.startswith("encoder") and name != "decoder").cuda()
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16 if bf16 else torch.float32,
                        requires_grad=input_grad)
        dy = torch.randn(m, n, device="cuda", dtype=x.dtype)
        for p in layer.parameters():
            p.grad = torch.zeros_like(p)
        if input_grad:
            x.grad = torch.zeros_like(x)

        def forward(value: Tensor) -> Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                return layer(value)

        compiled = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False})

        def update() -> None:
            for parameter in layer.parameters():
                parameter.grad.zero_()
            if input_grad:
                x.grad.zero_()
            compiled(x).backward(dy)

        ms = measure(update)
        flops = (3 if input_grad else 2) * 2 * m * k * n
        row = {"name": name, "M": m, "K": k, "N": n, "dtype": str(x.dtype),
               "occurrences": count, "forward_backward_ms": ms,
               "useful_TFLOP_s": flops / (ms * 1e9)}
        report["matrices"].append(row)
        print(json.dumps(row), flush=True)
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    base = torch.randn(256, 513, 192, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(256, 513, 64, device="cuda", dtype=torch.bfloat16)
    reference = None
    for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION):
        for padded in (16, 32, 64):
            qkv = base.clone().requires_grad_()
            qkv.grad = torch.zeros_like(qkv)
            out = None

            def attention(value: Tensor) -> Tensor:
                q, k, v = (t.reshape(256, 513, 4, 16).transpose(1, 2) for t in value.chunk(3, -1))
                if padded != 16:
                    q, k, v = (F.pad(t, (0, padded - 16)) for t in (q, k, v))
                attended = F.scaled_dot_product_attention(q, k, v, scale=16**-.5)[..., :16]
                return attended.transpose(1, 2).reshape(256, 513, 64)

            with sdpa_kernel(backend):
                compiled = torch.compile(attention, fullgraph=True, options={"triton.cudagraphs": False})

                def update_attention() -> None:
                    nonlocal out
                    qkv.grad.zero_()
                    out = compiled(qkv)
                    out.backward(upstream)

                ms = measure(update_attention)
            if reference is None:
                reference = out.detach().clone(), qkv.grad.clone()
            errors = [((a.float() - b.float()).norm() / b.float().norm()).item()
                      for a, b in zip((out.detach(), qkv.grad), reference, strict=True)]
            assert all(value < .02 for value in errors), errors
            row = {"backend": str(backend), "head_dim": 16, "padded_dim": padded,
                   "forward_backward_ms": ms, "output_relative_difference": errors[0],
                   "gradient_relative_difference": errors[1]}
            report["attention"].append(row)
            print(json.dumps(row), flush=True)
            (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            volume.commit()
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
    report["function_seconds"] = perf_counter() - started
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
