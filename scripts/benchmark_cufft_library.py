"""Measure one explicitly selected cuFFT library in a fresh process, without JAX."""
from __future__ import annotations

import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


def checked(library: ct.CDLL, name: str, *args: object) -> None:
    status = getattr(library, name)(*args)
    if status:
        raise RuntimeError(f"{name} failed with status {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    driver = ct.CDLL("libcuda.so.1")
    cufft = ct.CDLL(str(args.library.resolve()))
    checked(driver, "cuInit", 0)
    device, context, stream = ct.c_int(), ct.c_void_p(), ct.c_void_p()
    checked(driver, "cuDeviceGet", ct.byref(device), 0)
    checked(driver, "cuDevicePrimaryCtxRetain", ct.byref(context), device)
    checked(driver, "cuCtxSetCurrent", context)
    checked(driver, "cuStreamCreate", ct.byref(stream), 1)
    version = ct.c_int()
    checked(cufft, "cufftGetVersion", ct.byref(version))
    result = {
        "library": str(args.library.resolve()), "cufft_version": version.value,
        "device": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,pstate",
            "--format=csv,noheader"], text=True).strip(),
        "timing": "256 cuFFT executions per CUDA graph; zero inputs; random-input correctness",
        "measurements": [],
    }
    rng = np.random.default_rng(17)
    specs = [(kind, n, batch) for kind, n in (
        ("Z2Z", 1024), ("D2Z", 1024), ("Z2D", 1024), ("Z2D", 2048))
        for batch in (1, 2, 4)] + [("R2C", 1024, 32)]
    for kind, n, batch in specs:
        real = rng.normal(size=(batch, n))
        if kind == "Z2Z":
            source = (real + 1j * rng.normal(size=real.shape)).astype(np.complex128)
            expected, transform = np.fft.fft(source), 0x69
        elif kind == "Z2D":
            source = np.fft.rfft(real)
            expected, transform = real * n, 0x6C
        else:
            source = real.astype(np.float32 if kind == "R2C" else np.float64)
            expected = np.fft.rfft(source).astype(np.complex64 if kind == "R2C" else np.complex128)
            transform = 0x2A if kind == "R2C" else 0x6A
        output = np.empty_like(expected)
        input_device, output_device = ct.c_uint64(), ct.c_uint64()
        checked(driver, "cuMemAlloc_v2", ct.byref(input_device), ct.c_size_t(source.nbytes))
        checked(driver, "cuMemAlloc_v2", ct.byref(output_device), ct.c_size_t(output.nbytes))
        checked(driver, "cuMemcpyHtoD_v2", input_device, ct.c_void_p(source.ctypes.data), ct.c_size_t(source.nbytes))
        plan = ct.c_int()
        checked(cufft, "cufftPlan1d", ct.byref(plan), n, transform, batch)
        checked(cufft, "cufftSetStream", plan, stream)
        fft_args = (plan, ct.c_void_p(input_device.value), ct.c_void_p(output_device.value))
        if kind == "Z2Z":
            fft_args += (ct.c_int(-1),)
        checked(cufft, f"cufftExec{kind}", *fft_args)
        checked(driver, "cuStreamSynchronize", stream)
        checked(driver, "cuMemcpyDtoH_v2", ct.c_void_p(output.ctypes.data), output_device, ct.c_size_t(output.nbytes))
        error = float(np.linalg.norm(output - expected) / np.linalg.norm(expected))
        tolerance = 2e-6 if kind == "R2C" else 2e-12
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(f"{kind}/{n}/{batch}: relative error {error}")
        # C2R can overwrite its input; zero-valued timing inputs remain stable.
        checked(driver, "cuMemsetD8_v2", input_device, 0, ct.c_size_t(source.nbytes))
        for _ in range(10):
            checked(cufft, f"cufftExec{kind}", *fft_args)
        checked(driver, "cuStreamSynchronize", stream)
        graph, executable = ct.c_void_p(), ct.c_void_p()
        checked(driver, "cuStreamBeginCapture", stream, 0)
        for _ in range(256):
            checked(cufft, f"cufftExec{kind}", *fft_args)
        checked(driver, "cuStreamEndCapture", stream, ct.byref(graph))
        checked(driver, "cuGraphInstantiateWithFlags", ct.byref(executable), graph, ct.c_uint64(0))
        warm_until = time.perf_counter() + 0.2
        while time.perf_counter() < warm_until:
            checked(driver, "cuGraphLaunch", executable, stream)
            checked(driver, "cuStreamSynchronize", stream)
        start, stop = ct.c_void_p(), ct.c_void_p()
        checked(driver, "cuEventCreate", ct.byref(start), 0)
        checked(driver, "cuEventCreate", ct.byref(stop), 0)
        graph_us, wall_us = [], []
        for _ in range(args.repeats):
            checked(driver, "cuEventRecord", start, stream)
            checked(driver, "cuGraphLaunch", executable, stream)
            checked(driver, "cuEventRecord", stop, stream)
            checked(driver, "cuEventSynchronize", stop)
            elapsed = ct.c_float()
            checked(driver, "cuEventElapsedTime", ct.byref(elapsed), start, stop)
            graph_us.append(elapsed.value * 1000 / 256)
            before = time.perf_counter()
            checked(cufft, f"cufftExec{kind}", *fft_args)
            checked(driver, "cuStreamSynchronize", stream)
            wall_us.append((time.perf_counter() - before) * 1e6)
        record = {"kind": kind, "n": n, "batch": batch, "relative_error": error,
                  "graph_us": statistics.median(graph_us), "graph_samples_us": graph_us,
                  "synchronized_wall_us": statistics.median(wall_us)}
        result["measurements"].append(record)
        print(json.dumps(record), flush=True)
        checked(driver, "cuEventDestroy_v2", start)
        checked(driver, "cuEventDestroy_v2", stop)
        checked(driver, "cuGraphExecDestroy", executable)
        checked(driver, "cuGraphDestroy", graph)
        checked(cufft, "cufftDestroy", plan)
        checked(driver, "cuMemFree_v2", input_device)
        checked(driver, "cuMemFree_v2", output_device)
    checked(driver, "cuStreamDestroy_v2", stream)
    checked(driver, "cuDevicePrimaryCtxRelease_v2", device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
