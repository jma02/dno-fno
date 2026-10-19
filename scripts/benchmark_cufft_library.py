"""Standalone cuFFT timing and an opt-in capture-compatible JAX FFT wrapper."""
from __future__ import annotations

import argparse
import ctypes as ct
from functools import cache, partial
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sysconfig
import time
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import jax

ROOT = Path(__file__).resolve().parents[1]
FFI_LIBRARY = ROOT / "outputs/cufft_graph_ffi_20261019/cufft_graph_ffi_state.so"


@cache
def register_graph_ffi(library: Path, input_alias: bool = False) -> ct.CDLL:
    import jax

    module = ct.CDLL(str(library.resolve()))
    registration = {}
    for name in ("type_id", "type_info"):
        getter = getattr(module, f"dno_cufft_{name}")
        getter.restype = ct.c_void_p
        registration[name] = jax.ffi.pycapsule(ct.c_void_p(getter()))
    suffix = "_alias" if input_alias else ""
    jax.ffi.register_ffi_type(f"dno_cufft_plan{suffix}", {
        "type_id": registration["type_id"], "type_info": registration["type_info"],
    }, platform="CUDA")
    jax.ffi.register_ffi_target(f"dno_cufft_graph{suffix}", {
        "instantiate": jax.ffi.pycapsule(module.dno_cufft_instantiate),
        "initialize": jax.ffi.pycapsule(getattr(module, f"dno_cufft{suffix}_initialize")),
        "execute": jax.ffi.pycapsule(getattr(module, f"dno_cufft{suffix}_execute")),
    }, platform="CUDA")
    return module


def graph_fft(a: jax.Array, n: int | None = None, axis: int = -1, norm: str | None = None,
              *, inverse: bool = False,
              library: Path = FFI_LIBRARY, input_alias: bool = False,
              endpoint_mask: bool = False, hermitian_input: bool = False) -> jax.Array:
    """Real FFT with JAX axis/normalization semantics and capture-compatible cuFFT.

    Executable-owned plans are initialized per device before capture. Scratch is
    an extra XLA result. The optional alias trial lets XLA reuse dead inputs or
    copy live inputs before C2R, instead of always copying inside the handler;
    forward transforms keep the accepted non-destructive implementation.
    ``hermitian_input`` skips endpoint clearing when the caller guarantees real
    DC and, for even lengths, Nyquist coefficients.
    """
    import jax
    import jax.numpy as jnp

    if input_alias and not inverse:
        return graph_fft(a, n=n, axis=axis, norm=norm, library=FFI_LIBRARY)
    expected = (np.dtype("complex64"), np.dtype("complex128")) if inverse else (
        np.dtype("float32"), np.dtype("float64"))
    if a.ndim < 1 or not -a.ndim <= axis < a.ndim or norm not in (None, "backward", "forward", "ortho"):
        raise ValueError("graph_fft requires a valid axis and JAX FFT normalization mode")
    a = jnp.moveaxis(a, axis, -1)
    if n is None:
        n = 2 * (a.shape[-1] - 1) if inverse else a.shape[-1]
    if a.dtype not in expected or a.ndim < 1 or n < 1:
        raise ValueError("graph_fft requires positive n and real/complex FP32 or FP64 axis-last inputs")
    register_graph_ffi(library, input_alias)
    input_n = n // 2 + 1 if inverse else n
    a = a[..., :input_n]
    if a.shape[-1] < input_n:
        a = jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, input_n - a.shape[-1])])
    if inverse and not hermitian_input:
        if endpoint_mask:
            mode = jnp.arange(input_n)
            a = jnp.where((mode == 0) | ((mode == input_n - 1) & (n % 2 == 0)), a.real, a)
        else:
            a = a.at[..., 0].set(a[..., 0].real)
            if n % 2 == 0:
                a = a.at[..., -1].set(a[..., -1].real)
    fp64 = a.dtype in (np.dtype("float64"), np.dtype("complex128"))
    real_dtype = np.dtype("float64" if fp64 else "float32")
    complex_dtype = np.dtype("complex128" if fp64 else "complex64")
    output = jax.ShapeDtypeStruct((*a.shape[:-1], n if inverse else n // 2 + 1),
                                real_dtype if inverse else complex_dtype)
    # cuFFT documents an 8*batch*n complex-element worst-case workspace bound.
    # Initialization verifies the actual requirement before execution/capture.
    copy_bytes = a.size * a.dtype.itemsize if inverse and not input_alias else 0
    scratch_bytes = (copy_bytes + 255) // 256 * 256 + 8 * math.prod(a.shape[:-1]) * n * complex_dtype.itemsize
    results = [output, jax.ShapeDtypeStruct((scratch_bytes,), jnp.uint8)]
    if input_alias:
        results.append(jax.ShapeDtypeStruct(a.shape, a.dtype))
    result = jax.ffi.ffi_call(f"dno_cufft_graph{'_alias' if input_alias else ''}",
        results, vmap_method="broadcast_all", input_output_aliases={0: 2} if input_alias else {},
    )(a, n=np.int64(n), inverse=inverse)[0]
    result = cast(jax.Array, result)
    scale = (1 / math.sqrt(n) if norm == "ortho" else
             1 / n if (inverse and norm in (None, "backward")) or (not inverse and norm == "forward") else 1)
    if scale != 1:
        result = result * jnp.asarray(scale, dtype=real_dtype)
    return jnp.moveaxis(result, -1, axis)


def checked(library: ct.CDLL, name: str, *args: object) -> None:
    status = getattr(library, name)(*args)
    if status:
        raise RuntimeError(f"{name} failed with status {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-ffi", type=Path, metavar="NEW_SO", help="Build the opt-in JAX wrapper without GPU calls; refuse an existing output.")
    parser.add_argument("--check-ffi", type=Path, metavar="SO", help="Run native-GPU correctness checks only; print results, without timing or files.")
    parser.add_argument("--alias-ffi-input", action="store_true", help="Check the explicit input/output-alias trial in --check-ffi.")
    parser.add_argument("--mask-ffi-endpoints", action="store_true", help="Check elementwise endpoint masking instead of scatters in --check-ffi.")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.build_ffi:
        import jax

        destination = args.build_ffi.resolve()
        if destination.exists():
            parser.error("--build-ffi must name a new output; existing libraries are never overwritten")
        cufft_root = Path(sysconfig.get_path("purelib")) / "nvidia/cufft"
        cufft_library = cufft_root / "lib/libcufft.so.11"
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            shutil.which("nvcc") or "/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O3",
            "--shared", "-Xcompiler", "-fPIC", f"-I{jax.ffi.include_dir()}",
            f"-I{cufft_root / 'include'}", str(ROOT / "scripts/cufft_graph_ffi.cu"),
            "-Xlinker", str(cufft_library), "-Xlinker", "-rpath", "-Xlinker", str(cufft_library.parent),
            "-o", str(destination),
        ], check=True)
        print(f"Built {destination}; linked {cufft_library}; no GPU execution")
        return
    if args.check_ffi:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        if jax.default_backend() != "gpu":
            parser.error("--check-ffi requires a GPU backend")
        ffi_module = register_graph_ffi(args.check_ffi, args.alias_ffi_input)
        transform_ffi = partial(graph_fft, library=args.check_ffi, input_alias=args.alias_ffi_input,
                                endpoint_mask=args.mask_ffi_endpoints)
        version = ct.c_int()
        checked(ffi_module, "cufftGetVersion", ct.byref(version))
        libraries = sorted({line.rsplit(None, 1)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "libcufft.so" in line})
        print(json.dumps({"ffi_library": str(args.check_ffi.resolve()), "cufft_version": version.value,
                          "mapped_cufft_libraries": libraries}), flush=True)
        rng = np.random.default_rng(20261019)
        count, maximum = 0, {"float32": 0.0, "float64": 0.0}
        for dtype in (np.float32, np.float64):
            tolerance = 2e-6 if dtype == np.float32 else 2e-12
            for inverse, shape, axis, n, vmapped in (
                (False, (2, 1024), -1, 1024, False), (False, (2, 1024, 8), 1, 1024, False),
                (False, (1, 1024), -1, 2048, False), (False, (32, 1024), -1, 1024, False),
                (True, (2, 513), -1, 1024, False), (True, (2, 513, 8), 1, 1024, False),
                (True, (2, 513), -1, 2048, False), (True, (32, 513), -1, 1024, False),
                (False, (2, 3, 1024), -1, 1024, True), (True, (2, 3, 513), -1, 1024, True),
            ):
                host = rng.normal(size=shape).astype(dtype)
                if inverse:
                    host = (host + 1j * rng.normal(size=shape)).astype(
                        np.complex64 if dtype == np.float32 else np.complex128)
                values = jnp.asarray(host)
                reference = jnp.fft.irfft if inverse else jnp.fft.rfft
                expected = np.asarray(jax.jit(lambda x: reference(x, n=n, axis=axis))(values))
                ffi_transform = partial(transform_ffi, n=n, axis=axis, inverse=inverse)
                candidate = jax.jit(jax.vmap(ffi_transform) if vmapped else ffi_transform)
                for _ in range(2):
                    actual = np.asarray(candidate(values))
                    relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                    if not np.isfinite(relative) or relative > tolerance:
                        raise AssertionError(f"cuFFT FFI {dtype}/{inverse}/{shape}/vmap={vmapped}: {relative}")
                    maximum[np.dtype(dtype).name] = max(maximum[np.dtype(dtype).name], relative)
                    np.testing.assert_array_equal(np.asarray(values), host)
                # Keep the caller's operand live across the FFI invocation.
                live = jax.jit(lambda x: (ffi_transform(x), 2 * x))(values)
                np.testing.assert_array_equal(np.asarray(live[1]), 2 * host)
                np.testing.assert_allclose(np.asarray(live[0]), expected,
                                           rtol=tolerance, atol=tolerance * np.max(np.abs(expected)))
                count += 1
            values = jnp.asarray(rng.normal(size=(2, 1024)).astype(dtype))
            roundtrip = jax.jit(lambda x: transform_ffi(transform_ffi(x), inverse=True))
            np.testing.assert_allclose(np.asarray(roundtrip(values)), np.asarray(values),
                                       rtol=tolerance, atol=tolerance)

            def scan(values: jax.Array, use_ffi: bool) -> jax.Array:
                def step(state: jax.Array, unused: None) -> tuple[jax.Array, None]:
                    if use_ffi:
                        restored = transform_ffi(transform_ffi(state), inverse=True)
                    else:
                        restored = jnp.fft.irfft(jnp.fft.rfft(state), n=state.shape[-1])
                    return 0.999 * restored + 0.001 * jnp.roll(state, 1, axis=-1), None

                return jax.lax.scan(step, values, None, length=16)[0]

            expected = np.asarray(jax.jit(lambda x: scan(x, False))(values))
            scanned = jax.jit(lambda x: scan(x, True)).lower(values).compile()
            scan_errors = []
            # Reuse this executable without intervening compilations: cached
            # graphs must retain live cuFFT plans after the first execution.
            for repeat in range(4):
                actual = np.asarray(scanned(values))
                relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                if not np.isfinite(relative) or relative > tolerance:
                    raise AssertionError(f"cuFFT FFI dependent scan {dtype}/repeat={repeat}: {relative}")
                scan_errors.append(relative)
            hlo = cast(str, scanned.as_text())
            print(json.dumps({"dtype": np.dtype(dtype).name, "scan_relative_errors": scan_errors,
                              "scan_native_fft_ops": hlo.count("fft_type="),
                              "scan_command_buffer_mentions": hlo.count("command_buffer") + hlo.count("command-buffer")}))
        print(json.dumps({"checks": count, "max_relative_error": maximum,
                          "scope": "GPU correctness only; HLO mentions alone do not prove whole-loop capture"}))
        return
    if args.library is None or args.output is None:
        parser.error("--library and --output are required for standalone GPU timing")
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
