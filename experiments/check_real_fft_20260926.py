"""Check real FFT adjoints, endpoint weights, and full-model gradients."""

import json
from pathlib import Path
import sys
import subprocess
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-attention"))
import real_fft
import spectral

torch.manual_seed(1)
errors = []
for n in (7, 8, 31, 32, 1024):
    for norm in ("backward", "forward", "ortho"):
        for inverse in (False, True):
            x = torch.randn(2, n // 2 + 1 if inverse else n,
                            dtype=torch.complex128 if inverse else torch.float64, requires_grad=True)
            actual = (real_fft.irfft if inverse else real_fft.rfft)(x, n=n, norm=norm)
            expected = (torch.fft.irfft if inverse else torch.fft.rfft)(x, n=n, norm=norm)
            gradient = torch.randn_like(actual)
            a = torch.autograd.grad(actual, x, gradient)[0]
            b = torch.autograd.grad(expected, x, gradient)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
            errors.append((a - b).abs().max().item())

model = spectral.SpectralDNO(n=32, width=16, heads=4, depth=2, branches=4).double()
with torch.no_grad():
    model.decoder.weight.normal_(std=.1)
inputs = [torch.randn(2, 32, dtype=torch.float64, requires_grad=True) * .02,
          torch.randn(2, 32, dtype=torch.float64, requires_grad=True),
          torch.ones(2, 1, dtype=torch.float64, requires_grad=True)]
probe = torch.randn(2, 32, dtype=torch.float64)
outputs, gradients = [], []
for custom in (False, True):
    spectral.rfft = real_fft.rfft if custom else torch.fft.rfft
    spectral.irfft = real_fft.irfft if custom else torch.fft.irfft
    value = model(*inputs)
    outputs.append(value)
    gradients.append(torch.autograd.grad(value, tuple(inputs) + tuple(model.parameters()), probe))
torch.testing.assert_close(*outputs, rtol=0, atol=0)
for actual, expected in zip(*gradients, strict=True):
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
report = {"fft_gradient_max_abs_difference": max(errors), "full_model_outputs_identical": True,
          "full_model_gradient_max_abs_difference": max((a - b).abs().max().item()
                                                        for a, b in zip(*gradients, strict=True)),
          "fft_cases": len(errors), "dtype": "float64", "nonzero_correction_head": True}
reference_module = types.ModuleType("spectral_original")
exec(compile(subprocess.check_output(["git", "show", "7d673f6:torch-attention/spectral.py"],
                                    cwd=Path(__file__).resolve().parents[1]), "spectral_original.py", "exec"),
     reference_module.__dict__)
reference = reference_module.SpectralDNO(n=32, width=16, heads=4, depth=2, branches=4).double()
reference.load_state_dict(model.state_dict())
actual, expected = model(*inputs), reference(*inputs)
torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
a = torch.autograd.grad(actual, tuple(inputs) + tuple(model.parameters()), probe)
b = torch.autograd.grad(expected, tuple(inputs) + tuple(reference.parameters()), probe)
for first, second in zip(a, b, strict=True):
    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-12)
report["original_model_output_max_difference"] = (actual - expected).abs().max().item()
report["original_model_gradient_max_difference"] = max((first - second).abs().max().item()
                                                       for first, second in zip(a, b, strict=True))
Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
