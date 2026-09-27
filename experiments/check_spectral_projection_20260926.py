"""Check optional Fourier projection, constrained operator and checkpoint compatibility."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT / "torch-attention"))
    from model import baseline
    from spectral import SpectralDNO

    torch.set_num_threads(8)
    torch.manual_seed(926)
    results = []
    for n, cutoff, length in ((32, 8, 6.283185307179586), (64, 12, 4.)):
        model = SpectralDNO(n=n, width=16, depth=2, branches=4, length=length, max_mode=cutoff).double()
        torch.nn.init.normal_(model.decoder.weight, std=.03)
        original = SpectralDNO(n=n, width=16, depth=2, branches=4, length=length).double()
        original.load_state_dict(model.state_dict(), strict=True)
        eta = (torch.randn(2, n, dtype=torch.float64) * .001).requires_grad_()
        xi, other = (torch.randn(2, n, dtype=torch.float64) for _ in range(2))
        h = torch.tensor([[.07], [2.]], dtype=torch.float64)
        predicted = model(eta, xi, h)
        reference = model.project(original(eta, model.project(xi), h))
        torch.testing.assert_close(predicted, reference, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(model(eta, .7 * xi - 1.3 * other, h),
                                   .7 * predicted - 1.3 * model(eta, other, h), atol=1e-11, rtol=1e-11)
        lhs = (other * predicted).sum(-1)
        rhs = (xi * model(eta, other, h)).sum(-1)
        torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(model(eta, torch.ones_like(xi), h), torch.zeros_like(xi), atol=1e-12, rtol=0)
        leakage = torch.fft.rfft(predicted)[:, cutoff + 1:].abs().max().item()
        assert leakage < 1e-10
        actual_grad = torch.autograd.grad((predicted * other).sum(), (eta, *model.parameters()))
        expected_grad = torch.autograd.grad((reference * other).sum(), (eta, *original.parameters()))
        for a, b in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        flat = torch.zeros_like(eta, requires_grad=True)
        correction = model(flat, xi, h) - model.project(baseline(flat, model.project(xi), h, length))
        torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-12, rtol=0)
        derivative = torch.autograd.grad((correction * other).sum(), flat)[0]
        torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-10, rtol=0)
        results.append({"n": n, "cutoff": cutoff, "length": length, "max_output_leakage": leakage,
                        "self_adjoint_absolute_error": (lhs-rhs).abs().max().item()})
    checkpoint = torch.load(ROOT.parent / "pilot-checkpoints/torch_spectral_width256_epoch2_20260926.pt",
                            map_location="cpu", weights_only=True)
    model = SpectralDNO(n=1024, width=256, depth=2, branches=32, max_mode=128, bf16=True,
                        feature_scales=checkpoint["model"]["feature_scales"])
    model.load_state_dict(checkpoint["model"], strict=True)
    import numpy as np
    source = ROOT.parent / "local-data/paper_equal_subset_20260924"
    batch = [torch.as_tensor(np.load(source / f"{key}.npy")[:4], dtype=torch.float32)
             for key in ("eta", "xi", "depth")]
    with torch.no_grad():
        predicted = model(*batch)
        spectrum = torch.fft.rfft(predicted)
        leakage = (spectrum[:, 129:].abs().norm(dim=-1) / spectrum.abs().norm(dim=-1)).max().item()
        assert leakage < 1e-6 and torch.isfinite(predicted).all()
    report = {"small_model_invariants": results, "production_checkpoint_bf16_max_outband_relative_norm": leakage,
              "checks": ["P G P forward and gradients", "xi linearity", "self-adjointness",
                         "constant-xi nullspace", "zero flat correction and surface derivative",
                         "strict legacy checkpoint loading", "real BF16 checkpoint finite band-limited output"]}
    (ROOT / "experiments/check_spectral_projection_20260926.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
