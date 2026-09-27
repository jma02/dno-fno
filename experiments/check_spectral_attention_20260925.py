"""Check the new correction's physical invariants and real-data gradient flow."""

import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from model import baseline  # noqa: E402
from spectral import SpectralDNO, surface_features  # noqa: E402
from train import Waves, build_model, build_optimizers, relative_l2  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(0)
    x = torch.arange(64, dtype=torch.float64) * (2 * math.pi / 64)
    wave = .2 * torch.sin(3 * x)
    expected = torch.stack((wave, wave**2, wave**3, .6 * torch.cos(3 * x),
                            -9 * wave, math.sqrt(3) * wave, -.2 * torch.cos(3 * x)), -1)
    torch.testing.assert_close(surface_features(wave[None], 2 * math.pi)[0], expected,
                               rtol=1e-10, atol=1e-12)
    model = SpectralDNO(n=64, width=16, branches=4).double()
    eta, xi, other = torch.randn(3, 3, 64, dtype=torch.float64)
    h = torch.tensor([.2, 1., 3.], dtype=torch.float64)
    torch.testing.assert_close(model(eta, xi, h), baseline(eta, xi, h, model.length), rtol=0, atol=0)
    with torch.no_grad():
        model.decoder.weight.normal_(std=.1)
    correction = model.correction(eta, xi, h)
    linear = model.correction(eta, .3 * xi - .7 * other, h)
    torch.testing.assert_close(linear, .3 * correction - .7 * model.correction(eta, other, h), rtol=1e-10, atol=1e-10)
    left = (other * correction).sum(-1)
    right = (xi * model.correction(eta, other, h)).sum(-1)
    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
    flat = torch.zeros_like(eta, requires_grad=True)
    flat_correction = model.correction(flat, xi, h)
    assert torch.count_nonzero(flat_correction) == 0
    assert torch.count_nonzero(torch.autograd.grad((flat_correction * other).sum(), flat)[0]) == 0
    assert torch.count_nonzero(model.correction(eta, torch.ones_like(xi), h)) == 0
    # Smooth, low-amplitude eta keeps derivative features in the asymptotic regime.
    a = model.correction(1e-4 * wave.expand_as(eta), xi, h).norm()
    b = model.correction(5e-5 * wave.expand_as(eta), xi, h).norm()
    ratio = (a / b).item()
    assert abs(ratio - 4) < .01, ratio
    dataset = Waves(ROOT.parent / "local-data/paper_equal_subset_20260924", "train")
    real_model = build_model("spectral", dataset, 1024, 2 * math.pi, 2, False)
    optimizers = build_optimizers(real_model, "muon", 1e-5)
    batch = default_collate([dataset[i] for i in (0, 1024, 2048, 3072)])
    losses = []
    for _ in range(3):
        real_model.zero_grad(set_to_none=True)
        loss = relative_l2(real_model(*batch[:3]), batch[3])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in real_model.parameters())
        for optimizer in optimizers:
            optimizer.step()
        losses.append(loss.item())
    gradients = {group: sum(p.grad.detach().square().sum().item() for name, p in real_model.named_parameters()
                           if name.split(".")[0] == group)**.5
                 for group in ("encoder", "frequency", "frequency_in", "frequency_out", "decoder", "depth_filters")}
    assert all(value > 0 for value in gradients.values())
    result = {"invariants": "feature formulas, initial baseline, linearity, self-adjointness, zero flat correction/derivative, constant-xi nullspace passed",
              "quadratic_ratio": ratio, "parameters": sum(p.numel() for p in real_model.parameters()),
              "training_feature_scales": real_model.feature_scales.tolist(),
              "four_example_cpu_smoke_losses": losses, "final_group_gradient_norms": gradients}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
