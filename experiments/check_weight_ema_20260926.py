"""Check weight EMA against a closed-form average and unchanged online training."""

import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from spectral import SpectralDNO  # noqa: E402
from weight_ema import WeightEMA  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(926)
    for decay in (0., .9, .999):
        model = torch.nn.Linear(3, 2).double()
        model.register_buffer("fixed", torch.tensor([2.]))
        control = copy.deepcopy(model)
        ema = WeightEMA(model, decay)
        history = [{name: p.detach().clone() for name, p in model.named_parameters()}]
        optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in (model, control)]
        for _ in range(7):
            x = torch.randn(4, 3, dtype=torch.float64)
            for m, optimizer in zip((model, control), optimizers, strict=True):
                optimizer.zero_grad()
                m(x).square().mean().backward()
                optimizer.step()
            ema.update()
            history.append({name: p.detach().clone() for name, p in model.named_parameters()})
        state = ema.state_dict()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, dict(control.named_parameters())[name], rtol=0, atol=0)
            expected = decay**7 * history[0][name] + sum(
                (1 - decay) * decay**(7 - i) * history[i][name] for i in range(1, 8))
            torch.testing.assert_close(state[name], expected, rtol=1e-13, atol=1e-15)
        assert ema.updates == 7 and torch.equal(state["fixed"], model.fixed)
        state["weight"].zero_()
        assert torch.count_nonzero(ema.state_dict()["weight"]) > 0

    model = SpectralDNO(n=32, width=16, heads=4, branches=4, depth=2, max_mode=4).double()
    ema = WeightEMA(model, .999)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.01 * torch.randn_like(parameter))
    ema.update()
    model.load_state_dict(ema.state_dict(), strict=True)
    eta, x, y = (torch.randn(2, 32, dtype=torch.float64) * .01 for _ in range(3))
    depth = torch.tensor([.1, 1.], dtype=torch.float64)
    with torch.no_grad():
        ax, ay = model(eta, x, depth), model(eta, y, depth)
        torch.testing.assert_close(model(eta, 2 * x - y, depth), 2 * ax - ay, rtol=1e-10, atol=1e-13)
        torch.testing.assert_close((ax * y).sum(-1), (x * ay).sum(-1), rtol=1e-10, atol=1e-13)
        torch.testing.assert_close(model.correction(torch.zeros_like(eta), x, depth), torch.zeros_like(x))
    result = {"decays_checked": [0., .9, .999], "closed_form_average": "passed",
              "online_training_unchanged": "passed", "fixed_buffers_and_no_aliasing": "passed",
              "averaged_model_linearity_self_adjointness_flat_correction": "passed"}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
