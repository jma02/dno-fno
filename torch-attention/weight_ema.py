"""Weight averaging for evaluation, independent of optimizer/gradient state."""

import torch
from torch import Tensor, nn


class WeightEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("Weight EMA decay must be in [0,1)")
        self.model, self.decay, self.updates = model, decay, 0
        self.parameters = list(model.parameters())
        self.averages = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self.averages, self.parameters, 1 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict[str, Tensor]:
        # This model's buffers are fixed physical features/scales, not running
        # statistics. Copy them unchanged; average only the learned parameters.
        state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        state.update({name: value.clone() for (name, _), value in
                      zip(self.model.named_parameters(), self.averages, strict=True)})
        return state
