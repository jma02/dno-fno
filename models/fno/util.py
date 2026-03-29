from __future__ import annotations

import torch


class LpLoss:
    """Relative Lp loss from the original FNO codebase."""

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        size_average: bool = True,
        reduction: bool = True,
    ) -> None:
        if d <= 0 or p <= 0:
            raise ValueError("d and p must be positive")
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if not self.reduction:
            return values
        if self.size_average:
            return values.mean()
        return values.sum()

    def abs(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = prediction.shape[0]
        mesh_width = 1.0 / max(1, prediction.shape[1] - 1)
        error = (prediction - target).reshape(batch_size, -1)
        norms = (mesh_width ** (self.d / self.p)) * torch.norm(error, p=self.p, dim=1)
        return self._reduce(norms)

    def rel(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = prediction.shape[0]
        error = (prediction - target).reshape(batch_size, -1)
        baseline = target.reshape(batch_size, -1)
        error_norm = torch.norm(error, p=self.p, dim=1)
        baseline_norm = torch.norm(baseline, p=self.p, dim=1).clamp_min(1e-12)
        return self._reduce(error_norm / baseline_norm)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.rel(prediction, target)


class HsLoss:
    """Sobolev loss that compares fields in Fourier space."""

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        k: int = 1,
        a: list[float] | None = None,
        group: bool = False,
        size_average: bool = True,
        reduction: bool = True,
    ) -> None:
        if d <= 0 or p <= 0:
            raise ValueError("d and p must be positive")
        self.d = d
        self.p = p
        self.k = k
        self.a = a or [1.0] * k
        self.group = group
        self.size_average = size_average
        self.reduction = reduction

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if not self.reduction:
            return values
        if self.size_average:
            return values.mean()
        return values.sum()

    def _complex_lp_norm(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values.abs() ** self.p, dim=1) ** (1.0 / self.p)

    def _relative_lp(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = prediction.shape[0]
        error = prediction.reshape(batch_size, -1) - target.reshape(batch_size, -1)
        baseline = target.reshape(batch_size, -1)
        error_norm = self._complex_lp_norm(error)
        baseline_norm = self._complex_lp_norm(baseline).clamp_min(1e-12)
        return self._reduce(error_norm / baseline_norm)

    def _one_dimensional_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        a: list[float],
    ) -> torch.Tensor:
        nx = prediction.shape[1]
        prediction = prediction.view(prediction.shape[0], nx, -1)
        target = target.view(target.shape[0], nx, -1)

        wave_numbers = torch.cat(
            (
                torch.arange(0, nx // 2),
                torch.arange(-nx // 2, 0),
            )
        )
        wave_numbers = wave_numbers.abs().view(1, nx, 1).to(prediction.device)

        prediction_fft = torch.fft.fftn(prediction, dim=[1])
        target_fft = torch.fft.fftn(target, dim=[1])

        if not self.group:
            weight: torch.Tensor | int = 1
            if self.k >= 1:
                weight = weight + a[0] ** 2 * wave_numbers**2
            if self.k >= 2:
                weight = weight + a[1] ** 2 * wave_numbers**4
            return self._relative_lp(prediction_fft * torch.sqrt(weight), target_fft * torch.sqrt(weight))

        loss = self._relative_lp(prediction_fft, target_fft)
        if self.k >= 1:
            loss = loss + self._relative_lp(
                prediction_fft * (a[0] * torch.sqrt(wave_numbers**2)),
                target_fft * (a[0] * torch.sqrt(wave_numbers**2)),
            )
        if self.k >= 2:
            loss = loss + self._relative_lp(
                prediction_fft * (a[1] * torch.sqrt(wave_numbers**4)),
                target_fft * (a[1] * torch.sqrt(wave_numbers**4)),
            )
        return loss / (self.k + 1)

    def _two_dimensional_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        a: list[float],
    ) -> torch.Tensor:
        nx = prediction.shape[1]
        ny = prediction.shape[2]
        prediction = prediction.view(prediction.shape[0], nx, ny, -1)
        target = target.view(target.shape[0], nx, ny, -1)

        kx = torch.cat((torch.arange(0, nx // 2), torch.arange(-nx // 2, 0))).view(nx, 1).repeat(1, ny)
        ky = torch.cat((torch.arange(0, ny // 2), torch.arange(-ny // 2, 0))).view(1, ny).repeat(nx, 1)
        kx = kx.abs().view(1, nx, ny, 1).to(prediction.device)
        ky = ky.abs().view(1, nx, ny, 1).to(prediction.device)

        prediction_fft = torch.fft.fftn(prediction, dim=[1, 2])
        target_fft = torch.fft.fftn(target, dim=[1, 2])

        if not self.group:
            weight: torch.Tensor | int = 1
            if self.k >= 1:
                weight = weight + a[0] ** 2 * (kx**2 + ky**2)
            if self.k >= 2:
                weight = weight + a[1] ** 2 * (kx**4 + 2 * kx**2 * ky**2 + ky**4)
            return self._relative_lp(prediction_fft * torch.sqrt(weight), target_fft * torch.sqrt(weight))

        loss = self._relative_lp(prediction_fft, target_fft)
        if self.k >= 1:
            first_order = a[0] * torch.sqrt(kx**2 + ky**2)
            loss = loss + self._relative_lp(prediction_fft * first_order, target_fft * first_order)
        if self.k >= 2:
            second_order = a[1] * torch.sqrt(kx**4 + 2 * kx**2 * ky**2 + ky**4)
            loss = loss + self._relative_lp(prediction_fft * second_order, target_fft * second_order)
        return loss / (self.k + 1)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        a: list[float] | None = None,
    ) -> torch.Tensor:
        weights = a or self.a
        if self.d == 1:
            return self._one_dimensional_loss(prediction, target, weights)
        if self.d == 2:
            return self._two_dimensional_loss(prediction, target, weights)
        raise ValueError(f"Unsupported Sobolev dimension d={self.d}")


def count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
