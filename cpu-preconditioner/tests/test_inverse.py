"""Independent checks of the learned correction, without a Krylov solver."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse.linalg import splu
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laplace import assemble_stiffness  # noqa: E402
from run import (  # noqa: E402
    correction, flat_inverse, flat_inverse_blocks, geometry_features, neural_inverse,
    training_pairs,
)


class InverseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 8, 3), *(nn.Conv2d(8, 8, 3) for _ in range(3)),
            *(nn.Conv2d(16, 8, 3) for _ in range(3)), nn.Conv2d(8, 1, 1),
        ])
        self.geometry = geometry_features(.03 * np.cos(np.arange(64) * 2 * np.pi / 64), .25, 16)
        self.initial = torch.randn(3, 1, 16, 64)

    def test_zero_and_positive_scaling(self) -> None:
        with torch.inference_mode():
            zero = correction(self.layers, torch.zeros_like(self.initial), self.geometry)
            self.assertEqual(torch.count_nonzero(zero).item(), 0)
            value = correction(self.layers, self.initial, self.geometry)
            for scale in (.01, 100.):
                scaled = correction(self.layers, scale * self.initial, self.geometry)
                torch.testing.assert_close(scaled / scale, value, rtol=2e-5, atol=2e-6)

    def test_batch_and_finite_gradients(self) -> None:
        value = correction(self.layers, self.initial, self.geometry)
        individual = torch.cat([
            correction(self.layers, initial[None], self.geometry) for initial in self.initial
        ])
        torch.testing.assert_close(value, individual, rtol=2e-5, atol=2e-6)
        value.square().mean().backward()
        for parameter in self.layers.parameters():
            self.assertIsNotNone(parameter.grad)
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_zero_head_preserves_flat_inverse(self) -> None:
        for parameter in self.layers[-1].parameters():
            nn.init.zeros_(parameter)
        with torch.inference_mode():
            torch.testing.assert_close(
                correction(self.layers, self.initial, self.geometry), self.initial,
                rtol=2e-7, atol=2e-7,
            )
        matrix = assemble_stiffness(np.zeros(64), .25, 16)[:-64, :-64].tocsc()
        factor = splu(matrix)
        rhs = np.random.default_rng(8).normal(size=matrix.shape[0])
        expected = factor.solve(rhs)
        actual = neural_inverse(rhs, self.layers, self.geometry, flat_inverse_blocks(.25, 64, 16))
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-7)

    def test_flat_inverse_matches_fem_and_periodicity(self) -> None:
        rng = np.random.default_rng(19)
        for nx, nz in ((24, 8), (64, 16)):
            matrix = assemble_stiffness(np.full(nx, .04), .25, nz)[:-nx, :-nx].tocsc()
            blocks = flat_inverse_blocks(.29, nx, nz)
            rhs = rng.normal(size=(3, nx * nz))
            expected = splu(matrix).solve(rhs.T).T
            actual = flat_inverse(rhs, blocks)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(flat_inverse(rhs[0], blocks), actual[0], rtol=1e-12, atol=1e-12)
            shifted = np.roll(rhs.reshape(3, nz, nx), 3, axis=-1).reshape(3, -1)
            np.testing.assert_allclose(
                flat_inverse(shifted, blocks).reshape(3, nz, nx),
                np.roll(actual.reshape(3, nz, nx), 3, axis=-1), rtol=1e-10, atol=1e-10,
            )

    def test_mixed_training_labels_solve_the_same_system(self) -> None:
        nx, nz = 24, 8
        full = assemble_stiffness(.04 * np.cos(np.arange(nx) * 2 * np.pi / nx), .25, nz)
        rhs, target = training_pairs(full, nx, nz, 32, np.random.default_rng(20))
        np.testing.assert_allclose(np.mean(target ** 2, axis=1), 1, rtol=1e-14)
        residual = (full[:-nx, :-nx] @ target.T).T - rhs
        self.assertLess(np.max(np.linalg.norm(residual, axis=1) / np.linalg.norm(rhs, axis=1)), 1e-12)
        # Only the boundary-forced half should have identically zero deep forcing.
        np.testing.assert_array_equal(rhs[16:, :-nx], 0)
        self.assertTrue(np.all(np.linalg.norm(rhs[:16, :-nx], axis=1) > 0))


if __name__ == "__main__":
    unittest.main()
