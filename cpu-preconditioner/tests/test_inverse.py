"""Checks of the cached linear inverse and its differentiable training path."""

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
    cached_inverse, diagonal_scaling, flat_inverse, flat_inverse_blocks,
    geometry_features, tensor_inverse, training_pairs,
)


class InverseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.layers = nn.Sequential(nn.Conv2d(4, 8, 1), nn.Tanh(), nn.Conv2d(8, 8, 1),
                                    nn.Tanh(), nn.Conv2d(8, 2, 1))
        self.geometry = geometry_features(.03 * np.cos(np.arange(64) * 2 * np.pi / 64), .25, 16)
        self.blocks = flat_inverse_blocks(.25, 64, 16)
        self.rhs = np.random.default_rng(8).normal(size=(3, 16 * 64))

    def test_linearity_symmetry_and_positive_quadratic_form(self) -> None:
        with torch.inference_mode():
            scaling = diagonal_scaling(self.layers, self.geometry).numpy().ravel().astype(np.float64)
        values = cached_inverse(self.rhs, scaling, self.blocks)
        actual = cached_inverse(2 * self.rhs[0] - 3 * self.rhs[1], scaling, self.blocks)
        np.testing.assert_allclose(actual, 2 * values[0] - 3 * values[1], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(self.rhs[0] @ values[1], self.rhs[1] @ values[0], rtol=1e-12)
        self.assertTrue(np.all(np.sum(self.rhs * values, axis=1) > 0))
        np.testing.assert_array_equal(cached_inverse(np.zeros(1024), scaling, self.blocks), 0)

    def test_tensor_numpy_agreement_and_finite_gradients(self) -> None:
        for dtype, tolerance in ((torch.float32, 3e-5), (torch.float64, 1e-11)):
            self.layers.to(dtype=dtype)
            self.layers.zero_grad()
            scaling = diagonal_scaling(self.layers, self.geometry.to(dtype=dtype))
            rhs = torch.tensor(self.rhs.reshape(3, 1, 16, 64), dtype=dtype)
            blocks = torch.tensor(self.blocks, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)
            value = tensor_inverse(rhs, scaling, blocks)
            expected = cached_inverse(self.rhs, scaling.detach().numpy().ravel().astype(np.float64), self.blocks)
            np.testing.assert_allclose(value.detach().numpy().reshape(3, -1), expected,
                                       rtol=tolerance, atol=tolerance)
            value.square().mean().backward()
            for parameter in self.layers.parameters():
                self.assertIsNotNone(parameter.grad)
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_flat_limit_and_zero_head(self) -> None:
        with torch.inference_mode():
            geometry = geometry_features(np.full(64, .04), .25, 16)
            torch.testing.assert_close(diagonal_scaling(self.layers, geometry), torch.ones(1, 1, 16, 64),
                                       rtol=2e-7, atol=0)
        for parameter in self.layers[-1].parameters():
            nn.init.zeros_(parameter)
        with torch.inference_mode():
            scaling = diagonal_scaling(self.layers, self.geometry).numpy().ravel().astype(np.float64)
        np.testing.assert_array_equal(scaling, 1)
        np.testing.assert_array_equal(cached_inverse(self.rhs, scaling, self.blocks), flat_inverse(self.rhs, self.blocks))

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
        rhs, target = training_pairs(full, nx, nz, 32, np.random.default_rng(20), flat_inverse_blocks(.25, nx, nz))
        np.testing.assert_allclose(np.mean(target ** 2, axis=1), 1, rtol=1e-14)
        residual = (full[:-nx, :-nx] @ target.T).T - rhs
        self.assertLess(np.max(np.linalg.norm(residual, axis=1) / np.linalg.norm(rhs, axis=1)), 1e-12)
        # Half the ordinary draws are boundary forcing; the final half are Krylov directions.
        np.testing.assert_array_equal(rhs[8:16, :-nx], 0)
        self.assertTrue(np.all(np.linalg.norm(rhs[:8, :-nx], axis=1) > 0))
        self.assertTrue(np.all(np.linalg.norm(rhs[16:], axis=1) > 0))


if __name__ == "__main__":
    unittest.main()
