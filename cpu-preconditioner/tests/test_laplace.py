"""Discrete energy identities and analytic flat-strip convergence."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laplace import assemble_stiffness  # noqa: E402


class LaplaceTests(unittest.TestCase):
    def test_stiffness_and_surface_energy(self) -> None:
        for nx, nz in ((16, 4), (24, 6)):
            x = np.arange(nx) * (2 * np.pi / nx)
            eta = .08 * np.cos(x) + .03 * np.sin(3 * x)
            stiffness = assemble_stiffness(eta, .25, nz)
            np.testing.assert_allclose(stiffness.toarray(), stiffness.T.toarray(), atol=1e-14)
            np.testing.assert_allclose(stiffness @ np.ones((nz + 1) * nx), 0, atol=1e-13)
            interior = stiffness[:-nx, :-nx]
            self.assertTrue(np.isfinite(np.linalg.cholesky(interior.toarray())).all())

            boundary = np.column_stack((np.ones(nx), np.cos(x) + .3 * np.sin(2 * x)))
            solution = np.asarray(spsolve(interior, -stiffness[:-nx, -nx:] @ boundary))
            full = np.vstack((solution, boundary))
            reaction = stiffness[-nx:] @ full
            np.testing.assert_allclose(full[:, 0], 1, atol=1e-13)
            np.testing.assert_allclose(reaction[:, 0], 0, atol=1e-13)
            np.testing.assert_allclose(reaction.sum(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(boundary.T @ reaction, full.T @ (stiffness @ full), atol=1e-12)
            self.assertGreater(boundary[:, 1] @ reaction[:, 1], 0)

    def test_flat_mode_refinement(self) -> None:
        depth = .25
        for mode in (1, 3):
            errors = []
            for nx, nz in ((32, 8), (64, 16), (128, 32)):
                dx = 2 * np.pi / nx
                x = np.arange(nx) * dx
                boundary = np.cos(mode * x)
                stiffness = assemble_stiffness(np.zeros(nx), depth, nz)
                solution = spsolve(stiffness[:-nx, :-nx], -stiffness[:-nx, -nx:] @ boundary)
                flux = (stiffness[-nx:, :-nx] @ solution + stiffness[-nx:, -nx:] @ boundary) / dx
                exact = mode * np.tanh(mode * depth) * boundary
                errors.append(float(np.linalg.norm(flux - exact) / np.linalg.norm(exact)))
            print(f"Flat mode {mode}, relative DNO errors: {errors}")
            self.assertLess(errors[1], .4 * errors[0])
            self.assertLess(errors[2], .4 * errors[1])
            self.assertLess(errors[-1], .002)


if __name__ == "__main__":
    unittest.main()
