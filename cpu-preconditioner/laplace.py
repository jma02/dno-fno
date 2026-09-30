"""Periodic-x P1 Laplace stiffness on a vertically mapped triangular grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def assemble_stiffness(eta: NDArray[np.float64], depth: float, nz: int,
                       length: float = 2 * np.pi) -> csr_matrix:
    """Return full stiffness in (z, x) order, with the nx surface nodes last.

    There are nz vertical cells and nx periodic horizontal cells. The bottom
    has natural homogeneous Neumann data; eliminate only the final nx nodes
    to impose surface Dirichlet data. Surface reaction divided by length/nx
    is the mass-lumped, unnormalized surface flux.
    """
    nx = len(eta)
    dx = length / nx
    z = -depth + np.arange(nz + 1)[:, None] / nz * (depth + eta[None, :])
    lower = np.arange(nz * nx).reshape(nz, nx)
    right = np.roll(lower, -1, axis=1)
    corners = np.stack((lower, right, right + nx, lower + nx), axis=-1).reshape(-1, 4)
    triangles = corners[:, np.array(((0, 1, 2), (0, 2, 3)))].reshape(-1, 3)

    # Each cell uses local, unwrapped x coordinates, including at the seam.
    triangle_x = np.tile(((0., dx, dx), (0., dx, 0.)), (nz * nx, 1))
    triangle_z = z.ravel()[triangles]
    b = triangle_z[:, [1, 2, 0]] - triangle_z[:, [2, 0, 1]]
    c = triangle_x[:, [2, 0, 1]] - triangle_x[:, [1, 2, 0]]
    twice_area = np.sum(triangle_x * b, axis=1)
    if not np.isfinite(twice_area).all() or np.any(twice_area <= 0):
        raise ValueError("Mapped mesh contains nonfinite or nonpositive triangle areas")
    local = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (
        2 * twice_area[:, None, None])
    rows = np.broadcast_to(triangles[:, :, None], local.shape).ravel()
    columns = np.broadcast_to(triangles[:, None, :], local.shape).ravel()
    size = (nz + 1) * nx
    return csr_matrix((local.ravel(), (rows, columns)), shape=(size, size))
