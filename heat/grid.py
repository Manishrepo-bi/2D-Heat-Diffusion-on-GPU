"""Grid set-up, input/output and validation shared by every backend.

The problem
    An N x N grid. The outer ring of cells holds the boundary conditions: fixed temperatures
    that are never updated. Only interior cells (1 <= i, j <= N-2) are updated:

        T_new[i, j] = (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1]) / 4

    Boundary cases supported
        3a  uniform  : every boundary cell at one temperature            make_uniform()
        3b  edges    : one temperature per edge (or any value per cell)  make_grid() / load_grid()
        3c  hot edge : one hot edge (top), the other three at 0          make_hot_edge()
    Corner cells are never read by the five-point stencil, so their value does not matter.

Every backend offers
    solve(grid, eps, max_iter=100_000, check_every=1) -> (final_grid, iterations, seconds)
"""
import os

import numpy as np


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run on this machine (no GPU, library missing)."""


def make_grid(n, top=100.0, bottom=0.0, left=0.0, right=0.0, interior=0.0):
    """Grid with one fixed temperature per edge. Corners get the average of their two edges
    (only for display: the stencil never reads them)."""
    if n < 3:
        raise ValueError("n must be at least 3")
    g = np.full((n, n), float(interior))
    g[0, :], g[-1, :] = top, bottom
    g[:, 0], g[:, -1] = left, right
    g[0, 0], g[0, -1] = (top + left) / 2, (top + right) / 2
    g[-1, 0], g[-1, -1] = (bottom + left) / 2, (bottom + right) / 2
    return g


def make_uniform(n, boundary=75.0, interior=0.0):
    """Case 3a: every boundary cell at the same temperature."""
    return make_grid(n, boundary, boundary, boundary, boundary, interior)


def make_hot_edge(n, hot=100.0, interior=0.0):
    """Case 3c: one hot edge (the top) and the other three edges at 0."""
    return make_grid(n, top=hot, bottom=0.0, left=0.0, right=0.0, interior=interior)


def load_grid(path):
    """Read an initial grid (boundary values on the outer ring) from .csv or .npy."""
    path = str(path)
    g = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    return check_grid(g)


def save_grid(path, grid):
    path = str(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if path.endswith(".npy"):
        np.save(path, grid)
    else:
        np.savetxt(path, grid, delimiter=",", fmt="%.10g")


def check_grid(grid):
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] < 3:
        raise ValueError(f"grid must be square and at least 3 x 3, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid contains NaN or infinite values")
    return grid


def check_args(eps, max_iter, check_every):
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if max_iter < 1 or check_every < 1:
        raise ValueError("max_iter and check_every must be >= 1")
