import os

import numpy as np


class BackendUnavailable(RuntimeError):
    pass


def make_grid(n, top=100.0, bottom=0.0, left=0.0, right=0.0, interior=0.0):
    if n < 3:
        raise ValueError("n must be at least 3")
    g = np.full((n, n), float(interior))
    g[0, :], g[-1, :] = top, bottom
    g[:, 0], g[:, -1] = left, right
    g[0, 0], g[0, -1] = (top + left) / 2, (top + right) / 2
    g[-1, 0], g[-1, -1] = (bottom + left) / 2, (bottom + right) / 2
    return g


def make_uniform(n, boundary=75.0, interior=0.0):
    return make_grid(n, boundary, boundary, boundary, boundary, interior)


def make_hot_edge(n, hot=100.0, interior=0.0):
    return make_grid(n, top=hot, bottom=0.0, left=0.0, right=0.0, interior=interior)


def load_grid(path):
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
