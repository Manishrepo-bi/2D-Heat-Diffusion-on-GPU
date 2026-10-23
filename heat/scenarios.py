import numpy as np

from heat.grid import make_grid, make_hot_edge, make_uniform


def sweep_6x6(n):
    grid = make_grid(6, top=100, bottom=100, left=0, right=0)
    def check(T):
        want = np.zeros((4, 4)); want[0, :] = want[3, :] = 25.0
        return float(np.max(np.abs(T[1:5, 1:5] - want))), \
            f"rows 1 and 4 = {T[1, 1]:.2f}, rows 2 and 3 = {T[2, 1]:.2f}"
    return grid, check


def case_3a_uniform(n):
    grid = make_uniform(n, boundary=75.0, interior=0.0)
    def check(T):
        e = float(np.max(np.abs(T - 75.0)))
        return e, f"max |T - 75| = {e:.2e}"
    return grid, check


def case_3b_edges(n):
    grid = make_grid(n, top=100, bottom=40, left=20, right=0)
    def check(T):
        c = n // 2
        return abs(T[c, c] - 40.0), f"centre = {T[c, c]:.6f} (exact 40)"
    return grid, check


def case_3b_ramp(n):
    exact = np.tile(100.0 * (1 - np.arange(n) / (n - 1)), (n, 1))
    grid = exact.copy()
    grid[1:-1, 1:-1] = 0.0
    def check(T):
        e = float(np.max(np.abs(T - exact)))
        return e, f"max |T - ramp| = {e:.2e}"
    return grid, check


def case_3c_hot_edge(n):
    grid = make_hot_edge(n, hot=100.0)
    def check(T):
        c = n // 2
        return abs(T[c, c] - 25.0), f"centre = {T[c, c]:.6f} (exact 25)"
    return grid, check


PROFILES = {
    "sim": {"n": 5, "cases": {
        "sweep_6x6": (sweep_6x6, 0.0, 1e-12, 1),
        "case_3a_uniform": (case_3a_uniform, 1e-6, 1e-4, 100_000),
        "case_3b_edges": (case_3b_edges, 1e-6, 1e-4, 100_000),
        "case_3b_ramp": (case_3b_ramp, 1e-6, 1e-4, 100_000),
        "case_3c_hot_edge": (case_3c_hot_edge, 1e-6, 1e-4, 100_000)}},
    "quick": {"n": 9, "cases": {
        "sweep_6x6": (sweep_6x6, 0.0, 1e-12, 1),
        "case_3a_uniform": (case_3a_uniform, 1e-8, 1e-5, 100_000),
        "case_3b_edges": (case_3b_edges, 1e-8, 1e-5, 100_000),
        "case_3b_ramp": (case_3b_ramp, 1e-8, 1e-5, 100_000),
        "case_3c_hot_edge": (case_3c_hot_edge, 1e-8, 1e-5, 100_000)}},
    "full": {"n": 33, "cases": {
        "sweep_6x6": (sweep_6x6, 0.0, 1e-12, 1),
        "case_3a_uniform": (case_3a_uniform, 1e-11, 1e-7, 200_000),
        "case_3b_edges": (case_3b_edges, 1e-11, 1e-7, 200_000),
        "case_3b_ramp": (case_3b_ramp, 1e-11, 1e-7, 200_000),
        "case_3c_hot_edge": (case_3c_hot_edge, 1e-11, 1e-7, 200_000)}},
}


def cases(profile="quick"):
    p = PROFILES[profile]
    for name, (build, eps, tol, max_iter) in p["cases"].items():
        grid, check = build(p["n"])
        yield name, grid, check, eps, tol, max_iter
