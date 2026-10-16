"""Problems with a known exact answer, one or more for each boundary case (3a, 3b, 3c).

Profiles: sim (5 x 5, for the slow CUDA simulator), quick (9 x 9, default), full (33 x 33).
"""
import numpy as np

from heat.grid import make_grid, make_hot_edge, make_uniform


def sweep_6x6(n):
    """One sweep by hand (docs/heat_diffusion_sweep.xlsx): top and bottom 100, sides 0.
    After exactly one sweep rows 1 and 4 are 25 and rows 2 and 3 are still 0."""
    grid = make_grid(6, top=100, bottom=100, left=0, right=0)
    def check(T):
        want = np.zeros((4, 4)); want[0, :] = want[3, :] = 25.0
        return float(np.max(np.abs(T[1:5, 1:5] - want))), \
            f"rows 1 and 4 = {T[1, 1]:.2f}, rows 2 and 3 = {T[2, 1]:.2f}"
    return grid, check


def case_3a_uniform(n):
    """Case 3a: every boundary cell at 75. Nothing is hotter or colder than the boundary, so the
    whole plate must end at 75."""
    grid = make_uniform(n, boundary=75.0, interior=0.0)
    def check(T):
        e = float(np.max(np.abs(T - 75.0)))
        return e, f"max |T - 75| = {e:.2e}"
    return grid, check


def case_3b_edges(n):
    """Case 3b: top 100, bottom 40, left 20, right 0. Each edge contributes a quarter of its
    temperature at the centre (rotate the plate four times and add), so the centre is
    (100 + 40 + 20 + 0) / 4 = 40 exactly (n odd)."""
    grid = make_grid(n, top=100, bottom=40, left=20, right=0)
    def check(T):
        c = n // 2
        return abs(T[c, c] - 40.0), f"centre = {T[c, c]:.6f} (exact 40)"
    return grid, check


def case_3b_ramp(n):
    """Case 3b with a different value on every boundary cell: 100 * (1 - j/(n-1)), i.e. 100 on the
    left, 0 on the right and a straight line along top and bottom. A linear function equals the
    average of its four neighbours, so the exact answer is the same straight line in every row."""
    exact = np.tile(100.0 * (1 - np.arange(n) / (n - 1)), (n, 1))
    grid = exact.copy()
    grid[1:-1, 1:-1] = 0.0
    def check(T):
        e = float(np.max(np.abs(T - exact)))
        return e, f"max |T - ramp| = {e:.2e}"
    return grid, check


def case_3c_hot_edge(n):
    """Case 3c: only the top edge is hot (100), the other three are 0. Rotating this plate four
    times and adding gives a plate that is 100 everywhere, so the centre is exactly 100 / 4 = 25."""
    grid = make_hot_edge(n, hot=100.0)
    def check(T):
        c = n // 2
        return abs(T[c, c] - 25.0), f"centre = {T[c, c]:.6f} (exact 25)"
    return grid, check


# name -> (builder, eps, allowed error, max_iter)
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
    """Yield (name, grid, check, eps, tol, max_iter)."""
    p = PROFILES[profile]
    for name, (build, eps, tol, max_iter) in p["cases"].items():
        grid, check = build(p["n"])
        yield name, grid, check, eps, tol, max_iter
