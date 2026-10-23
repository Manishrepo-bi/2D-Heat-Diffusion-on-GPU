from time import perf_counter

import numpy as np

from heat.grid import check_args, check_grid

MODE = "NumPy on CPU (reference)"


def solve(grid, eps, max_iter=100_000, check_every=1):
    cur = check_grid(grid).copy()
    check_args(eps, max_iter, check_every)
    nxt = cur.copy()
    t0 = perf_counter()
    it = 0
    while it < max_iter:
        nxt[1:-1, 1:-1] = 0.25 * (cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:])
        it += 1
        check = it % check_every == 0
        if check:
            diff = float(np.max(np.abs(nxt[1:-1, 1:-1] - cur[1:-1, 1:-1])))
        cur, nxt = nxt, cur
        if check and diff < eps:
            break
    return cur, it, perf_counter() - t0
