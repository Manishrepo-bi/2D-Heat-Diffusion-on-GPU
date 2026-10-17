"""Participant 2: the CUDA host-side driver shared by both kernels.

What stays on the GPU
    Both temperature grids live on the GPU for the whole run. Per convergence check only ONE
    number (8 bytes, the largest change) is copied back to the CPU. The full grid is copied
    once in (two buffers) and once out at the end. `LAST_RUN` reports the bytes moved.

How the convergence check runs in parallel
    Each kernel launch also reduces |new - old| inside every block (shared memory), and thread 0
    of each block folds its block maximum into a global value with an atomic max. The global
    value has two slots: a check writes slot s and block (0, 0) zeroes the other slot, so the
    next check starts from 0 without an extra kernel launch or a host-to-device copy.

Modes (environment variable HEAT_CUDA)
    auto (default) -> real NVIDIA GPU, otherwise BackendUnavailable
    sim            -> Numba's CUDA simulator on the CPU (very slow, correctness only)
"""
import math
import os
from time import perf_counter

import numpy as np

from heat.grid import BackendUnavailable, check_args, check_grid

if os.environ.get("HEAT_CUDA", "auto").lower() == "sim":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"      # must be set before numba is imported

from numba import cuda                               # noqa: E402

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
TPB = 4 if SIM else 16           # threads per block in each direction (16 x 16 = 256 on a GPU)
TPB2 = TPB * TPB
WHERE = "CUDA simulator on CPU (slow, correctness only)" if SIM else "CUDA on GPU"
LAST_RUN = {}


def launch_config(n):
    blocks = (math.ceil(n / TPB), math.ceil(n / TPB))
    return blocks, (TPB, TPB)


def run(kernel, grid, eps, max_iter=100_000, check_every=1):
    """Run Jacobi sweeps with `kernel` until the max change < eps.

    kernel signature: kernel(cur, nxt, dmax, slot) where slot is 0 or 1 on a check sweep
    and -1 on a sweep without a check. Returns (final_grid, iterations, seconds)."""
    grid = check_grid(grid)
    check_args(eps, max_iter, check_every)
    if not cuda.is_available():
        raise BackendUnavailable("no NVIDIA GPU found; set HEAT_CUDA=sim to use the simulator")
    n = grid.shape[0]
    blocks, threads = launch_config(n)

    d_cur = cuda.to_device(grid)                  # copy in once ...
    d_nxt = cuda.to_device(grid)                  # ... boundary values live in both buffers
    d_max = cuda.to_device(np.zeros(2))
    h_max = cuda.pinned_array(2, dtype=np.float64) if not SIM else np.zeros(2)
    h2d = 2 * grid.nbytes + 16
    d2h = 0
    checks = 0

    cuda.synchronize()
    t0 = perf_counter()
    it = 0
    while it < max_iter:
        it += 1
        check = it % check_every == 0
        slot = checks % 2 if check else -1
        kernel[blocks, threads](d_cur, d_nxt, d_max, slot)
        d_cur, d_nxt = d_nxt, d_cur
        if check:
            checks += 1
            d_max.copy_to_host(h_max)              # 16 bytes: the only copy inside the loop
            d2h += 16
            if h_max[slot] < eps:
                break
    cuda.synchronize()                            # launches are asynchronous: wait before timing
    seconds = perf_counter() - t0
    out = d_cur.copy_to_host()                    # copy out once
    d2h += out.nbytes
    LAST_RUN.clear()
    LAST_RUN.update(n=n, iterations=it, checks=checks, blocks=blocks, threads=threads,
                    h2d_bytes=h2d, d2h_bytes=d2h, grid_bytes=grid.nbytes, where=WHERE)
    return out, it, seconds
