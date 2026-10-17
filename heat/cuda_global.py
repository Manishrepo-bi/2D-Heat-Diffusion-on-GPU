"""Straightforward global-memory CUDA kernel.

One thread per cell (or per COARSEN cells down a column). Each interior thread reads its four
neighbours and itself directly from global memory (5 reads, 1 write per cell). The same launch
reduces |new - old| per block in shared memory for the parallel convergence check (see
heat/cuda_common.py). Block shape, cells per thread and float precision come from the tuning
settings in heat/cuda_common.py (HEAT_BLOCK, HEAT_COARSEN, HEAT_DTYPE).
"""
import numpy as np

from heat.cuda_common import WHERE, run   # first: it may switch on the simulator

from numba import cuda, float32, float64   # noqa: E402

MODE = f"global memory, {WHERE}"
_KERNELS = {}


def _build(cfg):
    BX, BY, C, NT = cfg.bx, cfg.by, cfg.coarsen, cfg.threads
    FT = float32 if cfg.dtype == "float32" else float64      # numba type of the shared array
    NP = np.float32 if cfg.dtype == "float32" else np.float64  # same type for constants

    @cuda.jit
    def jacobi_global(cur, nxt, dmax, slot):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        j = cuda.blockIdx.x * BX + tx            # x -> column: neighbouring threads touch neighbouring memory
        i0 = cuda.blockIdx.y * (BY * C) + ty     # first of this thread's C rows (BY rows apart)
        n = cur.shape[0]
        red = cuda.shared.array(NT, dtype=FT)

        d = NP(0.0)
        for r in range(C):                       # thread coarsening: C cells per thread
            i = i0 + r * BY
            if i > 0 and i < n - 1 and j > 0 and j < n - 1:      # interior cells only
                v = NP(0.25) * (cur[i - 1, j] + cur[i + 1, j] + cur[i, j - 1] + cur[i, j + 1])
                nxt[i, j] = v
                c = abs(v - cur[i, j])
                if c > d:
                    d = c

        if slot >= 0:                              # same value for every thread: safe around syncthreads
            t = ty * BX + tx
            red[t] = d
            cuda.syncthreads()
            s = NT // 2
            while s > 0:                           # tree reduction: halve the active threads each step
                if t < s and red[t + s] > red[t]:
                    red[t] = red[t + s]
                cuda.syncthreads()
                s //= 2
            if t == 0:
                cuda.atomic.max(dmax, slot, red[0])
                if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
                    dmax[1 - slot] = 0.0           # prepare the other slot for the next check

    return jacobi_global


def kernel_for(cfg):
    """Compiled kernel for a tuning Config (compiled once per setting)."""
    if cfg not in _KERNELS:
        _KERNELS[cfg] = _build(cfg)
    return _KERNELS[cfg]


def solve(grid, eps, max_iter=100_000, check_every=1):
    """Return (final_grid, iterations, seconds)."""
    return run(kernel_for, grid, eps, max_iter, check_every)
