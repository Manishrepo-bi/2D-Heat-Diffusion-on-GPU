"""Participant 2: straightforward global-memory CUDA kernel.

One thread per cell. Each interior thread reads its four neighbours and itself directly from
global memory (5 reads, 1 write per cell). The same launch reduces |new - old| per block in
shared memory for the parallel convergence check (see heat/cuda_common.py).
"""
from heat.cuda_common import TPB, TPB2, WHERE, run   # first: it may switch on the simulator

from numba import cuda, float64                       # noqa: E402

MODE = f"global memory, {WHERE}"


@cuda.jit
def jacobi_global(cur, nxt, dmax, slot):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * TPB + tx            # x -> column: neighbouring threads touch neighbouring memory
    i = cuda.blockIdx.y * TPB + ty
    n = cur.shape[0]
    red = cuda.shared.array(TPB2, dtype=float64)

    d = 0.0
    if i > 0 and i < n - 1 and j > 0 and j < n - 1:          # interior cells only
        v = 0.25 * (cur[i - 1, j] + cur[i + 1, j] + cur[i, j - 1] + cur[i, j + 1])
        nxt[i, j] = v
        d = abs(v - cur[i, j])

    if slot >= 0:                              # same value for every thread: safe around syncthreads
        t = ty * TPB + tx
        red[t] = d
        cuda.syncthreads()
        s = TPB2 // 2
        while s > 0:                           # tree reduction: halve the active threads each step
            if t < s and red[t + s] > red[t]:
                red[t] = red[t + s]
            cuda.syncthreads()
            s //= 2
        if t == 0:
            cuda.atomic.max(dmax, slot, red[0])
            if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
                dmax[1 - slot] = 0.0           # prepare the other slot for the next check


def solve(grid, eps, max_iter=100_000, check_every=1):
    """Return (final_grid, iterations, seconds)."""
    return run(jacobi_global, grid, eps, max_iter, check_every)
