"""Participant 3: shared-memory tiled CUDA kernel.

Each block first copies its TPB x TPB cells plus a one-cell halo (the neighbours just outside
the tile) from global memory into a shared-memory tile. The stencil then reads all five values
from the fast on-chip tile. Every cell is read from global memory about once instead of five
times. The convergence reduction is the same as in the global-memory kernel.
"""
from heat.cuda_common import TPB, TPB2, WHERE, run   # first: it may switch on the simulator

from numba import cuda, float64                       # noqa: E402

MODE = f"shared-memory tiling, {WHERE}"


@cuda.jit
def jacobi_shared(cur, nxt, dmax, slot):
    tile = cuda.shared.array((TPB + 2, TPB + 2), dtype=float64)
    red = cuda.shared.array(TPB2, dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * TPB + tx
    i = cuda.blockIdx.y * TPB + ty
    n = cur.shape[0]
    li = ty + 1                                # position inside the tile (row 0 / col 0 = halo)
    lj = tx + 1

    if i < n and j < n:                        # 1. load my cell, plus the halo at the tile edges
        tile[li, lj] = cur[i, j]
        if tx == 0 and j > 0:
            tile[li, 0] = cur[i, j - 1]
        if tx == TPB - 1 and j < n - 1:
            tile[li, TPB + 1] = cur[i, j + 1]
        if ty == 0 and i > 0:
            tile[0, lj] = cur[i - 1, j]
        if ty == TPB - 1 and i < n - 1:
            tile[TPB + 1, lj] = cur[i + 1, j]
    cuda.syncthreads()                         # 2. the whole tile is loaded before anyone reads it

    d = 0.0
    if i > 0 and i < n - 1 and j > 0 and j < n - 1:          # 3. stencil from shared memory
        v = 0.25 * (tile[li - 1, lj] + tile[li + 1, lj] + tile[li, lj - 1] + tile[li, lj + 1])
        nxt[i, j] = v
        d = abs(v - tile[li, lj])

    if slot >= 0:                              # 4. parallel convergence check (as in cuda_global)
        t = ty * TPB + tx
        red[t] = d
        cuda.syncthreads()
        s = TPB2 // 2
        while s > 0:
            if t < s and red[t + s] > red[t]:
                red[t] = red[t + s]
            cuda.syncthreads()
            s //= 2
        if t == 0:
            cuda.atomic.max(dmax, slot, red[0])
            if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
                dmax[1 - slot] = 0.0


def solve(grid, eps, max_iter=100_000, check_every=1):
    """Return (final_grid, iterations, seconds)."""
    return run(jacobi_shared, grid, eps, max_iter, check_every)
