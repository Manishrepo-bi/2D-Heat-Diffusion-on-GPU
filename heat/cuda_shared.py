"""Shared-memory tiled CUDA kernel.

Each block first copies its tile (BLOCK WIDTH x BLOCK HEIGHT x COARSEN cells) plus a one-cell halo
(the neighbours just outside the tile) from global memory into a shared-memory tile. The stencil
then reads all five values from the fast on-chip tile. Every cell is read from global memory
about once instead of five times. All threads of the block take part in the load, in a strided loop
that works for any block shape. The convergence reduction is the same as in the global-memory
kernel. Block shape, cells per thread and float precision come from the tuning settings in
heat/cuda_common.py (HEAT_BLOCK, HEAT_COARSEN, HEAT_DTYPE).
"""
import numpy as np

from heat.cuda_common import WHERE, run   # first: it may switch on the simulator

from numba import cuda, float32, float64   # noqa: E402

MODE = f"shared-memory tiling, {WHERE}"
_KERNELS = {}


def _build(cfg):
    BX, BY, C, NT = cfg.bx, cfg.by, cfg.coarsen, cfg.threads
    ROWS = BY * C                                  # grid rows covered by one block
    TH, TW = ROWS + 2, BX + 2                      # tile size including the halo
    FT = float32 if cfg.dtype == "float32" else float64
    NP = np.float32 if cfg.dtype == "float32" else np.float64

    @cuda.jit
    def jacobi_shared(cur, nxt, dmax, slot):
        tile = cuda.shared.array((TH, TW), dtype=FT)
        red = cuda.shared.array(NT, dtype=FT)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        t = ty * BX + tx
        n = cur.shape[0]
        row0 = cuda.blockIdx.y * ROWS              # first grid row of the tile
        col0 = cuda.blockIdx.x * BX

        k = t                                      # 1. all threads load the tile and its halo, TW cells per row
        while k < TH * TW:
            li = k // TW
            lj = k % TW
            gi = row0 - 1 + li
            gj = col0 - 1 + lj
            if gi >= 0 and gi < n and gj >= 0 and gj < n:
                tile[li, lj] = cur[gi, gj]
            k += NT
        cuda.syncthreads()                         # 2. the whole tile is loaded before anyone reads it

        j = col0 + tx
        d = NP(0.0)
        for r in range(C):                         # 3. stencil from shared memory, C cells per thread
            li = ty + r * BY + 1
            i = row0 + li - 1
            if i > 0 and i < n - 1 and j > 0 and j < n - 1:
                v = NP(0.25) * (tile[li - 1, tx + 1] + tile[li + 1, tx + 1] + tile[li, tx] + tile[li, tx + 2])
                nxt[i, j] = v
                c = abs(v - tile[li, tx + 1])
                if c > d:
                    d = c

        if slot >= 0:                              # 4. parallel convergence check (as in cuda_global)
            red[t] = d
            cuda.syncthreads()
            s = NT // 2
            while s > 0:
                if t < s and red[t + s] > red[t]:
                    red[t] = red[t + s]
                cuda.syncthreads()
                s //= 2
            if t == 0:
                cuda.atomic.max(dmax, slot, red[0])
                if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0:
                    dmax[1 - slot] = 0.0

    return jacobi_shared


def kernel_for(cfg):
    """Compiled kernel for a tuning Config (compiled once per setting)."""
    if cfg not in _KERNELS:
        _KERNELS[cfg] = _build(cfg)
    return _KERNELS[cfg]


def solve(grid, eps, max_iter=100_000, check_every=1):
    """Return (final_grid, iterations, seconds)."""
    return run(kernel_for, grid, eps, max_iter, check_every)
