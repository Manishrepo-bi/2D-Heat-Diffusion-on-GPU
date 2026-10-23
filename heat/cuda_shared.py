import numpy as np

from heat.cuda_common import WHERE, run

from numba import cuda, float32, float64

MODE = f"shared-memory tiling, {WHERE}"
_KERNELS = {}


def _build(cfg):
    BX, BY, C, NT = cfg.bx, cfg.by, cfg.coarsen, cfg.threads
    ROWS = BY * C
    TH, TW = ROWS + 2, BX + 2
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
        row0 = cuda.blockIdx.y * ROWS
        col0 = cuda.blockIdx.x * BX

        k = t
        while k < TH * TW:
            li = k // TW
            lj = k % TW
            gi = row0 - 1 + li
            gj = col0 - 1 + lj
            if gi >= 0 and gi < n and gj >= 0 and gj < n:
                tile[li, lj] = cur[gi, gj]
            k += NT
        cuda.syncthreads()

        j = col0 + tx
        d = NP(0.0)
        for r in range(C):
            li = ty + r * BY + 1
            i = row0 + li - 1
            if i > 0 and i < n - 1 and j > 0 and j < n - 1:
                v = NP(0.25) * (tile[li - 1, tx + 1] + tile[li + 1, tx + 1] + tile[li, tx] + tile[li, tx + 2])
                nxt[i, j] = v
                c = abs(v - tile[li, tx + 1])
                if c > d:
                    d = c

        if slot >= 0:
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
    if cfg not in _KERNELS:
        _KERNELS[cfg] = _build(cfg)
    return _KERNELS[cfg]


def solve(grid, eps, max_iter=100_000, check_every=1):
    return run(kernel_for, grid, eps, max_iter, check_every)
