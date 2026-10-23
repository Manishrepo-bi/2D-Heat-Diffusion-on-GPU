import numpy as np

from heat.cuda_common import WHERE, run

from numba import cuda, float32, float64

MODE = f"global memory, {WHERE}"
_KERNELS = {}


def _build(cfg):
    BX, BY, C, NT = cfg.bx, cfg.by, cfg.coarsen, cfg.threads
    FT = float32 if cfg.dtype == "float32" else float64
    NP = np.float32 if cfg.dtype == "float32" else np.float64

    @cuda.jit
    def jacobi_global(cur, nxt, dmax, slot):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        j = cuda.blockIdx.x * BX + tx
        i0 = cuda.blockIdx.y * (BY * C) + ty
        n = cur.shape[0]
        red = cuda.shared.array(NT, dtype=FT)

        d = NP(0.0)
        for r in range(C):
            i = i0 + r * BY
            if i > 0 and i < n - 1 and j > 0 and j < n - 1:
                v = NP(0.25) * (cur[i - 1, j] + cur[i + 1, j] + cur[i, j - 1] + cur[i, j + 1])
                nxt[i, j] = v
                c = abs(v - cur[i, j])
                if c > d:
                    d = c

        if slot >= 0:
            t = ty * BX + tx
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

    return jacobi_global


def kernel_for(cfg):
    if cfg not in _KERNELS:
        _KERNELS[cfg] = _build(cfg)
    return _KERNELS[cfg]


def solve(grid, eps, max_iter=100_000, check_every=1):
    return run(kernel_for, grid, eps, max_iter, check_every)
