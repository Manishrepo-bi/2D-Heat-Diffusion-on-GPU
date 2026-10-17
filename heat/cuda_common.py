"""The CUDA host-side driver shared by both kernels.

What stays on the GPU
    Both temperature grids live on the GPU for the whole run. Per convergence check only ONE
    number (8 bytes, the largest change) is copied back to the CPU. The full grid is copied
    once in (two buffers) and once out at the end. `LAST_RUN` reports the bytes moved.

How the convergence check runs in parallel
    Each kernel launch also reduces |new - old| inside every block (shared memory), and thread 0
    of each block folds its block maximum into a global value with an atomic max. The global
    value has two slots: a check writes slot s and block (0, 0) zeroes the other slot, so the
    next check starts from 0 without an extra kernel launch or a host-to-device copy.

Tuning settings (read when a kernel is first used; change them with configure())
    HEAT_BLOCK    thread-block shape, "WIDTHxHEIGHT" (x = columns). Default 16x16 on a GPU, 4x4 in the
                  simulator. The number of threads must be a power of two and at most 1024.
                  A wide block such as 32x8 reads whole 32-element rows, which suits GPU memory.
    HEAT_COARSEN  cells per thread down a column (default 1). A block then covers HEIGHT x COARSEN rows,
                  so there are fewer threads and less index arithmetic per cell.
    HEAT_DTYPE    float64 (default) or float32. float32 halves the memory traffic, but it can only
                  resolve changes of about 1e-5 to 1e-6, so use an epsilon of 1e-3 or larger.

Modes (environment variable HEAT_CUDA)
    auto (default) -> real NVIDIA GPU, otherwise BackendUnavailable
    sim            -> Numba's CUDA simulator on the CPU (very slow, correctness only)
"""
import math
import os
from collections import namedtuple
from time import perf_counter

import numpy as np

from heat.grid import BackendUnavailable, check_args, check_grid

if os.environ.get("HEAT_CUDA", "auto").lower() == "sim":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"      # must be set before numba is imported

from numba import cuda                               # noqa: E402

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
DEFAULT_TPB = 4 if SIM else 16   # default block is DEFAULT_TPB x DEFAULT_TPB (16 x 16 = 256 on a GPU)
WHERE = "CUDA simulator on CPU (slow, correctness only)" if SIM else "CUDA on GPU"
LAST_RUN = {}


class Config(namedtuple("Config", "bx by coarsen dtype")):
    """Kernel tuning settings. Hashable, so a compiled kernel can be cached per setting."""
    __slots__ = ()

    @property
    def threads(self):
        return self.bx * self.by

    @property
    def rows(self):
        """Grid rows covered by one block."""
        return self.by * self.coarsen

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    def label(self):
        return f"{self.bx}x{self.by} c{self.coarsen} {self.dtype}"


def parse_block(text):
    """'32x8' -> (32, 8)."""
    try:
        bx, by = (int(p) for p in str(text).lower().split("x"))
    except ValueError:
        raise ValueError(f"block must look like 32x8, got {text!r}") from None
    return bx, by


def make_config(block=None, coarsen=1, dtype="float64"):
    bx, by = parse_block(block) if block else (DEFAULT_TPB, DEFAULT_TPB)
    dtype = str(dtype).lower()
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}")
    if bx < 1 or by < 1 or bx * by > 1024 or (bx * by) & (bx * by - 1):
        raise ValueError(f"block {bx}x{by}: the thread count must be a power of two and at most 1024")
    if int(coarsen) < 1:
        raise ValueError("coarsen must be at least 1")
    return Config(bx, by, int(coarsen), dtype)


_CONFIG = make_config(os.environ.get("HEAT_BLOCK"), os.environ.get("HEAT_COARSEN", 1),
                      os.environ.get("HEAT_DTYPE", "float64"))


def config():
    return _CONFIG


def configure(block=None, coarsen=None, dtype=None):
    """Change the tuning settings; anything left as None keeps its current value. Returns the Config."""
    global _CONFIG
    c = _CONFIG
    _CONFIG = make_config(block or f"{c.bx}x{c.by}", c.coarsen if coarsen is None else coarsen,
                          dtype or c.dtype)
    return _CONFIG


def launch_config(n, cfg=None):
    cfg = cfg or _CONFIG
    return (math.ceil(n / cfg.bx), math.ceil(n / cfg.rows)), (cfg.bx, cfg.by)


def run(make_kernel, grid, eps, max_iter=100_000, check_every=1):
    """Run Jacobi sweeps until the max change < eps.

    make_kernel(cfg) returns the compiled kernel for a Config. Its signature is
    kernel(cur, nxt, dmax, slot) where slot is 0 or 1 on a check sweep and -1 on a sweep
    without a check. Returns (final_grid, iterations, seconds); the grid is always float64."""
    grid = check_grid(grid)
    check_args(eps, max_iter, check_every)
    if not cuda.is_available():
        raise BackendUnavailable("no NVIDIA GPU found; set HEAT_CUDA=sim to use the simulator")
    cfg = _CONFIG
    kernel = make_kernel(cfg)
    n = grid.shape[0]
    blocks, threads = launch_config(n, cfg)
    dt = np.dtype(cfg.dtype)
    host = grid.astype(dt)                        # float32 halves the bytes moved by every sweep

    d_cur = cuda.to_device(host)                  # copy in once ...
    d_nxt = cuda.to_device(host)                  # ... boundary values live in both buffers
    d_max = cuda.to_device(np.zeros(2, dtype=dt))
    h_max = cuda.pinned_array(2, dtype=dt) if not SIM else np.zeros(2, dtype=dt)
    h2d = 2 * host.nbytes + h_max.nbytes
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
            d_max.copy_to_host(h_max)              # 16 bytes (8 for float32): the only copy inside the loop
            d2h += h_max.nbytes
            if h_max[slot] < eps:
                break
    cuda.synchronize()                            # launches are asynchronous: wait before timing
    seconds = perf_counter() - t0
    out = d_cur.copy_to_host()                    # copy out once
    d2h += out.nbytes
    LAST_RUN.clear()
    LAST_RUN.update(n=n, iterations=it, checks=checks, blocks=blocks, threads=threads,
                    h2d_bytes=h2d, d2h_bytes=d2h, grid_bytes=host.nbytes, where=WHERE,
                    block=f"{cfg.bx}x{cfg.by}", coarsen=cfg.coarsen, dtype=cfg.dtype)
    return out.astype(np.float64, copy=False), it, seconds
