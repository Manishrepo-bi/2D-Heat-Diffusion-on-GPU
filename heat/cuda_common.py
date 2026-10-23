import math
import os
from collections import namedtuple
from time import perf_counter

import numpy as np

from heat.grid import BackendUnavailable, check_args, check_grid

if os.environ.get("HEAT_CUDA", "auto").lower() == "sim":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

from numba import cuda

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
DEFAULT_TPB = 4 if SIM else 16
WHERE = "CUDA simulator on CPU (slow, correctness only)" if SIM else "CUDA on GPU"
LAST_RUN = {}


class Config(namedtuple("Config", "bx by coarsen dtype")):
    __slots__ = ()

    @property
    def threads(self):
        return self.bx * self.by

    @property
    def rows(self):
        return self.by * self.coarsen

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    def label(self):
        return f"{self.bx}x{self.by} c{self.coarsen} {self.dtype}"


def parse_block(text):
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
    global _CONFIG
    c = _CONFIG
    _CONFIG = make_config(block or f"{c.bx}x{c.by}", c.coarsen if coarsen is None else coarsen,
                          dtype or c.dtype)
    return _CONFIG


def launch_config(n, cfg=None):
    cfg = cfg or _CONFIG
    return (math.ceil(n / cfg.bx), math.ceil(n / cfg.rows)), (cfg.bx, cfg.by)


def run(make_kernel, grid, eps, max_iter=100_000, check_every=1):
    grid = check_grid(grid)
    check_args(eps, max_iter, check_every)
    if not cuda.is_available():
        raise BackendUnavailable("no NVIDIA GPU found; set HEAT_CUDA=sim to use the simulator")
    cfg = _CONFIG
    kernel = make_kernel(cfg)
    n = grid.shape[0]
    blocks, threads = launch_config(n, cfg)
    dt = np.dtype(cfg.dtype)
    host = grid.astype(dt)

    d_cur = cuda.to_device(host)
    d_nxt = cuda.to_device(host)
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
            d_max.copy_to_host(h_max)
            d2h += h_max.nbytes
            if h_max[slot] < eps:
                break
    cuda.synchronize()
    seconds = perf_counter() - t0
    out = d_cur.copy_to_host()
    d2h += out.nbytes
    LAST_RUN.clear()
    LAST_RUN.update(n=n, iterations=it, checks=checks, blocks=blocks, threads=threads,
                    h2d_bytes=h2d, d2h_bytes=d2h, grid_bytes=host.nbytes, where=WHERE,
                    block=f"{cfg.bx}x{cfg.by}", coarsen=cfg.coarsen, dtype=cfg.dtype)
    return out.astype(np.float64, copy=False), it, seconds
