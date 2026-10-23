import argparse
import sys

import numpy as np

from heat.backends import BACKENDS, load
from heat.grid import BackendUnavailable, load_grid, make_grid, make_hot_edge, make_uniform, save_grid


def main():
    ap = argparse.ArgumentParser(description="2D heat diffusion with fixed-temperature boundaries")
    ap.add_argument("--backend", choices=BACKENDS, default="cpu_reference")
    ap.add_argument("--n", type=int, default=64, help="grid size N (ignored with --grid)")
    ap.add_argument("--case", choices=["edges", "uniform", "hot-edge"], default="edges",
                    help="boundary case: uniform (3a), edges (3b) or hot-edge (3c)")
    ap.add_argument("--grid", help="initial grid file (.csv or .npy); its outer ring is the boundary")
    ap.add_argument("--boundary", type=float, default=75.0, help="uniform case: boundary temperature")
    ap.add_argument("--hot", type=float, default=100.0, help="hot-edge case: temperature of the top edge")
    ap.add_argument("--top", type=float, default=100.0)
    ap.add_argument("--bottom", type=float, default=0.0)
    ap.add_argument("--left", type=float, default=0.0)
    ap.add_argument("--right", type=float, default=0.0)
    ap.add_argument("--interior", type=float, default=0.0, help="initial temperature of interior cells")
    ap.add_argument("--eps", type=float, default=1e-4, help="stop when max |change| < eps")
    ap.add_argument("--max-iter", type=int, default=1_000_000)
    ap.add_argument("--check-every", type=int, default=1, help="convergence check every k sweeps")
    ap.add_argument("--block", help="CUDA thread block WIDTHxHEIGHT, e.g. 32x8 (default 16x16)")
    ap.add_argument("--coarsen", type=int, help="CUDA cells updated per thread (default 1)")
    ap.add_argument("--dtype", choices=["float64", "float32"], help="CUDA precision (float32: use --eps 1e-3 or larger)")
    ap.add_argument("--out", help="save the converged grid (.csv or .npy)")
    a = ap.parse_args()

    if a.grid:
        grid, desc = load_grid(a.grid), f"from {a.grid}"
    elif a.case == "uniform":
        grid, desc = make_uniform(a.n, a.boundary, a.interior), f"case 3a uniform, boundary = {a.boundary:g}"
    elif a.case == "hot-edge":
        grid, desc = make_hot_edge(a.n, a.hot, a.interior), f"case 3c hot top edge = {a.hot:g}, other edges 0"
    else:
        grid, desc = make_grid(a.n, a.top, a.bottom, a.left, a.right, a.interior), \
            f"case 3b edges top/bottom/left/right = {a.top:g}/{a.bottom:g}/{a.left:g}/{a.right:g}"
    try:
        mod = load(a.backend)
        if a.backend.startswith("cuda") and (a.block or a.coarsen or a.dtype):
            from heat.cuda_common import configure
            configure(block=a.block, coarsen=a.coarsen, dtype=a.dtype)
        out, iters, secs = mod.solve(grid, a.eps, max_iter=a.max_iter, check_every=a.check_every)
    except BackendUnavailable as e:
        print(f"{a.backend} cannot run here: {e}")
        return 1

    n = grid.shape[0]
    print(f"backend      : {a.backend} ({mod.MODE})")
    print(f"grid         : {n} x {n}, {desc}")
    print(f"epsilon      : {a.eps:g}   (checked every {a.check_every} sweep(s))")
    print(f"iterations   : {iters}" + ("   (stopped at --max-iter, NOT converged)" if iters >= a.max_iter else ""))
    print(f"time         : {secs * 1000:.3f} ms")
    if a.backend.startswith("cuda"):
        from heat.cuda_common import LAST_RUN as r
        print(f"CPU->GPU     : {r['h2d_bytes']:,} bytes   GPU->CPU: {r['d2h_bytes']:,} bytes "
              f"(one grid = {r['grid_bytes']:,} bytes)")
        print(f"tuning       : block {r['block']}, {r['coarsen']} cell(s) per thread, {r['dtype']}")
    if n <= 12:
        print("converged grid:")
        with np.printoptions(precision=3, suppress=True, linewidth=160):
            print(out)
    if a.out:
        save_grid(a.out, out)
        print(f"saved        : {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
