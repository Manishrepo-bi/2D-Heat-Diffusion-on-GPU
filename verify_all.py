import argparse
import datetime
import os
import platform
import sys

import numpy as np

from heat.backends import BACKENDS, load
from heat.grid import BackendUnavailable
from heat.scenarios import PROFILES, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--backend", choices=BACKENDS + ["all"], default="all")
    ap.add_argument("--block", help="CUDA thread block WIDTHxHEIGHT, e.g. 32x8")
    ap.add_argument("--coarsen", type=int, help="CUDA cells updated per thread")
    ap.add_argument("--dtype", choices=["float64", "float32"], help="CUDA precision; float32 uses a looser epsilon and tolerance")
    a = ap.parse_args()

    print("=" * 80)
    print(f"2D heat diffusion verification   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"computer: {platform.node()}   python {platform.python_version()}   profile: {a.profile}"
          f"   HEAT_CUDA={os.environ.get('HEAT_CUDA', 'auto')}")
    print("=" * 80)
    passed = failed = skipped = 0
    refs = {}
    for name in (BACKENDS if a.backend == "all" else [a.backend]):
        try:
            mod = load(name)
        except BackendUnavailable as e:
            print(f"\n[{name}]  SKIP: {e}")
            skipped += 1
            continue
        profile = "sim" if "simulator" in mod.MODE else a.profile
        note = "  -> small 'sim' profile" if profile != a.profile else ""
        loose = False
        if name.startswith("cuda"):
            from heat.cuda_common import configure
            cfg = configure(block=a.block, coarsen=a.coarsen, dtype=a.dtype)
            loose = cfg.dtype == "float32"
            note += f"\n  tuning: block {cfg.bx}x{cfg.by}, {cfg.coarsen} cell(s) per thread, {cfg.dtype}"
        print(f"\n[{name}]  {mod.MODE}{note}")
        for case, grid, check, eps, tol, max_iter in cases(profile):
            if loose and eps > 0:
                eps, tol = max(eps, 1e-3), max(tol, 1e-2)
            try:
                out, iters, secs = mod.solve(grid, eps, max_iter=max_iter)
            except BackendUnavailable as e:
                print(f"  {case:<17} SKIP  {e}")
                skipped += 1
                continue
            except Exception as e:
                print(f"  {case:<17} FAIL  crashed: {type(e).__name__}: {e}")
                failed += 1
                continue
            err, msg = check(out)
            same_edges = np.array_equal(out[0], grid[0]) and np.array_equal(out[-1], grid[-1]) \
                and np.array_equal(out[:, 0], grid[:, 0]) and np.array_equal(out[:, -1], grid[:, -1])
            ok = err <= tol and same_edges
            passed += ok
            failed += not ok
            extra = "" if same_edges else "  BOUNDARY CHANGED"
            print(f"  {case:<17} {'PASS' if ok else 'FAIL'}  {iters:>6} iterations  {secs * 1000:>10.2f} ms   {msg}{extra}")
    print("\n" + "-" * 80)
    verdict = "ALL CHECKS PASSED" if failed == 0 and passed else ("FAILED" if failed else "NOTHING RAN")
    print(f"RESULT: {verdict}   ({passed} passed, {failed} failed, {skipped} skipped)")
    print("-" * 80)
    return 0 if failed == 0 and passed else 1


if __name__ == "__main__":
    sys.exit(main())
