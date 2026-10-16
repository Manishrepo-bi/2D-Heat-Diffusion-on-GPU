"""Compare the global-memory and shared-memory kernels (and the CPU) and save a CSV.

Each run does a fixed number of sweeps (eps = 0) so every backend does the same work:
    time per sweep, effective memory bandwidth, speed-up over the CPU, shared vs global.

    python benchmark.py                              # N = 256 512 1024 2048
    python benchmark.py --sizes 512 1024 --sweeps 1000 --check-every 1 10 50
"""
import argparse
import csv
import datetime
import os
import platform
import sys

from heat.backends import load
from heat.grid import BackendUnavailable, make_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--sweeps", type=int, default=500, help="sweeps per GPU run")
    ap.add_argument("--cpu-sweeps", type=int, default=20, help="sweeps per CPU run (the CPU is slow)")
    ap.add_argument("--check-every", type=int, nargs="+", default=[1, 50])
    ap.add_argument("--repeats", type=int, default=3, help="keep the fastest of this many runs")
    ap.add_argument("--out", default=os.path.join("results", "benchmark.csv"))
    a = ap.parse_args()

    mods = {}
    for name in ("cpu_reference", "cuda_global", "cuda_shared"):
        try:
            mods[name] = load(name)
        except BackendUnavailable as e:
            print(f"{name}: skipped ({e})")
    if any("simulator" in m.MODE for m in mods.values()):
        print("STOP: the CUDA simulator is not a GPU, its timings mean nothing. Unset HEAT_CUDA.")
        return 1

    rows = []
    print(f"{'N':>6} {'backend':<14} {'check':>5} {'ms/sweep':>10} {'GB/s':>8} {'vs CPU':>8}")
    for n in a.sizes:
        grid = make_grid(n, top=100, bottom=0, left=0, right=0)
        cpu_ms = None
        for name, mod in mods.items():
            for k in ([1] if name == "cpu_reference" else a.check_every):
                sweeps = a.cpu_sweeps if name == "cpu_reference" else a.sweeps
                try:
                    mod.solve(grid, 0.0, max_iter=min(sweeps, 5), check_every=k)       # warm-up / compile
                    best = min(mod.solve(grid, 0.0, max_iter=sweeps, check_every=k)[2] for _ in range(a.repeats))
                except BackendUnavailable as e:
                    print(f"{n:>6} {name:<14} skipped: {e}")
                    break
                ms = best * 1000 / sweeps
                gbs = 2 * 8 * n * n / (ms / 1000) / 1e9      # ideal traffic: read + write each cell once
                if name == "cpu_reference":
                    cpu_ms = ms
                speed = cpu_ms / ms if cpu_ms else float("nan")
                rows.append(dict(n=n, backend=name, check_every=k, sweeps=sweeps, ms_per_sweep=round(ms, 5),
                                 effective_gb_s=round(gbs, 2), speedup_vs_cpu=round(speed, 2)))
                print(f"{n:>6} {name:<14} {k:>5} {ms:>10.4f} {gbs:>8.2f} {speed:>8.1f}x")
        g = [r for r in rows if r["n"] == n and r["backend"] == "cuda_global" and r["check_every"] == 1]
        s = [r for r in rows if r["n"] == n and r["backend"] == "cuda_shared" and r["check_every"] == 1]
        if g and s:
            print(f"{'':>6} shared vs global at N={n}: {g[0]['ms_per_sweep'] / s[0]['ms_per_sweep']:.2f}x")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# {datetime.datetime.now():%Y-%m-%d %H:%M}  {platform.node()}  python {platform.python_version()}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
