import argparse
import csv


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for r in rows:
        r["n"], r["check_every"] = int(r["n"]), int(r["check_every"])
        r["ms"] = float(r["ms_per_sweep"])
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def one(rows, **kw):
    found = pick(rows, **kw)
    return found[0]["ms"] if found else None


def line(*cols):
    print("  ".join(f"{c:>10}" if i else f"{c:<22}" for i, c in enumerate(cols)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results/benchmark_t4.csv")
    ap.add_argument("--tuning", default="results/tuning_t4.csv")
    a = ap.parse_args()
    bench, tune = read(a.bench), read(a.tuning)

    print("1. GPU against CPU (ms per sweep, speed-up over the CPU)")
    line("N", "CPU", "global", "x CPU", "shared", "x CPU")
    for n in sorted({r["n"] for r in bench}):
        cpu, g, s = (one(bench, n=n, backend=b, check_every=1) for b in ("cpu_reference", "cuda_global", "cuda_shared"))
        line(n, f"{cpu:.3f}", f"{g:.3f}", f"{cpu / g:.1f}", f"{s:.3f}", f"{cpu / s:.1f}")

    print("\n2. Shared memory against global memory (above 1 = shared is faster)")
    line("N", "check 1", "check 50")
    for n in sorted({r["n"] for r in bench if r["backend"] != "cpu_reference"}):
        line(n, *(f"{one(bench, n=n, backend='cuda_global', check_every=k) / one(bench, n=n, backend='cuda_shared', check_every=k):.2f}" for k in (1, 50)))

    print("\n3. Checking convergence every 50 sweeps instead of every sweep")
    line("N", "global", "shared")
    for n in sorted({r["n"] for r in bench if r["backend"] != "cpu_reference"}):
        line(n, *(f"{one(bench, n=n, backend=b, check_every=1) / one(bench, n=n, backend=b, check_every=50):.2f}x" for b in ("cuda_global", "cuda_shared")))

    print("\n4. Tuning at N = 1024, check every 50 sweeps (ms per sweep, lower is better)")
    gpu = [r for r in tune if r["backend"] != "cpu_reference" and r["check_every"] == 50]
    for b in ("cuda_global", "cuda_shared"):
        rows = sorted(pick(gpu, backend=b), key=lambda r: r["ms"])
        base = one(gpu, backend=b, block="16x16", coarsen=1, dtype="float64")
        print(f"  {b}: default (16x16, 1 cell, float64) {base:.4f} ms")
        for r in rows[:3]:
            print(f"    best  {r['block']:>5} c{r['coarsen']} {r['dtype']}  {r['ms']:.4f} ms  ({base / r['ms']:.2f}x the default)")
        worst = rows[-1]
        print(f"    worst {worst['block']:>5} c{worst['coarsen']} {worst['dtype']}  {worst['ms']:.4f} ms")
        f64 = sum(r["ms"] for r in pick(gpu, backend=b, dtype="float64")) / len(pick(gpu, backend=b, dtype="float64"))
        f32 = sum(r["ms"] for r in pick(gpu, backend=b, dtype="float32")) / len(pick(gpu, backend=b, dtype="float32"))
        print(f"    float32 against float64 (mean of all settings): {f64 / f32:.2f}x faster")
        for blk in ("16x16", "32x8", "32x16"):
            m = [r["ms"] for r in pick(gpu, backend=b, block=blk, dtype="float64")]
            print(f"    block {blk} float64, mean over 1/2/4 cells: {sum(m) / len(m):.4f} ms")
        for c in (1, 2, 4):
            m = [r["ms"] for r in pick(gpu, backend=b, coarsen=c, dtype="float64")]
            print(f"    {c} cell(s) per thread, float64, mean over blocks: {sum(m) / len(m):.4f} ms")

    print("\n5. Check every sweep (ms per sweep) at N = 1024")
    every = [r for r in tune if r["backend"] != "cpu_reference" and r["check_every"] == 1]
    for b in ("cuda_global", "cuda_shared"):
        best = min(pick(every, backend=b), key=lambda r: r["ms"])
        base = one(every, backend=b, block="16x16", coarsen=1, dtype="float64")
        print(f"  {b}: default {base:.4f} ms, best {best['block']} c{best['coarsen']} {best['dtype']} {best['ms']:.4f} ms ({base / best['ms']:.2f}x)")


if __name__ == "__main__":
    main()
