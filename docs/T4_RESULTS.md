# Results on a Tesla T4 (Google Colab)

The benchmark ran on a free Colab Tesla T4 on 2026-10-17 with Python 3.13.15 and Numba 0.61.2. Each GPU run does 500 sweeps and each CPU run 20 sweeps (`eps = 0`, so every backend does the same work per sweep). The fastest of 3 runs is kept. The raw numbers are in `results/benchmark_t4.csv` and `results/tuning_t4.csv`. The tables below come from `python compare_results.py`.

Before the benchmark, both kernels passed all five known-answer cases on the GPU (centres exactly 40 and 25) and all 44 tests passed.

## Reproduce

Open `colab/heat_diffusion_colab.ipynb` in Colab, choose Runtime > Change runtime type > T4 GPU and run all cells. The notebook clones this repository and runs only the checks, the benchmark and the comparison.

## Comparison output

```
1. GPU against CPU (ms per sweep, speed-up over the CPU)
N                              CPU      global       x CPU      shared       x CPU
256                          0.525       0.090         5.8       0.088         6.0
512                          4.368       0.134        32.7       0.121        36.0
1024                        12.959       0.172        75.2       0.212        61.2
2048                        69.914       0.503       138.9       0.618       113.2

2. Shared memory against global memory (above 1 = shared is faster)
N                          check 1    check 50
256                           1.02        1.00
512                           1.10        0.97
1024                          0.81        0.88
2048                          0.81        0.84

3. Checking convergence every 50 sweeps instead of every sweep
N                           global      shared
256                          2.11x       2.05x
512                          2.94x       2.60x
1024                         2.23x       2.40x
2048                         1.72x       1.78x

4. Tuning at N = 1024, check every 50 sweeps (ms per sweep, lower is better)
  cuda_global: default (16x16, 1 cell, float64) 0.0774 ms
    best  16x16 c1 float32  0.0415 ms  (1.87x the default)
    best   32x8 c4 float32  0.0429 ms  (1.80x the default)
    best  16x16 c4 float32  0.0431 ms  (1.80x the default)
    worst 32x16 c4 float64  0.0822 ms
    float32 against float64 (mean of all settings): 1.75x faster
    block 16x16 float64, mean over 1/2/4 cells: 0.0793 ms
    block 32x8 float64, mean over 1/2/4 cells: 0.0777 ms
    block 32x16 float64, mean over 1/2/4 cells: 0.0796 ms
    1 cell(s) per thread, float64, mean over blocks: 0.0771 ms
    2 cell(s) per thread, float64, mean over blocks: 0.0784 ms
    4 cell(s) per thread, float64, mean over blocks: 0.0812 ms
  cuda_shared: default (16x16, 1 cell, float64) 0.0879 ms
    best   32x8 c4 float32  0.0474 ms  (1.85x the default)
    best  16x16 c4 float32  0.0476 ms  (1.85x the default)
    best  32x16 c4 float32  0.0501 ms  (1.75x the default)
    worst 32x16 c1 float64  0.0979 ms
    float32 against float64 (mean of all settings): 1.70x faster
    block 16x16 float64, mean over 1/2/4 cells: 0.0884 ms
    block 32x8 float64, mean over 1/2/4 cells: 0.0874 ms
    block 32x16 float64, mean over 1/2/4 cells: 0.0939 ms
    1 cell(s) per thread, float64, mean over blocks: 0.0912 ms
    2 cell(s) per thread, float64, mean over blocks: 0.0883 ms
    4 cell(s) per thread, float64, mean over blocks: 0.0902 ms

5. Check every sweep (ms per sweep) at N = 1024
  cuda_global: default 0.2000 ms, best 32x8 c2 float32 0.1040 ms (1.92x)
  cuda_shared: default 0.2075 ms, best 16x16 c4 float32 0.1105 ms (1.88x)
```

## What the comparison shows

- **GPU against CPU.** The T4 is already 5.8x faster than the CPU at N = 256 (global kernel). The gap grows with the grid: 32.7x at 512, 75.2x at 1024 and 138.9x at 2048, because small grids cannot fill the GPU.
- **Shared memory against global memory.** Tiling is only 1.02x to 1.10x faster at N = 256 and 512 when the check runs every sweep. At N = 1024 and 2048 it is slower (0.81x). With a check every 50 sweeps the global kernel is ahead at every size except N = 256, where they tie. The T4's caches already serve most of the neighbour reads, so the extra tile loading and synchronisation cost more than they save.
- **Convergence check interval.** Checking every 50 sweeps instead of every sweep makes a sweep 1.7x to 2.9x faster, because the 16-byte copy back to the CPU and the wait for it happen far less often.
- **float32.** Halving the precision cuts the time per sweep by about 1.7x to 1.75x on average over all block shapes and cell counts, because the kernels are limited by memory traffic. The best global setting (16x16, 1 cell, float32) takes 0.0415 ms per sweep against 0.0774 ms for the default, which is 1.87x. float32 only resolves changes down to about 1e-5, so it needs an epsilon of 1e-3 or larger.
- **Block shape and cells per thread.** Both change the time by less than about 8% in float64. The one clear loser is a 32x16 block with the shared kernel (0.0939 ms against about 0.088 ms). Coarsening helps when the check runs every sweep (global, 16x16: 0.2000 ms with 1 cell against 0.1592 and 0.1560 ms with 2 and 4) and does nothing once the check runs every 50 sweeps. The defaults are already close to the best in float64.

## Limits

Each number is the fastest of 3 runs on a shared Colab GPU, so differences of a few percent are noise. The tuning benchmark covers N = 1024 only. The results describe the T4 and may not carry over to other GPUs.
