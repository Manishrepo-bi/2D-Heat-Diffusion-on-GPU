# 2D Heat Diffusion on the GPU

CUDA (Numba) implementation of 2D heat diffusion with Jacobi iteration and a five-point stencil,
written by a team of two as a git / pull-request exercise.

## Problem

An N x N grid represents a plate. The outer ring of cells holds fixed-temperature boundary
conditions. Every iteration updates each **interior** cell:

    T(i,j)^(t+1) = [ T(i-1,j)^t + T(i+1,j)^t + T(i,j-1)^t + T(i,j+1)^t ] / 4

until the maximum absolute change between two consecutive grids is below epsilon.

**Input:** N, the initial grid, the boundary conditions and epsilon.
**Output:** the converged grid, the number of iterations and the execution time.

### Boundary conditions
Boundary cells are inputs. They are never recalculated and keep their values to the end. Corner cells
are never read by the five-point stencil, so their value has no effect. Three cases are supported:

| Case | Boundary | Command | Exact answer used as a check |
|---|---|---|---|
| 3a uniform | every boundary cell at one temperature | `--case uniform --boundary 75` | the whole plate ends at 75 |
| 3b edges | one temperature per edge, or any value per cell from a file | `--case edges --top 100 --bottom 40 --left 20 --right 0`, or `--grid examples\initial_8x8.csv` | centre = average of the four edges = 40; a linear boundary stays a straight line |
| 3c hot edge | top edge 100, the other three 0 | `--case hot-edge --hot 100` | centre = 25 (each edge gives a quarter of its value at the centre) |

## Implementations

| Backend | File | Idea |
|---|---|---|
| `cpu_reference` | `heat/cpu_reference.py` | NumPy, the reference for correctness |
| `cuda_global` | `heat/cuda_global.py` | one thread per cell, reads its 4 neighbours from global memory |
| `cuda_shared` | `heat/cuda_shared.py` | each block loads a 16 x 16 tile plus a 1-cell halo into shared memory, then computes from it |

Both CUDA kernels share `heat/cuda_common.py`:
* **Parallel convergence check.** Each launch reduces |new - old| per block in shared memory, and one
  atomic max per block produces the global maximum on the GPU.
* **No unnecessary transfers.** Both grids stay on the GPU for the whole run. Per check only 16 bytes
  come back. `solve.py` prints the bytes moved.
* `--check-every k` checks convergence every k sweeps to cut even those copies.

### Tuning options (both CUDA kernels)

| Option | Environment variable | Meaning |
|---|---|---|
| `--block 32x8` | `HEAT_BLOCK` | Thread-block shape, width x height (default 16 x 16). The thread count must be a power of two, at most 1024. A wide block reads whole 32-element rows, which suits GPU memory. |
| `--coarsen 2` | `HEAT_COARSEN` | Cells updated per thread down a column (default 1): fewer threads and less index arithmetic. |
| `--dtype float32` | `HEAT_DTYPE` | Half the memory traffic. It resolves changes down to about 1e-5 only, so use `--eps 1e-3` or larger. |

`solve.py` and `verify_all.py` accept the options directly. `benchmark.py --blocks 16x16 32x8 --coarsen 1 2 4
--dtype float64 float32` tries every combination. The best block shape depends on the GPU, so it is measured and not assumed.

## Work breakdown

| Part | Files |
|---|---|
| Foundation | grid set-up, CPU reference, known-answer checks, `solve.py`, `verify_all.py`, `benchmark.py`, tests, emulator tools |
| CUDA driver and global-memory kernel (pull request #1) | `heat/cuda_common.py`, `heat/cuda_global.py` |
| Shared-memory kernel (pull request #2) | `heat/cuda_shared.py` |
| Tuning options (pull request #3) | block shape, cells per thread and float32 in `heat/cuda_common.py`, both kernels, `solve.py`, `verify_all.py`, `benchmark.py`, `tests/test_tuning.py` |

Code moves through a git remote in a folder on disk. Pull requests and merges go through Vercel Labs'
GitHub emulator (`tools\start_emulator.bat`, http://127.0.0.1:4000) using `tools/gh_emu.py`, which saves
every action to disk. The team has no reviewer, so merges use `pr-merge N --no-review`. The merge still
runs the full test suite and pushes only if it passes.

## Results on a Tesla T4

The benchmark was run on a Google Colab Tesla T4. The full comparison is in `docs/T4_RESULTS.md`, the raw numbers in `results/benchmark_t4.csv` and `results/tuning_t4.csv`, and `python compare_results.py` prints the tables again. `colab/heat_diffusion_colab.ipynb` reproduces the run.

| N | CPU ms/sweep | global ms/sweep | shared ms/sweep | global over CPU | shared over global |
|---|---|---|---|---|---|
| 256 | 0.525 | 0.090 | 0.088 | 5.8x | 1.02x |
| 512 | 4.368 | 0.134 | 0.121 | 32.7x | 1.10x |
| 1024 | 12.959 | 0.172 | 0.212 | 75.2x | 0.81x |
| 2048 | 69.914 | 0.503 | 0.618 | 138.9x | 0.81x |

Shared-memory tiling is not faster on the T4 at large sizes. float32 is about 1.9x faster than float64 at N = 1024.

## Commands

    python check_env.py                                   what this computer can run
    python -m pytest                                      tests
    python verify_all.py                                  known-answer checks for every backend
    python solve.py --backend cuda_shared --n 512 --case edges --top 100 --bottom 40 --left 20 --eps 1e-4
    python solve.py --case uniform --boundary 75 --n 9
    python solve.py --case hot-edge --hot 100 --n 9
    python solve.py --grid examples\initial_8x8.csv --eps 1e-6 --out results\grid.csv
    python benchmark.py                                   global vs shared vs CPU -> results\benchmark.csv

No NVIDIA GPU: `set HEAT_CUDA=sim` runs the same kernels in Numba's CUDA simulator. It is slow and
for correctness only. `benchmark.py` refuses to time it.

## Known-answer checks

| Check | Setup | Exact answer |
|---|---|---|
| sweep_6x6 | top/bottom 100, sides 0, one sweep | rows 1 and 4 = 25, rows 2 and 3 = 0 (docs/heat_diffusion_sweep.xlsx) |
| case_3a_uniform | every boundary cell 75 | 75 everywhere |
| case_3b_edges | top 100, bottom 40, left 20, right 0 | centre = 40 (average of the edges, by symmetry) |
| case_3b_ramp | boundary = 100 (1 - j/(N-1)) | the same straight line everywhere |
| case_3c_hot_edge | top 100, other edges 0 | centre = 25 |
