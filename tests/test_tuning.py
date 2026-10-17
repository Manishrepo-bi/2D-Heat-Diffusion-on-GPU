"""Block shape, cells per thread and precision must not change the answer."""
import numpy as np
import pytest

from heat.backends import load
from heat.grid import BackendUnavailable, make_grid

cc = pytest.importorskip("heat.cuda_common")

CONFIGS = [("4x4", 1, "float64"), ("8x2", 2, "float64"), ("2x8", 3, "float64"), ("4x2", 1, "float32")]


@pytest.fixture(autouse=True)
def restore_settings():
    before = cc.config()
    yield
    cc.configure(block=f"{before.bx}x{before.by}", coarsen=before.coarsen, dtype=before.dtype)


def cuda_backend(name):
    try:
        return load(name)
    except BackendUnavailable as e:
        pytest.skip(str(e))


def solve(mod, *args, **kw):
    try:
        return mod.solve(*args, **kw)
    except BackendUnavailable as e:                             # no GPU and no HEAT_CUDA=sim
        pytest.skip(str(e))


@pytest.mark.parametrize("bad", [dict(block="3x5"), dict(block="64x32"), dict(block="abc"),
                                 dict(block="4x4", coarsen=0), dict(block="4x4", dtype="float16")])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        cc.make_config(**bad)


def test_launch_shape_covers_the_grid():
    cfg = cc.make_config("8x2", 3)
    blocks, threads = cc.launch_config(50, cfg)
    assert threads == (8, 2)
    assert blocks[0] * 8 >= 50 and blocks[1] * 2 * 3 >= 50


@pytest.mark.parametrize("backend", ["cuda_global", "cuda_shared"])
@pytest.mark.parametrize("block,coarsen,dtype", CONFIGS)
def test_same_answer_as_the_cpu(backend, block, coarsen, dtype):
    mod = cuda_backend(backend)
    cpu = load("cpu_reference")
    grid = make_grid(7, top=100, bottom=0, left=0, right=0)     # 7 does not divide evenly into any block
    ref, ref_it, _ = cpu.solve(grid, 5e-2)
    cc.configure(block=block, coarsen=coarsen, dtype=dtype)
    out, it, _ = solve(mod, grid, 5e-2)
    if dtype == "float64":
        assert it == ref_it
        assert np.max(np.abs(out - ref)) < 1e-12
    else:
        assert abs(it - ref_it) <= 2
        assert np.max(np.abs(out - ref)) < 5e-2
    assert out.dtype == np.float64                              # callers always get float64 back
    assert np.array_equal(out[0], grid[0]) and np.array_equal(out[:, 0], grid[:, 0])


def test_float32_halves_the_traffic():
    mod = cuda_backend("cuda_global")
    grid = make_grid(7, top=100, bottom=0, left=0, right=0)
    cc.configure(block="4x4", coarsen=1, dtype="float32")
    _, it, _ = solve(mod, grid, 5e-2)
    r = cc.LAST_RUN
    assert r["grid_bytes"] == grid.nbytes // 2
    assert r["d2h_bytes"] == grid.nbytes // 2 + 8 * it          # one float32 grid out + 8 bytes per check
