import numpy as np
import pytest

from heat.backends import BACKENDS, load
from heat.grid import BackendUnavailable, make_grid
from heat.scenarios import cases


def get(name):
    try:
        return load(name)
    except BackendUnavailable as e:
        pytest.skip(str(e))


def solve(name, *args, **kw):
    mod = get(name)
    try:
        return mod.solve(*args, **kw)
    except BackendUnavailable as e:
        pytest.skip(str(e))


def profile_for(name):
    return "sim" if "simulator" in get(name).MODE else "quick"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", [c[0] for c in cases("quick")])
def test_known_answers(backend, case):
    for name, grid, check, eps, tol, max_iter in cases(profile_for(backend)):
        if name == case:
            out, _, _ = solve(backend, grid, eps, max_iter=max_iter)
            err, msg = check(out)
            assert err <= tol, f"{backend} {case}: {msg}"


@pytest.mark.parametrize("case", [c[0] for c in cases("full")])
def test_cpu_full_profile(case):
    for name, grid, check, eps, tol, max_iter in cases("full"):
        if name == case:
            out, _, _ = solve("cpu_reference", grid, eps, max_iter=max_iter)
            assert check(out)[0] <= tol


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("check_every", [1, 3])
def test_same_result_as_cpu_and_boundary_fixed(backend, check_every):
    grid = np.random.default_rng(7).uniform(0, 100, (6, 6))
    ref, ref_it, _ = solve("cpu_reference", grid, 1e-2, check_every=check_every)
    out, it, _ = solve(backend, grid, 1e-2, check_every=check_every)
    assert it == ref_it and it % check_every == 0
    assert np.max(np.abs(out - ref)) < 1e-12
    for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        assert np.array_equal(out[edge], grid[edge])


@pytest.mark.parametrize("backend", ["cuda_global", "cuda_shared"])
def test_grid_is_not_copied_every_iteration(backend):
    grid = make_grid(6)
    _, it, _ = solve(backend, grid, 0.0, max_iter=4)
    from heat.cuda_common import LAST_RUN
    assert LAST_RUN["d2h_bytes"] == grid.nbytes + 16 * it
    assert LAST_RUN["h2d_bytes"] == 2 * grid.nbytes + 16


def test_bad_input_is_rejected():
    from heat.cpu_reference import solve as cpu
    with pytest.raises(ValueError):
        cpu(np.zeros((4, 5)), 1e-3)
    with pytest.raises(ValueError):
        cpu(np.zeros((4, 4)), 1e-3, check_every=0)
