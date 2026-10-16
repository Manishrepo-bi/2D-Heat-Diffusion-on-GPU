"""Load a backend by name and turn 'cannot run here' into a clear reason."""
import importlib

from heat.grid import BackendUnavailable

BACKENDS = ["cpu_reference", "cuda_global", "cuda_shared"]


def load(name):
    """Return the backend module, or raise BackendUnavailable with the reason."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    try:
        mod = importlib.import_module(f"heat.{name}")
    except ImportError as e:
        raise BackendUnavailable(f"{e.name} is not installed") from None
    if getattr(mod, "PLACEHOLDER", False):
        raise BackendUnavailable("not implemented yet")
    return mod
