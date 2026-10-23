import os
import platform
import shutil
import subprocess
import sys
import urllib.request

OK, MISS, INFO = "[ OK ]", "[MISS]", "[INFO]"


def probe(code, sim=False):
    env = dict(os.environ)
    env.pop("NUMBA_ENABLE_CUDASIM", None)
    if sim:
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    out = r.stdout.strip()
    if not out and r.stderr.strip():
        out = r.stderr.strip().splitlines()[-1]
    return r.returncode == 0, out


def main():
    can_run = []
    print(f"{INFO} computer {platform.node()}, {platform.system()} {platform.release()}")
    print(f"{OK if sys.version_info >= (3, 9) else MISS} python {platform.python_version()} (need 3.9 or newer)")
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"{OK if venv else INFO} virtual environment {'active' if venv else 'not active (recommended: .venv)'}")
    if shutil.which("git"):
        print(f"{OK} " + subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip())
    else:
        print(f"{MISS} git not found on PATH")
    for mod in ("numpy", "pytest"):
        ok, out = probe(f"import {mod}; print({mod}.__version__)")
        print(f"{OK if ok else MISS} {mod} {out if ok else '(pip install -r requirements.txt)'}")
    can_run.append("cpu_reference")

    ok, out = probe("import numba; print(numba.__version__)")
    if ok:
        print(f"{OK} numba {out}")
        gpu, name = probe("from numba import cuda; import sys; ok = cuda.is_available(); "
                          "d = cuda.get_current_device().name if ok else b''; "
                          "print(d.decode() if isinstance(d, bytes) else d); sys.exit(0 if ok else 1)")
        sim, _ = probe("from numba import cuda; import sys; sys.exit(0 if cuda.is_available() else 1)", sim=True)
        print(f"{OK if gpu else INFO} NVIDIA GPU: {name if gpu else 'none found'}")
        print(f"{OK if sim else MISS} CUDA simulator (set HEAT_CUDA=sim): {'works' if sim else 'not working'}")
        if gpu:
            can_run += ["cuda_global (GPU)", "cuda_shared (GPU)", "benchmark.py"]
        elif sim:
            can_run += ["cuda_global (HEAT_CUDA=sim)", "cuda_shared (HEAT_CUDA=sim)"]
    else:
        print(f"{INFO} numba not installed (pip install -r requirements-cuda.txt)")

    print(f"{OK if shutil.which('npx') else INFO} node/npx {'found' if shutil.which('npx') else 'not found (only the emulator host needs it)'}")
    url = os.environ.get("GH_EMU_URL", "http://127.0.0.1:4000")
    try:
        urllib.request.urlopen(url + "/repos/participant1/2D-Heat-Diffusion-on-GPU", timeout=3)
        print(f"{OK} GitHub emulator answering at {url}")
    except Exception:
        print(f"{INFO} GitHub emulator not reachable at {url} (the emulator host starts it)")
    print(f"{INFO} GH_EMU_USER = {os.environ.get('GH_EMU_USER', '(not set)')}")
    print("\nCan run here: " + ", ".join(can_run))


if __name__ == "__main__":
    main()
