#!/usr/bin/env python3
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ===========================
# Config — set these as needed
# ===========================
M        = 2048
K        = 1280*4
N        = 960


M        = 64
K        = 256
N        = 64




RUNS     = 20
WARMUPS  = 3
SEED     = 42
THREADS  = 8

# Optional kernel/env tuning
SGEMM_KC = 512            # set None to skip exporting
SGEMM_NC = 528
SGEMM_MC = 64
PROC_BIND = "close"        # "close" | "spread" | None
PLACES    = "cores"        # "cores" | "threads" | None

# If you want NumPy med printed as well, set to "numpy".
# Otherwise "cpu" is fine (bench_single still prints Torch + CPU medians).
BACKEND  = "cpu"

# ===========================
# Paths
# ===========================
HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_single.py"

def main():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    if PROC_BIND: env["OMP_PROC_BIND"] = PROC_BIND
    if PLACES:    env["OMP_PLACES"]    = PLACES
    env["SGEMM_KC"] = str(SGEMM_KC)
    env["SGEMM_NC"] = str(SGEMM_NC)
    env["SGEMM_MC"] = str(SGEMM_MC)

    print(f"=== Single GEMM test @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} ===")
    print(f"Dims: M={M}, K={K}, N={N} | Threads={THREADS} | Runs={RUNS} | Warmups={WARMUPS}\n")

    cmd = [
        sys.executable, str(BENCH), BACKEND,
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--runs", str(RUNS), "--warmups", str(WARMUPS),
        "--seed", str(SEED),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode != 0:
        if result.stderr.strip():
            print("stderr:", result.stderr.strip())
        print(f"[{BACKEND.upper()}] exited with code {result.returncode}")

if __name__ == "__main__":
    main()
