#!/usr/bin/env python3
import os, sys, subprocess

backends = ["torch", "numpy", "cpu", "jax", "jax_jit"]
backends = ["torch", "numpy", "cpu"]

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "8"
env["OMP_PROC_BIND"] = "close"
env["OMP_PLACES"] = "cores"
env["SGEMM_MC"] = "256"
env["SGEMM_NC"] = "48"
env["SGEMM_KC"] = "1280"


print("=== Isolated Benchmarks ===")
here = os.path.dirname(__file__)

for backend in backends:
    print(f"\n>>> Running {backend.upper()} benchmark...")
    result = subprocess.run(
        [sys.executable, os.path.join(here, "bench_single.py"), backend],
        capture_output=True,
        text=True,
        env=env,
    )

    # Print stdout if present
    if result.stdout.strip():
        print(result.stdout.strip())

    # Print stderr if present
    if result.returncode != 0 and result.stderr.strip():
        print("stderr:", result.stderr.strip())

    # If the process crashed (non-zero exit code), say so
    if result.returncode != 0:
        print(f"[{backend.upper()}] exited with code {result.returncode}")
