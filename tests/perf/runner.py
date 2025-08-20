#!/usr/bin/env python3
import os, sys, subprocess, shlex

env = os.environ.copy()
# Thread placement (match your C++ runs)
env.setdefault("OMP_NUM_THREADS", "8")
env.setdefault("OPENBLAS_NUM_THREADS", "8")
env.setdefault("MKL_NUM_THREADS", "8")
env.setdefault("OMP_PROC_BIND", "close")
env.setdefault("OMP_PLACES", "cores")
env.setdefault("OMP_DYNAMIC", "FALSE")

# Block sizes
env.setdefault("SGEMM_MC", "64")
env.setdefault("SGEMM_NC", "528")
env.setdefault("SGEMM_KC", "512")

here = os.path.dirname(__file__)

def run(cmd):
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print("stderr:", r.stderr.strip())
    if r.returncode != 0:
        print(f"exited with code {r.returncode}")

print("=== Isolated Benchmarks ===")
print("\n>>> Running TORCH benchmark...")
run([sys.executable, os.path.join(here, "bench_single.py"), "torch", "--ref", "none"])

print("\n>>> Running NUMPY benchmark...")
run([sys.executable, os.path.join(here, "bench_single.py"), "numpy", "--ref", "none"])

print("\n>>> Running CPU benchmark...")
# IMPORTANT: do not pull PyTorch into this process; use numpy ref only
run([sys.executable, os.path.join(here, "bench_single.py"), "cpu", "--ref", "numpy"])
