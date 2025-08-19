#!/usr/bin/env python3
import sys, time, numpy as np, torch, os

# Force JAX to CPU (important on machines with GPUs)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp

from attn_cpu import gemm, gemm_jit

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
M = 2048
K = 1280
N = 960
N_RUNS = 20
N_WARMUPS = 3
SEED   = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
A_t, B_t = torch.tensor(A), torch.tensor(B)

A_j, B_j = jnp.asarray(A), jnp.asarray(B)

# ----------------------------------------------------------------------
# Candidate functions
# ----------------------------------------------------------------------
def torch_run(): return torch.matmul(A_t, B_t)
def numpy_run(): return A @ B
def cpu_run():   return gemm(A, B)
def cpu_jit_run(): return gemm_jit(A, B)

def jax_eager_run(): return A_j @ B_j

@jax.jit
def jax_jit_matmul(a, b):
    return a @ b

def jax_jit_run():
    return jax_jit_matmul(A_j, B_j)

fns = {
    "torch":    ("[PyTorch]", torch_run),
    "numpy":    ("[NumPy]", numpy_run),
    "cpu":      ("[GEMM]", cpu_run),
    "cpu_jit":  ("[GEMM JIT]", cpu_jit_run),
    "jax":      ("[JAX eager]", jax_eager_run),
    "jax_jit":  ("[JAX jit]", jax_jit_run),
}

# ----------------------------------------------------------------------
# Benchmark helper
# ----------------------------------------------------------------------
def timed_avg(fn, n=N_RUNS):
    # Warmup (3x). For JAX JIT this triggers compilation.
    for _ in range(N_WARMUPS):
        out = fn()
        # JAX returns lazy DeviceArray; block to measure fairly
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n, np.array(out)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    backend = sys.argv[1]
    name, fn = fns[backend]

    # Reference output (PyTorch)
    ref = torch_run().numpy()

    # Correctness check
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    atol, rtol = 1e-3, 1e-3
    try:
        np.testing.assert_allclose(np.array(out), ref, rtol=rtol, atol=atol)
        correct = "PASS"
    except AssertionError:
        correct = "FAIL"

    # Benchmark
    avg_time, _ = timed_avg(fn)
    print(f"{name:<12} {avg_time*1000:.3f} ms   [Correctness: {correct}]")
