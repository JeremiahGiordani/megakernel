#!/usr/bin/env python3
import os, sys, time, argparse

# ---- set env *before* importing numpy/torch ----
# Keep other libs from stealing threads or spin-waiting.
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")      # Intel OpenMP: no spin-wait
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # GCC OpenMP: sleep-yield
# Your kernel settings (inherit from parent if already set)
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

# os.environ.setdefault("SGEMM_U", "3")
# os.environ.setdefault("SGEMM_TUNE", "0")

import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["torch","numpy","cpu"], help="which backend to time")
    p.add_argument("--ref", choices=["none","numpy","torch"], default="numpy",
                   help="reference for correctness when timing 'cpu'")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--k", type=int, default=1280*4)
    p.add_argument("--n", type=int, default=960)
    return p.parse_args()

def timed_avg(fn, warmups, runs):
    import gc
    gc.disable()
    try:
        for _ in range(warmups): 
            out = fn()
            if hasattr(out, "block_until_ready"): out.block_until_ready()
        total = 0.0
        for _ in range(runs):
            t0 = time.perf_counter()
            out = fn()
            if hasattr(out, "block_until_ready"): out.block_until_ready()
            t1 = time.perf_counter()
            total += (t1 - t0)
    finally:
        gc.enable()
    return total / runs, np.array(out)

def main():
    args = parse_args()
    M, K, N = args.m, args.k, args.n
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    if args.backend == "torch" or args.ref == "torch":
        import torch
        A_t, B_t = torch.tensor(A), torch.tensor(B)
        def torch_run(): return torch.matmul(A_t, B_t)

    if args.backend == "numpy" or args.ref == "numpy":
        def numpy_run(): return A @ B

    if args.backend == "cpu":
        from mk_cpu import gemm
        def cpu_run(): return gemm(A, B)

        # correctness (cheap and single-threaded by env above)
        if args.ref == "numpy":
            ref = numpy_run()
        elif args.ref == "torch":
            ref = torch_run().numpy()
        else:
            ref = None

        if ref is not None:
            out = cpu_run()
            np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-3, atol=1e-3)
            correct = "PASS"
        else:
            correct = "SKIP"

        mean, _ = timed_avg(cpu_run, args.warmups, args.runs)
        print(f"[GEMM]      {mean*1000:.3f} ms   [Correctness: {correct}]")
        return

    if args.backend == "torch":
        mean, _ = timed_avg(torch_run, args.warmups, args.runs)
        print(f"[PyTorch]   {mean*1000:.3f} ms   [Correctness: PASS]")
        return

    if args.backend == "numpy":
        mean, _ = timed_avg(numpy_run, args.warmups, args.runs)
        print(f"[NumPy]     {mean*1000:.3f} ms   [Correctness: PASS]")
        return

if __name__ == "__main__":
    main()
