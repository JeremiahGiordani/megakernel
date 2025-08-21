#!/usr/bin/env python3
import sys, time, numpy as np, torch, os, argparse
from mk_cpu import gemm

def timed_median(fn, warmups: int, runs: int) -> float:
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return float(np.median(times))

def main():
    p = argparse.ArgumentParser(description="Single-case GEMM benchmark (median only).")
    p.add_argument("backend", choices=["torch", "numpy", "cpu"])
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--K", type=int, default=1280)
    p.add_argument("--N", type=int, default=960)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Optional threading env (kept compatible with your runner.py defaults)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    M, K, N = args.M, args.K, args.N

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    A_t = torch.tensor(A)
    B_t = torch.tensor(B)

    def torch_run(): return torch.matmul(A_t, B_t)
    def numpy_run(): return A @ B
    def cpu_run():   return gemm(A, B)

    # Always compute both torch & cpu medians for side-by-side view
    torch_med = timed_median(torch_run, args.warmups, args.runs)
    cpu_med   = timed_median(cpu_run,   args.warmups, args.runs)

    # Optional: if specifically asked for numpy backend, also compute it
    numpy_med = None
    if args.backend == "numpy":
        numpy_med = timed_median(numpy_run, args.warmups, args.runs)

    # Print a concise single line
    out = [f"Dims M={M} K={K} N={N}",
           f"[PyTorch] {torch_med:.3f} ms",
           f"[GEMM] {cpu_med:.3f} ms"]
    if numpy_med is not None:
        out.append(f"[NumPy] {numpy_med:.3f} ms")

    print(" | ".join(out))

if __name__ == "__main__":
    main()
