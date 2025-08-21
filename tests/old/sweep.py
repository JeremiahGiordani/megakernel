#!/usr/bin/env python3
"""
Full GEMM sweep across the Cartesian product of M, K, and N lists.

- TXT: hierarchical markers when M changes (big banner) and when K changes (smaller divider)
- CSV: M,K,N,cpu_med_ms (median only)

Usage examples:
    python test/sweep.py
    python test/sweep.py --M_list 64,128,256 --K_list 256,512 --N_list 960,1024
    python test/sweep.py --limit 20

Outputs (under ./results/):
    results/gemm_sweep_YYYYmmdd_HHMMSS.txt
    results/gemm_sweep_YYYYmmdd_HHMMSS.csv
"""
import os
import time
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from itertools import product
from mk_cpu import gemm

# -----------------------------
# Helpers
# -----------------------------
def set_omp_env(threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("SGEMM_KC", "1024")


def parse_int_list(s: str, name: str):
    try:
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
        if not vals:
            raise ValueError
        return vals
    except Exception:
        raise argparse.ArgumentTypeError(f"--{name} must be a comma-separated list of ints (e.g., 64,128,256)")

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

def bench_case(M, K, N, warmups, runs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    A_t = torch.tensor(A)
    B_t = torch.tensor(B)

    def torch_run():
        return torch.matmul(A_t, B_t)

    def cpu_run():
        return gemm(A, B)

    torch_med = timed_median(torch_run, warmups, runs)
    cpu_med   = timed_median(cpu_run,   warmups, runs)
    return torch_med, cpu_med

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sweep GEMM shapes (Cartesian product of M, K, N) and log medians.")
    ap.add_argument("--M_list", type=lambda s: parse_int_list(s, "M_list"),
                    default=[64, 128, 256, 512, 768, 1024, 1536, 2048, 3072])
    ap.add_argument("--K_list", type=lambda s: parse_int_list(s, "K_list"),
                    default=[256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 5120, 6144])
    ap.add_argument("--N_list", type=lambda s: parse_int_list(s, "N_list"),
                    default=[64, 128, 256, 480, 576, 672, 960, 1008, 1024, 2048, 3072, 4032])

    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap total number of cases (useful for quick trials).")
    args = ap.parse_args()

    # Threading env
    set_omp_env(args.threads)

    # Build the Cartesian product in a predictable order (M-major, then K, then N)
    all_shapes = [(M, K, N) for M, K, N in product(args.M_list, args.K_list, args.N_list)]
    if args.limit > 0:
        all_shapes = all_shapes[:args.limit]

    # Output files
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = Path(args.outdir) / f"gemm_sweep_{stamp}.txt"
    csv_path = Path(args.outdir) / f"gemm_sweep_{stamp}.csv"

    # CSV header (CPU median only, as requested)
    with open(csv_path, "w") as fcsv:
        fcsv.write("M,K,N,cpu_med_ms\n")

    # TXT header
    with open(txt_path, "w") as ftxt:
        ftxt.write(f"GEMM sweep @ {datetime.utcnow().strftime('%a %b %d %H:%M:%S UTC %Y')}\n")
        ftxt.write(f"Threads={args.threads} | Repeats={args.runs} | Warmups={args.warmups}\n")
        ftxt.write(f"M_list={args.M_list}\n")
        ftxt.write(f"K_list={args.K_list}\n")
        ftxt.write(f"N_list={args.N_list}\n")
        ftxt.write("\n" + "=" * 80 + "\n\n")

    # Iterate and log with markers
    last_M, last_K = None, None
    with open(txt_path, "a") as ftxt, open(csv_path, "a") as fcsv:
        for (M, K, N) in all_shapes:
            if M != last_M:
                ftxt.write("\n" + "=" * 80 + "\n")
                ftxt.write(f"============================== M = {M} ==============================\n")
                ftxt.write("=" * 80 + "\n\n")
                last_M = M
                last_K = None  # reset K grouping

            if K != last_K:
                ftxt.write(f"---- K = {K} ----\n")
                last_K = K

            torch_med, cpu_med = bench_case(M, K, N, args.warmups, args.runs, args.seed)
            pct_diff = ((cpu_med - torch_med) / torch_med * 100.0) if torch_med > 0 else 0.0

            line = (f"  N={N:>5}  |  "
                    f"[PyTorch] med={torch_med:8.3f} ms   "
                    f"[GEMM] med={cpu_med:8.3f} ms   "
                    f"(Δ {pct_diff:+6.2f}%)")
            ftxt.write(line + "\n")
            ftxt.flush()

            # CSV: only CPU median, per requirement
            fcsv.write(f"{M},{K},{N},{cpu_med:.3f}\n")
            fcsv.flush()

    print(f"Wrote TXT: {txt_path}")
    print(f"Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
