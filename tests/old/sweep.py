#!/usr/bin/env python3
"""
Process-isolated GEMM sweep that invokes bench_single.py for each (M,K,N).

- Runs each case in a separate Python process via bench_single.py to ensure fair comparisons.
- TXT output: big banner when M changes, small divider when K changes; shows Torch/GEMM medians and Δ%.
- CSV output: M,K,N,cpu_med_ms  (CPU median only).

Examples:
  python test/sweep.py
  python test/sweep.py --M_list 64,128 --K_list 256,512 --N_list 960,1024 --runs 15 --warmups 3 --threads 8
  python test/sweep.py --limit 20
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

SGEMM_KC = 1024
SGEMM_NC = 528
SGEMM_MC = 64

# -----------------------------
# Parsing helpers
# -----------------------------
def parse_int_list(s: str, name: str):
    if isinstance(s, list):  # already parsed (arg default)
        return s
    try:
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
        if not vals:
            raise ValueError
        return vals
    except Exception:
        raise argparse.ArgumentTypeError(f"--{name} must be a comma-separated list of ints, e.g. 64,128,256")

# -----------------------------
# Subprocess runner & parser
# -----------------------------
def call_bench_single(bench_path: Path, M: int, K: int, N: int, runs: int, warmups: int, seed: int, env: dict):
    """
    Invoke bench_single.py in a separate process, return (torch_med_ms, cpu_med_ms, stdout).
    bench_single.py prints a single line like:
      Dims M=... K=... N=... | [PyTorch] 12.345 ms | [GEMM] 11.876 ms
    (per your updated version that prints medians for both)
    """
    cmd = [
        sys.executable, str(bench_path), "cpu",
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--runs", str(runs), "--warmups", str(warmups),
        "--seed", str(seed),
    ]
    from subprocess import run, PIPE
    res = run(cmd, stdout=PIPE, stderr=PIPE, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"bench_single.py failed (code {res.returncode})\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    out = res.stdout.strip()

    # Try two regex formats (depending on your bench_single formatting)
    # 1) "[PyTorch] med=  12.345 ms   [GEMM] med=  11.876 ms"
    m = re.search(r"\[PyTorch\]\s*med=\s*([0-9.]+)\s*ms.*?\[GEMM\]\s*med=\s*([0-9.]+)\s*ms", out)
    if not m:
        # 2) "[PyTorch] 12.345 ms | [GEMM] 11.876 ms"
        m = re.search(r"\[PyTorch\]\s*([0-9.]+)\s*ms.*?\[GEMM\]\s*([0-9.]+)\s*ms", out)
    if not m:
        raise ValueError(f"Could not parse medians from bench_single.py output:\n{out}")

    torch_med = float(m.group(1))
    cpu_med   = float(m.group(2))
    return torch_med, cpu_med, out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Process-isolated GEMM sweep using bench_single.py")
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
    ap.add_argument("--limit", type=int, default=0, help="If >0, cap total number of cases (quick trials).")

    # Optional env/kernel hints (mirror your runner.py knobs)
    ap.add_argument("--sgemm_kc", type=int, default=None, help="Export SGEMM_KC to env if set")
    ap.add_argument("--proc_bind", type=str, default="close", choices=[None, "close", "spread", "master"], nargs='?')
    ap.add_argument("--places", type=str, default="cores", choices=[None, "cores", "threads", "sockets"], nargs='?')

    args = ap.parse_args()

    # Locate bench_single.py next to this script
    here = Path(__file__).resolve().parent
    bench_path = here / "bench_single.py"
    if not bench_path.exists():
        raise FileNotFoundError(f"bench_single.py not found at {bench_path}")

    # Environment for subprocess
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    if args.proc_bind:
        env["OMP_PROC_BIND"] = str(args.proc_bind)
    if args.places:
        env["OMP_PLACES"] = str(args.places)
    env["SGEMM_KC"] = str(SGEMM_KC)
    env["SGEMM_NC"] = str(SGEMM_NC)
    env["SGEMM_MC"] = str(SGEMM_MC)

    # Build Cartesian product (M-major, then K, then N)
    all_shapes = [(M, K, N) for M, K, N in product(args.M_list, args.K_list, args.N_list)]
    if args.limit > 0:
        all_shapes = all_shapes[:args.limit]

    # Prepare outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = outdir / f"gemm_sweep_{stamp}.txt"
    csv_path = outdir / f"gemm_sweep_{stamp}.csv"

    # Write headers
    with open(csv_path, "w") as fcsv:
        fcsv.write("M,K,N,cpu_med_ms\n")
    with open(txt_path, "w") as ftxt:
        ftxt.write(f"GEMM sweep @ {datetime.utcnow().strftime('%a %b %d %H:%M:%S UTC %Y')}\n")
        ftxt.write(f"Threads={args.threads} | Repeats={args.runs} | Warmups={args.warmups}\n")
        ftxt.write(f"M_list={args.M_list}\n")
        ftxt.write(f"K_list={args.K_list}\n")
        ftxt.write(f"N_list={args.N_list}\n")
        ftxt.write("\n" + "=" * 80 + "\n\n")

    # Sweep with hierarchical markers
    last_M, last_K = None, None
    with open(txt_path, "a") as ftxt, open(csv_path, "a") as fcsv:
        for (M, K, N) in all_shapes:
            if M != last_M:
                ftxt.write("\n" + "=" * 80 + "\n")
                ftxt.write(f"============================== M = {M} ==============================\n")
                ftxt.write("=" * 80 + "\n\n")
                last_M = M
                last_K = None

            if K != last_K:
                ftxt.write(f"---- K = {K} ----\n")
                last_K = K

            torch_med, cpu_med, _ = call_bench_single(
                bench_path=bench_path, M=M, K=K, N=N,
                runs=args.runs, warmups=args.warmups, seed=args.seed, env=env
            )
            pct_diff = ((cpu_med - torch_med) / torch_med * 100.0) if torch_med > 0 else 0.0

            line = (f"  N={N:>5}  |  "
                    f"[PyTorch] med={torch_med:8.3f} ms   "
                    f"[GEMM] med={cpu_med:8.3f} ms   "
                    f"(Δ {pct_diff:+6.2f}%)")
            ftxt.write(line + "\n")
            ftxt.flush()

            # CSV: CPU median only
            fcsv.write(f"{M},{K},{N},{cpu_med:.3f}\n")
            fcsv.flush()

    print(f"Wrote TXT: {txt_path}")
    print(f"Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
