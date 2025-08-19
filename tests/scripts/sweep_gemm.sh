#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
M="${M:-2048}"
N="${N:-960}"
K="${K:-1280}"

# Force the same OpenMP env as your Python harness
THREADS="${THREADS:-8}"               # << match Python
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
# kill stray BLAS threads from other libs just in case
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

REPEAT="${REPEAT:-5}"                # << match Python 20 runs
WARMUPS="${WARMUPS:-3}"               # << match Python

# Your benchmark exe (produced as build/attn)
CMD="${CMD:-./build/attn --bench gemm --m $M --n $N --k $K --repeats $REPEAT --warmups $WARMUPS --threads $THREADS}"

MC_LIST=(${MC_LIST:-128 192 256 512 2048})
KC_LIST=(${KC_LIST:-512 768 1024 1280})
NC_LIST=(${NC_LIST:-48 96 144 192 960})

OUT="${OUT:-results_sweep.csv}"

extract_ms() { grep -Eo '([0-9]+(\.[0-9]+)?)\s*ms' | head -n1 | sed -E 's/[[:space:]]*ms//'; }
mean_min() { awk 'BEGIN{n=0;s=0;min=1e99}{s+=$1;n++;if($1<min)min=$1}END{printf "%.6f,%.6f\n",s/n,min}'; }
gflops() { awk -v m="$M" -v n="$N" -v k="$K" -v ms="$1" 'BEGIN{ops=2.0*m*n*k;print ops/(ms/1000.0)/1e9}'; }

warmup_once() {
  SGEMM_TUNE=1 SGEMM_MC="$1" SGEMM_KC="$2" SGEMM_NC="$3" bash -c "$CMD" >/dev/null 2>&1 || true
}

run_one() {
  local mc="$1" kc="$2" nc="$3"
  local times=()

  warmup_once "$mc" "$kc" "$nc"

  for ((r=1; r<=REPEAT; ++r)); do
    out=$(SGEMM_TUNE=0 SGEMM_MC="$mc" SGEMM_KC="$kc" SGEMM_NC="$nc" bash -c "$CMD" 2>&1 || true)
    ms=$(printf "%s" "$out" | extract_ms || true)
    if [[ -z "${ms:-}" ]]; then
      echo "WARN: parse fail MC=$mc KC=$kc NC=$nc (run $r)" >&2; continue
    fi
    times+=("$ms")
    echo "MC=$mc KC=$kc NC=$nc :: run $r => ${ms} ms"
  done

  if [[ ${#times[@]} -eq 0 ]]; then echo "ERROR: no timings MC=$mc KC=$kc NC=$nc" >&2; return; fi
  stats=$(printf "%s\n" "${times[@]}" | mean_min)
  mean_ms="${stats%,*}"; min_ms="${stats#*,}"
  gf=$(gflops "$mean_ms")
  printf "%-8s %-8s %-8s  mean=%8.3f ms  min=%8.3f ms  GF/s=%8.2f\n" "$mc" "$kc" "$nc" "$mean_ms" "$min_ms" "$gf"
  printf "%s,%s,%s,%.6f,%.6f,%.3f\n" "$mc" "$kc" "$nc" "$mean_ms" "$min_ms" "$gf" >> "$OUT"
}

echo "Saving results to: $OUT"
echo "MC,KC,NC,mean_ms,min_ms,GFLOPs" > "$OUT"

round_to_multiple(){ echo $(( ( ($1 + $2 - 1) / $2 ) * $2 )); }

for mc in "${MC_LIST[@]}"; do
  mc_rt=$(round_to_multiple "$mc" 8)
  for kc in "${KC_LIST[@]}"; do
    (( kc > K )) && kc_rt="$K" || kc_rt="$kc"
    for nc in "${NC_LIST[@]}"; do
      nc_rt=$(round_to_multiple "$nc" 48); (( nc_rt > N )) && nc_rt="$N"
      run_one "$mc_rt" "$kc_rt" "$nc_rt"
    done
  done
done

echo; echo "Top 10 (by mean_ms):"
tail -n +2 "$OUT" | sort -t',' -k4,4g | head -n 10 | nl -w2 -s'. '
echo; echo "Done → $OUT"
