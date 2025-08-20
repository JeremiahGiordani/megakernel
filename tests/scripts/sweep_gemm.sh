#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG ---------
M="${M:-2048}"
N="${N:-960}"
K="${K:-1280}"

THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

REPEAT="${REPEAT:-8}"
WARMUPS="${WARMUPS:-3}"

CMD="${CMD:-./build/mk --bench gemm --m $M --n $N --k $K --repeats $REPEAT --warmups $WARMUPS --threads $THREADS}"

MC_LIST=(${MC_LIST:-48 64 128 256})
KC_LIST=(${KC_LIST:-256 512 768 1024 1280})
NC_LIST=(${NC_LIST:-48 96 192 240 336 528 816 960})

OUT_CSV="${OUT_CSV:-results_sweep.csv}"
OUT_TXT="${OUT_TXT:-results_sweep.txt}"

# --------- HELPERS ---------
extract_ms() { grep -Eo '([0-9]+(\.[0-9]+)?)\s*ms' | head -n1 | sed -E 's/[[:space:]]*ms//'; }
round_to_multiple(){ echo $(( ( ($1 + $2 - 1) / $2 ) * $2 )); }
gflops() { awk -v m="$M" -v n="$N" -v k="$K" -v ms="$1" 'BEGIN{ops=2.0*m*n*k;print ops/(ms/1000.0)/1e9}'; }

# mean & min from a newline-separated list (portable awk)
mean_min_from_list() {
  awk 'BEGIN{n=0;s=0;min=1e99}
       {s+=$1; n++; if($1<min)min=$1}
       END{if(n==0){print "NaN,NaN"}else{printf "%.6f,%.6f\n", s/n, min}}'
}

# median using sort -n (portable) on a newline-separated list
median_from_list() {
  local data sorted n mid a b
  sorted=$(LC_ALL=C sort -n)
  n=$(printf "%s\n" "$sorted" | wc -l | tr -d ' ')
  if [ "$n" -eq 0 ]; then
    echo "NaN"
    return
  fi
  if [ $((n % 2)) -eq 1 ]; then
    mid=$(( (n + 1) / 2 ))
    printf "%s\n" "$sorted" | sed -n "${mid}p"
  else
    a_idx=$(( n / 2 ))
    b_idx=$(( a_idx + 1 ))
    a=$(printf "%s\n" "$sorted" | sed -n "${a_idx}p")
    b=$(printf "%s\n" "$sorted" | sed -n "${b_idx}p")
    awk -v a="$a" -v b="$b" 'BEGIN{printf "%.6f\n", (a+b)/2.0}'
  fi
}

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

  if [[ ${#times[@]} -eq 0 ]]; then
    echo "ERROR: no timings MC=$mc KC=$kc NC=$nc" >&2
    return
  fi

  # Build a newline-separated list once
  local list
  list=$(printf "%s\n" "${times[@]}")

  # mean & min (portable awk)
  local mm mean_ms min_ms
  mm=$(printf "%s\n" "$list" | mean_min_from_list)
  mean_ms="${mm%,*}"
  min_ms="${mm#*,}"

  # median (portable: sort -n)
  local med_ms
  med_ms=$(printf "%s\n" "$list" | median_from_list)

  gf=$(gflops "$mean_ms")

  # Console summary
  printf "%-8s %-8s %-8s  med=%8.3f ms  mean=%8.3f ms  min=%8.3f ms  GF/s=%8.2f\n" \
    "$mc" "$kc" "$nc" "$med_ms" "$mean_ms" "$min_ms" "$gf"

  # CSV
  printf "%s,%s,%s,%.6f,%.6f,%.6f,%.3f\n" "$mc" "$kc" "$nc" "$mean_ms" "$min_ms" "$med_ms" "$gf" >> "$OUT_CSV"

  # TXT
  printf "  NC=%-5s  med=%8.3f ms  mean=%8.3f ms  min=%8.3f ms  GF/s=%8.2f\n" \
    "$nc" "$med_ms" "$mean_ms" "$min_ms" "$gf" >> "$OUT_TXT"
}

# --------- INIT OUTPUTS ---------
echo "Saving results to: $OUT_CSV and $OUT_TXT"
echo "MC,KC,NC,mean_ms,min_ms,med_ms,GFLOPs" > "$OUT_CSV"

{
  echo "GEMM sweep @ $(date)"
  echo "Dims: M=$M, K=$K, N=$N | Threads=$THREADS | Repeats=$REPEAT | Warmups=$WARMUPS"
  echo
} > "$OUT_TXT"

# --------- SWEEP ---------
for mc in "${MC_LIST[@]}"; do
  mc_rt=$(round_to_multiple "$mc" 8)
  {
    echo "================================================================================"
    printf "MC = %-6s  (rounded -> %-6s)\n" "$mc" "$mc_rt"
    echo "================================================================================"
    echo
  } >> "$OUT_TXT"

  for kc in "${KC_LIST[@]}"; do
    (( kc > K )) && kc_rt="$K" || kc_rt="$kc"
    {
      echo "---- KC = $kc  (clamped -> $kc_rt) ---------------------------------------------"
    } >> "$OUT_TXT"

    for nc in "${NC_LIST[@]}"; do
      nc_rt=$(round_to_multiple "$nc" 48); (( nc_rt > N )) && nc_rt="$N"
      run_one "$mc_rt" "$kc_rt" "$nc_rt"
    done

    echo >> "$OUT_TXT"
  done

  echo >> "$OUT_TXT"
done

# --------- TOP-N SUMMARY ---------
{
  echo
  echo "Top 10 (by mean_ms):"
  echo "--------------------"
} >> "$OUT_TXT"

top10=$(tail -n +2 "$OUT_CSV" | LC_ALL=C sort -t',' -k4,4g | head -n 10)
nl -w2 -s'. ' <<< "$top10" >> "$OUT_TXT"

echo
echo "Top 10 (by mean_ms):"
nl -w2 -s'. ' <<< "$top10"
echo
echo "Done → $OUT_CSV, $OUT_TXT"
