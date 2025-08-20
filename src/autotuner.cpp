#include "autotuner.h"

#include <cstdlib>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

namespace gemm {

static constexpr int MR = 8;   // must match kernel
static constexpr int NR = 48;  // must match kernel

// ----------- env helpers -----------

static inline const char* env_cstr(const char* key) {
  const char* v = std::getenv(key);
  return v && *v ? v : nullptr;
}

static inline int env_int_default(const char* key, int dflt) {
  if (const char* s = env_cstr(key)) return std::atoi(s);
  return dflt;
}

static inline double env_double_default(const char* key, double dflt) {
  if (const char* s = env_cstr(key)) return std::atof(s);
  return dflt;
}

// Parse sizes like: "1048576" or "256K" or "1M" or "24.5M" or "0.5G"
static inline size_t parse_size_env_bytes(const char* key, size_t dflt) {
  const char* s = env_cstr(key);
  if (!s) return dflt;

  // Trim leading spaces
  while (*s == ' ' || *s == '\t') ++s;

  // Copy to string until non [0-9 . a-z A-Z]
  std::string t(s);
  // strip trailing spaces
  while (!t.empty() && (t.back()==' ' || t.back()=='\t')) t.pop_back();
  if (t.empty()) return dflt;

  // Find unit suffix
  double value = 0.0;
  size_t i = 0;
  // parse number part
  while (i < t.size() && (std::isdigit(static_cast<unsigned char>(t[i])) || t[i]=='.')) ++i;
  if (i == 0) return dflt;

  value = std::atof(t.substr(0, i).c_str());

  // parse optional suffix
  double mult = 1.0;
  if (i < t.size()) {
    char c = t[i];
    if (c=='K' || c=='k') mult = 1024.0;
    else if (c=='M' || c=='m') mult = 1024.0*1024.0;
    else if (c=='G' || c=='g') mult = 1024.0*1024.0*1024.0;
  }
  double bytes_f = value * mult;
  if (bytes_f < 0.0) bytes_f = 0.0;
  return static_cast<size_t>(bytes_f + 0.5);
}

static inline int round_down_multiple(int x, int mult) {
  if (mult <= 0) return x;
  int r = x - (x % mult);
  return std::max(mult, r);
}

static inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

// ----------- scoring heuristic -----------
//
// We want to keep per-thread L2 pressure low and per-thread LLC panel reasonable.
// Define a light-weight score ~ number of panels processed + cache pressure penalties.
//
// score = Pk*Pi*Pj
//       + wL2 * max(0, (A+C)/L2_frac - 1)
//       + wL3 * max(0,  Bpanel/LLC_frac - 1)
//
// Where:
//  A = MC*KC*dtype
//  C = alpha * MC*NC*dtype  (alpha accounts for write-allocate effect)
//  L2_frac = beta * L2_bytes
//  Bpanel = KC*NC*dtype
//  LLC_frac = gamma * (LLC_eff * L3_bytes)
//
// The panel counts:
//  Pi = ceil(M/MC), Pj = ceil(N/NC), Pk = ceil(K/KC)
//
// We also enforce MC%MR==0, NC%NR==0.

struct CacheModel {
  size_t L1_bytes;
  size_t L2_bytes;
  size_t L3_bytes;
  double alpha;     // C weighting in L2
  double beta;      // fraction of L2 budget we allow
  double gamma;     // fraction of LLC budget we allow
  double llc_eff;   // effective share of LLC
};

static inline double ceil_div(double a, double b) {
  return std::ceil(a / b);
}

TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes) {
  // 1) Env overrides for tiles (exact)
  if (const char* sMC = env_cstr("SGEMM_MC");
      sMC || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")) {
    TileParams t{};
    t.MC = env_int_default("SGEMM_MC", 256);
    t.KC = env_int_default("SGEMM_KC", 512);
    t.NC = env_int_default("SGEMM_NC", N);
    // sanitize
    t.MC = clamp_int(round_down_multiple(std::max(MR, t.MC), MR), MR, std::max(MR, M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR, t.NC), NR), NR, std::max(NR, N));
    return t;
  }

  // 2) Cache model from env (or defaults)
  CacheModel cm;
  cm.L1_bytes = parse_size_env_bytes("SGEMM_L1D", 32 * 1024);          // default 32 KiB
  cm.L2_bytes = parse_size_env_bytes("SGEMM_L2",  1  * 1024 * 1024);   // default 1 MiB per core
  cm.L3_bytes = parse_size_env_bytes("SGEMM_L3",  32 * 1024 * 1024);   // default 32 MiB LLC

  cm.alpha   = env_double_default("SGEMM_ALPHA",   0.50);  // weight of C in L2 pressure
  cm.beta    = env_double_default("SGEMM_BETA",    0.55);  // allowed fraction of L2 for A + alpha*C
  cm.gamma   = env_double_default("SGEMM_GAMMA",   0.20);  // allowed fraction of (effective) LLC for B panel
  cm.llc_eff = env_double_default("SGEMM_LLC_EFF", 1.00);  // portion of LLC we can really use

  // 3) Candidate KC values to test (multiples of 16, friendly to prefetch distances)
  //    Include larger options as you observed KC↑ helping on your box.
  const int kc_cands_raw[] = {384, 512, 640, 768, 896, 1024, 1152, 1280, 1536};
  constexpr int num_kc = sizeof(kc_cands_raw)/sizeof(kc_cands_raw[0]);

  // 4) For each KC candidate, compute NC (LLC-limited) and MC (L2-limited),
  //    then score and pick the best.
  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {256, 512, std::min(N, 768)};

  auto round_down_NR = [](int x){ return round_down_multiple(x, NR); };
  auto round_down_MR = [](int x){ return round_down_multiple(x, MR); };

  // Helper: compute NC bound from LLC budget
  auto compute_nc_llc_bound = [&](int KC){
    const double llc_budget = cm.gamma * (cm.llc_eff * static_cast<double>(cm.L3_bytes));
    // KC * NC * dtype_bytes <= llc_budget
    double max_nc = llc_budget / (static_cast<double>(KC) * dtype_bytes);
    int nc = static_cast<int>(std::floor(max_nc));
    // Keep at least one NR
    nc = std::max(NR, round_down_NR(nc));
    return nc;
  };

  // Helper: compute MC bound from L2 budget
  auto compute_mc_l2_bound = [&](int KC, int NC){
    // (MC*KC + alpha*MC*NC) * dtype_bytes <= beta * L2
    // => MC <= beta*L2 / (dtype * (KC + alpha*NC))
    double denom = static_cast<double>(KC) + cm.alpha * static_cast<double>(NC);
    if (denom <= 0.0) return MR;
    double mc_max = (cm.beta * static_cast<double>(cm.L2_bytes)) /
                    (static_cast<double>(dtype_bytes) * denom);
    int mc = static_cast<int>(std::floor(mc_max));
    mc = std::max(MR, round_down_MR(mc));
    return mc;
  };

  // Simple penalty weights (tuned lightly)
  const double wL2 = 500.0;
  const double wL3 = 500.0;

  for (int i = 0; i < num_kc; ++i) {
    int KC = kc_cands_raw[i];
    if (KC <= 0) continue;
    if (KC > K)  KC = K; // cap to problem

    // Compute NC target from LLC; also clamp to N and keep it "healthy"
    int nc_llc = compute_nc_llc_bound(KC);
    int NC = std::min(N, nc_llc);

    // If N is small, don't over-tile
    if (NC > N) NC = N;
    if (NC < NR) NC = NR;

    // Prefer using as much N as allowed by LLC (good for B-pack amortization)
    NC = std::min(N, round_down_NR(NC));

    // Now compute MC from L2 budget
    int MC = compute_mc_l2_bound(KC, NC);
    if (MC > M) MC = round_down_MR(M);
    if (MC < MR) MC = MR;

    // If MC is implausibly tiny (<=MR), consider shrinking NC a notch to give MC room
    // Try a very small reduction on NC (one NR step) if MC is too tiny
    if (MC == MR && NC > NR) {
      int NC_try = std::max(NR, NC - NR);
      int MC_try = compute_mc_l2_bound(KC, NC_try);
      if (MC_try > MC) { NC = NC_try; MC = MC_try; }
    }

    // Panel counts
    double Pi = ceil_div(static_cast<double>(M), static_cast<double>(MC));
    double Pj = ceil_div(static_cast<double>(N), static_cast<double>(NC));
    double Pk = ceil_div(static_cast<double>(K), static_cast<double>(KC));

    // Cache pressure terms
    double A_bytes = static_cast<double>(MC) * KC * dtype_bytes;
    double C_bytes = cm.alpha * static_cast<double>(MC) * NC * dtype_bytes;
    double Bpanel_bytes = static_cast<double>(KC) * NC * dtype_bytes;

    double L2_budget = cm.beta * static_cast<double>(cm.L2_bytes);
    double L3_budget = cm.gamma * (cm.llc_eff * static_cast<double>(cm.L3_bytes));

    double l2_over = (A_bytes + C_bytes) / std::max(1.0, L2_budget);
    double l3_over = Bpanel_bytes / std::max(1.0, L3_budget);

    double penL2 = (l2_over > 1.0) ? (l2_over - 1.0) : 0.0;
    double penL3 = (l3_over > 1.0) ? (l3_over - 1.0) : 0.0;

    // Base score: number of panel triplets we execute
    double base = Pk * Pi * Pj;

    // Light bias: prefer larger KC (better pack amortization) and reasonable NC
    // but that is already implicit; keep the scoring simple.
    double score = base + wL2 * penL2 + wL3 * penL3;

    // Keep the best
    if (score < best_score) {
      best_score = score;
      best.MC = MC;
      best.KC = KC;
      best.NC = NC;
    }
  }

  // Final sanitization & caps
  best.MC = clamp_int(round_down_multiple(std::max(MR, best.MC), MR), MR, std::max(MR, M));
  best.KC = std::max(1, best.KC);
  best.NC = clamp_int(round_down_multiple(std::max(NR, best.NC), NR), NR, std::max(NR, N));
  return best;
}

} // namespace gemm
