// Cache-driven tiler with candidate search + heuristic scoring.
// Key idea: evaluate KC candidates; derive MC from L2, NC from L3; score by
//  (1) minimizing K-panels (C traffic) and (2) minimizing A-pack count,
//  with light tie-breaks (alignment, divisibility).

#include "autotuner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
  #define GEMM_SYSFS 1
#else
  #define GEMM_SYSFS 0
#endif

namespace gemm {

// Microkernel constants (your kernel)
static constexpr int MR = 8;
static constexpr int NR = 48;
static constexpr int VL_FLOATS = 16; // AVX-512

// ---------------- env helpers ----------------
static inline const char* env_cstr(const char* name) {
  const char* s = std::getenv(name);
  return (s && *s) ? s : nullptr;
}
static inline int env_int_default(const char* name, int dflt) {
  if (const char* s = env_cstr(name)) return std::atoi(s);
  return dflt;
}
static inline double env_float_default(const char* name, double dflt) {
  if (const char* s = env_cstr(name)) {
    char* end=nullptr; double v=std::strtod(s,&end);
    if (end && end!=s) return v;
  }
  return dflt;
}
static inline long long parse_size_string_bytes(std::string s) {
  // parse "32K", "1M", "24.8M" (we accept decimal), "512M", "8G"
  for (char& c: s) c = (char)std::toupper((unsigned char)c);
  // trim
  auto trim = [](std::string& t){
    size_t i=0; while (i<t.size() && std::isspace((unsigned char)t[i])) ++i;
    size_t j=t.size(); while (j>i && std::isspace((unsigned char)t[j-1])) --j;
    t = (i<j)? t.substr(i,j-i): std::string();
  };
  trim(s);
  if (s.empty()) return 0;
  // suffix
  double mul = 1.0;
  char last = s.back();
  if (last=='K' || last=='M' || last=='G'){
    s.pop_back();
    if (last=='K') mul = 1024.0;
    else if (last=='M') mul = 1024.0*1024.0;
    else mul = 1024.0*1024.0*1024.0;
  }
  double val = 0.0;
  try { val = std::stod(s); } catch (...) { return 0; }
  long long out = (long long) std::llround(val * mul);
  return (out>0)? out : 0;
}
static inline long long env_bytes_default(const char* name, long long dflt) {
  if (const char* s = env_cstr(name)) {
    long long v = parse_size_string_bytes(s);
    if (v>0) return v;
  }
  return dflt;
}

// ---------------- cache detect (Linux) ----------------
struct CacheInfo {
  long long l1d_bytes  = 32 * 1024;         // per-core
  long long l2_bytes   = 1LL * 1024 * 1024; // per-core
  long long l3_bytes   = 8LL * 1024 * 1024; // shared
  int       line_bytes = 64;
};

#if GEMM_SYSFS
static inline bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path);
  if (!f.good()) return false;
  std::ostringstream ss; ss<<f.rdbuf(); out=ss.str(); return true;
}
static inline long long parse_sysfs_size(const std::string& s) {
  return parse_size_string_bytes(s);
}
static bool detect_cache_linux(CacheInfo& ci) {
  const char* base="/sys/devices/system/cpu/cpu0/cache";
  bool ok=false;
  for (int idx=0; idx<8; ++idx) {
    std::string p = std::string(base)+"/index"+std::to_string(idx)+"/";
    std::string lvlS, typeS, sizeS, lineS;
    if (!read_file(p+"level", lvlS)) continue;
    if (!read_file(p+"type" , typeS)) continue;
    if (!read_file(p+"size" , sizeS)) continue;
    read_file(p+"coherency_line_size", lineS);
    int lvl = std::atoi(lvlS.c_str());
    for (auto& c: typeS) c = (char)std::toupper((unsigned char)c);
    long long sz = parse_sysfs_size(sizeS);
    if (lvl==1 && typeS.find("DATA")!=std::string::npos) {
      if (sz>1024) ci.l1d_bytes = sz;
      if (!lineS.empty()) ci.line_bytes = std::max(32, std::atoi(lineS.c_str()));
      ok=true;
    } else if (lvl==2) {
      if (sz>4096) ci.l2_bytes = sz;
      if (!lineS.empty()) ci.line_bytes = std::max(32, std::atoi(lineS.c_str()));
      ok=true;
    } else if (lvl==3 && typeS.find("UNIFIED")!=std::string::npos) {
      if (sz>4096) ci.l3_bytes = sz;
      if (!lineS.empty()) ci.line_bytes = std::max(32, std::atoi(lineS.c_str()));
      ok=true;
    }
  }
  return ok;
}
#endif

// ---------------- utilities ----------------
static inline int round_down(int x, int m){ return (m<=1)? x : (x/m)*m; }
static inline int round_up  (int x, int m){ return (m<=1)? x : ((x+m-1)/m)*m; }
static inline int clampi(int v, int lo, int hi){ return std::max(lo, std::min(hi, v)); }

// ---------------- main picker ----------------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes) {
  // 0) hard overrides
  if (const char* s = env_cstr("SGEMM_MC"); s || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")) {
    int MC = s ? std::max(MR, std::atoi(s)) : 256;
    int KC = env_int_default("SGEMM_KC", 512);
    int NC = env_int_default("SGEMM_NC", N);
    MC = round_down(std::max(MR, MC), MR);
    KC = std::max(1, KC);
    NC = round_down(clampi(NC, NR, N), NR);
    return {MC, KC, NC};
  }

  // 1) caches
  CacheInfo ci;
#if GEMM_SYSFS
  detect_cache_linux(ci); // best-effort
#endif
  ci.l1d_bytes = env_bytes_default("SGEMM_L1D", ci.l1d_bytes);
  ci.l2_bytes  = env_bytes_default("SGEMM_L2" , ci.l2_bytes);
  ci.l3_bytes  = env_bytes_default("SGEMM_L3" , ci.l3_bytes);

  // 2) knobs
  const double beta   = env_float_default("SGEMM_BETA" , 0.60);
  const double gamma  = env_float_default("SGEMM_GAMMA", 0.70);
  const double eta    = env_float_default("SGEMM_LLC_EFF", 0.60);

  const int MC_MIN    = round_up(std::max(MR, env_int_default("SGEMM_MC_MIN", 64)), MR);
  const int MC_CAP    = round_down(std::max(MC_MIN, env_int_default("SGEMM_MC_CAP", 256)), MR); // soft cap
  const int KC_MIN    = round_up(std::max(64, env_int_default("SGEMM_KC_MIN", 256)), 64);
  const int KC_MAX    = round_down(std::max(KC_MIN, env_int_default("SGEMM_KC_MAX", 2048)), 64);

  // 3) candidate KC grid (64-step, prefer 256..2048; clamp by K)
  std::vector<int> KCcands;
  int KC_hi = std::min(KC_MAX, std::max(64, K));
  for (int kc = KC_MIN; kc <= KC_hi; kc += 64) KCcands.push_back(kc);

  // 4) scoring weights (heuristic; tweak via env if needed)
  const double W_KPANELS    = env_float_default("SGEMM_W_KPANELS", 4.0); // fewer K panels is big win
  const double W_APACKS     = env_float_default("SGEMM_W_APACKS" , 1.2); // fewer A packs preferred
  const double W_KC_ALIGN   = env_float_default("SGEMM_W_KC_ALIGN", 0.15); // small bonus for kc%512==0

  // 5) evaluate candidates
  struct Choice { int KC, MC, NC; double score; };
  Choice best{0,0,0,1e300};

  const double L2_budget = beta * (double)ci.l2_bytes;
  const double L3_budget = gamma * eta * (double)ci.l3_bytes;

  for (int KC : KCcands) {
    if (KC <= 0) continue;
    // derive MC from L2: MC <= floor( L2_budget / (b*KC) )
    long long mc_ll = (long long)(L2_budget / ( (double)dtype_bytes * (double)KC ));
    int MC = (int)mc_ll;
    MC = round_down(std::max(MC_MIN, MC), MR);
    if (MC < MC_MIN) continue;           // too deep for L2 at this KC

    // optional soft cap on MC to avoid giant L2 working sets / conflict risk
    MC = std::min(MC, MC_CAP);

    // derive NC from L3: KC*NC*b <= L3_budget
    long long nc_ll = (long long)( L3_budget / ( (double)dtype_bytes * (double)KC ) );
    int NC = (int)nc_ll;
    NC = round_down(clampi(std::max(NR, NC), NR, N), NR);
    if (NC < NR) continue;

    // for very skinny N, pack all
    if (N <= 2*NR) NC = N;

    // compute K-panels and A-pack count
    const int Kpanels  = (K + KC - 1) / KC;
    const int Apacks   = ( (M + MC - 1) / MC ) * Kpanels;

    // score: lower is better
    double score = 0.0;
    score += W_KPANELS * (double)Kpanels;
    score += W_APACKS  * (double)Apacks;

    // small alignment bonus if KC nicely divisible (e.g., 512)
    if (KC % 512 == 0) score -= W_KC_ALIGN;

    // keep the best
    if (score < best.score) best = {KC, MC, NC, score};
  }

  // Fallback if nothing passed (shouldn't happen): use conservative defaults
  if (best.KC == 0) {
    int KC = clampi(round_down(std::min(K, 512), 64), 64, K);
    int MC = round_down(std::max(MC_MIN, (int)( (beta*ci.l2_bytes) / (double)(dtype_bytes*KC) )), MR);
    MC = std::min(MC, MC_CAP);
    int NC = round_down(clampi((int)( (gamma*eta*ci.l3_bytes) / (double)(dtype_bytes*KC)), NR, N), NR);
    if (N <= 2*NR) NC = N;
    best = {KC, std::max(MR,MC), std::max(NR,NC), 0.0};
  }

  if (env_int_default("SGEMM_DEBUG", 0)) {
    std::cerr << "[TILER] L1d=" << ci.l1d_bytes
              << " L2=" << ci.l2_bytes
              << " L3=" << ci.l3_bytes
              << "  -> KC=" << best.KC
              << " MC=" << best.MC
              << " NC=" << best.NC
              << " score=" << best.score << "\n";
  }
  return {best.MC, best.KC, best.NC};
}

} // namespace gemm
