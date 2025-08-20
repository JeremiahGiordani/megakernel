// AVX-512 SGEMM tiler tuned to avoid tiny KC (too many K-panels).
#include "autotuner.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#else
  static inline int omp_get_max_threads(){ return 1; }
#endif

namespace gemm {

// Microkernel facts (your kernel)
static constexpr int MR = 8;
static constexpr int NR = 48;
static constexpr int VL_FLOATS = 16; // AVX-512 (zmm) floats

// ---------------- env helpers ----------------
static inline const char* env_cstr(const char* name) {
  const char* s = std::getenv(name);
  return s && *s ? s : nullptr;
}
static inline int env_int_default(const char* name, int dflt) {
  if (const char* s = env_cstr(name)) return std::atoi(s);
  return dflt;
}
static inline double env_float_default(const char* name, double dflt) {
  if (const char* s = env_cstr(name)) {
    char* end=nullptr; double v = std::strtod(s,&end);
    if (end && end!=s) return v;
  }
  return dflt;
}
static inline long long parse_size_string_bytes(const std::string& s) {
  std::string t; t.reserve(s.size());
  for (char c: s) t.push_back(std::toupper(static_cast<unsigned char>(c)));
  size_t i=0; while (i<t.size() && std::isspace((unsigned char)t[i])) ++i;
  size_t j=t.size(); while (j>i && std::isspace((unsigned char)t[j-1])) --j;
  if (i>=j) return 0;
  t = t.substr(i, j-i);
  long long mul=1;
  if (!t.empty()) {
    char last=t.back();
    if (last=='K'){ mul=1024LL; t.pop_back(); }
    else if (last=='M'){ mul=1024LL*1024LL; t.pop_back(); }
    else if (last=='G'){ mul=1024LL*1024LL*1024LL; t.pop_back(); }
  }
  long long val=0; std::istringstream iss(t); iss>>val;
  return iss.fail()? 0 : val*mul;
}
static inline long long env_bytes_default(const char* name, long long dflt) {
  if (const char* s = env_cstr(name)) {
    long long v = parse_size_string_bytes(s);
    if (v>0) return v;
  }
  return dflt;
}

// ---------------- cache detect (Linux best-effort) ----------------
struct CacheInfo {
  long long l1d_bytes  = 32 * 1024;         // per-core default
  long long l2_bytes   = 1LL * 1024 * 1024; // per-core default
  long long l3_bytes   = 8LL * 1024 * 1024; // shared default
  int       line_bytes = 64;
};

static inline bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path);
  if (!f.good()) return false;
  std::ostringstream ss; ss<<f.rdbuf(); out=ss.str();
  return true;
}
static inline long long parse_sysfs_size(const std::string& s) {
  return parse_size_string_bytes(s);
}
static bool detect_cache_linux(CacheInfo& ci) {
#ifdef __linux__
  const char* base="/sys/devices/system/cpu/cpu0/cache";
  bool ok=false;
  for (int idx=0; idx<8; ++idx){
    std::string p = std::string(base)+"/index"+std::to_string(idx)+"/";
    std::string sLvl,sType,sSize,sLine;
    if (!read_file(p+"level", sLvl)) continue;
    if (!read_file(p+"type" , sType)) continue;
    if (!read_file(p+"size" , sSize)) continue;
    read_file(p+"coherency_line_size", sLine);
    int lvl = std::atoi(sLvl.c_str());
    for (auto& c: sType) c = (char)std::toupper((unsigned char)c);
    long long sz = parse_sysfs_size(sSize);
    if (lvl==1 && sType.find("DATA")!=std::string::npos) {
      if (sz>1024) ci.l1d_bytes = sz;
      if (!sLine.empty()) ci.line_bytes = std::max(32, std::atoi(sLine.c_str()));
      ok=true;
    } else if (lvl==2) {
      if (sz>4096) ci.l2_bytes = sz;
      if (!sLine.empty()) ci.line_bytes = std::max(32, std::atoi(sLine.c_str()));
      ok=true;
    } else if (lvl==3 && sType.find("UNIFIED")!=std::string::npos) {
      if (sz>4096) ci.l3_bytes = sz;
      if (!sLine.empty()) ci.line_bytes = std::max(32, std::atoi(sLine.c_str()));
      ok=true;
    }
  }
  return ok;
#else
  (void)ci; return false;
#endif
}

// ---------------- helpers ----------------
static inline int round_down_multiple(int x, int m){
  if (m<=1) return x; return (x/m)*m;
}
static inline int round_up_multiple(int x, int m){
  if (m<=1) return x; return ((x+m-1)/m)*m;
}
static inline int clampi(int v, int lo, int hi){
  return std::max(lo, std::min(hi, v));
}

// ---------------- main picker ----------------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes) {
  // 0) Hard overrides (keep your original behavior)
  const char* sMC = env_cstr("SGEMM_MC");
  const char* sKC = env_cstr("SGEMM_KC");
  const char* sNC = env_cstr("SGEMM_NC");
  if (sMC || sKC || sNC){
    TileParams t{
      sMC ? std::max(MR, std::atoi(sMC)) : 256,
      sKC ? std::max(1 , std::atoi(sKC)) : 512,
      sNC ? clampi(std::max(NR, std::atoi(sNC)), NR, N) : N
    };
    t.MC = round_down_multiple(std::max(MR, t.MC), MR);
    t.KC = std::max(1, t.KC);
    t.NC = round_down_multiple(clampi(t.NC, NR, N), NR);
    return t;
  }

  // 1) Detect caches (best-effort) and allow env overrides
  CacheInfo ci;
  detect_cache_linux(ci);
  ci.l1d_bytes = env_bytes_default("SGEMM_L1D", ci.l1d_bytes);
  ci.l2_bytes  = env_bytes_default("SGEMM_L2" , ci.l2_bytes);
  ci.l3_bytes  = env_bytes_default("SGEMM_L3" , ci.l3_bytes);

  // 2) Tunables
  const int    K_PANELS = std::max(2, env_int_default("SGEMM_K_PANELS", 8)); // aim for ~8 K-panels
  const int    KC_MIN   = std::max(2*VL_FLOATS, env_int_default("SGEMM_KC_MIN", 256)); // >=32, default 256
  const int    KC_MAX   = env_int_default("SGEMM_KC_MAX", 1024);
  const int    MC_MIN   = round_up_multiple(std::max(MR, env_int_default("SGEMM_MC_MIN", 64)), MR);
  const double beta     = env_float_default("SGEMM_BETA" , 0.60); // A-pack fraction of L2
  const double gamma    = env_float_default("SGEMM_GAMMA", 0.70); // B-panel fraction of LLC
  const double eta      = env_float_default("SGEMM_LLC_EFF", 0.60); // reserve some LLC

  // 3) Choose KC by "target number of K-panels" (avoid tiny KC)
  int KC = round_down_multiple(std::max(1, (K + K_PANELS - 1) / K_PANELS), 2*VL_FLOATS); // multiple of 32
  KC = clampi(KC, KC_MIN, std::min(KC_MAX, std::max(1, K)));

  // 4) Back-solve MC from L2 so A_pack fits beta * L2:
  //    bytes(A_{MC×KC}) = MC*KC*dtype <= beta * L2  => MC_max = floor(beta*L2 / (b*KC))
  long long l2_budget = (long long)(beta * (double)ci.l2_bytes);
  long long mc_ll = (KC>0) ? (l2_budget / ( (long long)dtype_bytes * (long long)KC )) : MR;
  int MC = (int)mc_ll;
  MC = round_down_multiple(std::max(MC_MIN, MC), MR);
  MC = clampi(MC, MR, std::max(MR, M));

  // If MC collapsed because KC is too large, lower KC until MC >= MC_MIN.
  while (MC < MC_MIN && KC > KC_MIN) {
    // reduce KC by one AVX-512 "chunk" and recompute MC
    KC = std::max(KC_MIN, KC - 2*VL_FLOATS);
    mc_ll = (KC>0) ? (l2_budget / ((long long)dtype_bytes * (long long)KC)) : MR;
    MC = round_down_multiple(std::max(MC_MIN, (int)mc_ll), MR);
  }

  // 5) Choose NC from LLC (team-shared B panel). We don't divide by threads because the
  //     B-panel is shared; the working set is roughly panel-sized at any moment.
  //    bytes(B_{KC×NC}) <= gamma * eta * L3  => NC_max = floor(gamma*eta*L3 / (b*KC))
  double llc_budget = gamma * eta * (double)ci.l3_bytes;
  long long nc_ll = (KC>0) ? (long long)(llc_budget / ( (double)dtype_bytes * (double)KC)) : NR;
  int NC = (int)nc_ll;
  NC = round_down_multiple(std::max(NR, NC), NR);
  NC = clampi(NC, NR, std::max(NR, N));

  // For very skinny-N, prefer packing all of N to avoid extra jc loops.
  if (N <= 2*NR) NC = N;

  // 6) Final clamps against problem shape
  KC = clampi(KC, 1, std::max(1, K));
  MC = clampi(MC, MR, std::max(MR, M));
  NC = clampi(NC, NR, std::max(NR, N));

  return TileParams{MC, KC, NC};
}

} // namespace gemm
