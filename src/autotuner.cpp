#include "autotuner.h"

#include <cstdlib>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
  #include <sys/types.h>
  #include <sys/sysctl.h>
#elif defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #include <vector>
#else
  // Linux / Unix
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fstream>
  #include <sstream>
#endif

namespace gemm {

static constexpr int MR = 8;   // must match kernel
static constexpr int NR = 48;  // must match kernel

// ---------------- Env helpers ----------------

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
static inline size_t parse_size_str_bytes(const std::string& s) {
  std::string t(s);
  auto trim = [](std::string& x){
    while (!x.empty() && std::isspace((unsigned char)x.front())) x.erase(x.begin());
    while (!x.empty() && std::isspace((unsigned char)x.back()))  x.pop_back();
  };
  trim(t);
  if (t.empty()) return 0;
  size_t i=0;
  while (i<t.size() && (std::isdigit((unsigned char)t[i]) || t[i]=='.')) ++i;
  if (i==0) return 0;
  double v = std::atof(t.substr(0,i).c_str());
  double mult = 1.0;
  if (i<t.size()) {
    char c=t[i];
    if (c=='K'||c=='k') mult=1024.0;
    else if (c=='M'||c=='m') mult=1024.0*1024.0;
    else if (c=='G'||c=='g') mult=1024.0*1024.0*1024.0;
  }
  double bytes = v*mult;
  return bytes>0.0 ? (size_t)(bytes+0.5) : 0;
}
static inline size_t parse_size_env_bytes(const char* key, size_t dflt) {
  if (const char* s = env_cstr(key)) {
    return parse_size_str_bytes(s);
  }
  return dflt;
}
static inline int round_down_multiple(int x, int mult) {
  if (mult <= 0) return x;
  int r = x - (x % mult);
  return std::max(mult, r);
}
static inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}
static inline double ceil_div(double a, double b) {
  return std::ceil(a / b);
}

// ---------------- Platform cache detection ----------------

struct DetectedCaches {
  size_t L1d = 0;
  size_t L2  = 0;
  size_t L3  = 0;
};

#if defined(__APPLE__)
static bool detect_caches_macos(DetectedCaches& out) {
  size_t val=0; size_t len=sizeof(val);
  if (sysctlbyname("hw.l1dcachesize", &val, &len, nullptr, 0) == 0) out.L1d = val;
  len = sizeof(val);
  if (sysctlbyname("hw.l2cachesize", &val, &len, nullptr, 0) == 0) out.L2 = val;
  len = sizeof(val);
  if (sysctlbyname("hw.l3cachesize", &val, &len, nullptr, 0) == 0) out.L3 = val;
  return out.L1d || out.L2 || out.L3;
}
#endif

#if defined(_WIN32)
static bool detect_caches_windows(DetectedCaches& out) {
  DWORD len = 0;
  GetLogicalProcessorInformation(nullptr, &len);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return false;
  std::vector<uint8_t> buf(len);
  auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buf.data());
  if (!GetLogicalProcessorInformation(p, &len)) return false;

  DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
  for (DWORD i=0;i<count;++i) {
    if (p[i].Relationship != RelationCache) continue;
    CACHE_DESCRIPTOR cd = p[i].Cache;
    if (cd.Type == CacheData || cd.Type == CacheUnified) {
      if (cd.Level == 1 && out.L1d == 0) out.L1d = cd.Size;
      else if (cd.Level == 2 && out.L2 == 0) out.L2 = cd.Size;
      else if (cd.Level == 3 && out.L3 == 0) out.L3 = cd.Size;
    }
  }
  return out.L1d || out.L2 || out.L3;
}
#endif

#if !defined(__APPLE__) && !defined(_WIN32)
static bool file_exists(const std::string& p) {
  struct stat st; return ::stat(p.c_str(), &st) == 0;
}
static bool read_text_file(const std::string& p, std::string& out) {
  std::ifstream ifs(p);
  if (!ifs) return false;
  std::ostringstream ss; ss << ifs.rdbuf();
  out = ss.str();
  return true;
}
static bool detect_caches_linux(DetectedCaches& out) {
  bool found=false;
  for (int idx=0; idx<10; ++idx) {
    std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/";
    if (!file_exists(base + "level")) continue;

    std::string slevel, stype, ssize;
    if (!read_text_file(base + "level", slevel)) continue;
    if (!read_text_file(base + "type",  stype )) continue;
    if (!read_text_file(base + "size",  ssize )) continue;
    auto trim = [](std::string& x){
      while (!x.empty() && std::isspace((unsigned char)x.front())) x.erase(x.begin());
      while (!x.empty() && std::isspace((unsigned char)x.back()))  x.pop_back();
    };
    trim(stype); trim(ssize);

    int level = std::atoi(slevel.c_str());
    size_t bytes = parse_size_str_bytes(ssize);
    if (!bytes) continue;

    if (level == 1 && stype == "Data") { if (!out.L1d) out.L1d = bytes, found = true; }
    else if (level == 2 && (stype == "Data" || stype=="Unified")) { if (!out.L2) out.L2 = bytes, found = true; }
    else if (level == 3 && stype == "Unified") { if (!out.L3) out.L3 = bytes, found = true; }
  }
  return found;
}
#endif

static DetectedCaches detect_caches() {
  DetectedCaches dc;

  // Env overrides (bytes or with K/M/G)
  size_t eL1 = parse_size_env_bytes("SGEMM_L1D", 0);
  size_t eL2 = parse_size_env_bytes("SGEMM_L2",  0);
  size_t eL3 = parse_size_env_bytes("SGEMM_L3",  0);
  if (eL1) dc.L1d = eL1;
  if (eL2) dc.L2  = eL2;
  if (eL3) dc.L3  = eL3;

  if (!dc.L1d || !dc.L2 || !dc.L3) {
#if defined(__APPLE__)
    DetectedCaches tmp; if (detect_caches_macos(tmp)) {
      if (!dc.L1d && tmp.L1d) dc.L1d = tmp.L1d;
      if (!dc.L2  && tmp.L2 ) dc.L2  = tmp.L2;
      if (!dc.L3  && tmp.L3 ) dc.L3  = tmp.L3;
    }
#elif defined(_WIN32)
    DetectedCaches tmp; if (detect_caches_windows(tmp)) {
      if (!dc.L1d && tmp.L1d) dc.L1d = tmp.L1d;
      if (!dc.L2  && tmp.L2 ) dc.L2  = tmp.L2;
      if (!dc.L3  && tmp.L3 ) dc.L3  = tmp.L3;
    }
#else
    DetectedCaches tmp; if (detect_caches_linux(tmp)) {
      if (!dc.L1d && tmp.L1d) dc.L1d = tmp.L1d;
      if (!dc.L2  && tmp.L2 ) dc.L2  = tmp.L2;
      if (!dc.L3  && tmp.L3 ) dc.L3  = tmp.L3;
    }
#endif
  }

  // Fallbacks
  if (!dc.L1d) dc.L1d = 32 * 1024;
  if (!dc.L2)  dc.L2  = 1  * 1024 * 1024;  // per-core typical
  if (!dc.L3)  dc.L3  = 32 * 1024 * 1024;
  return dc;
}

// ---------------- Heuristic search ----------------

TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes) {
  // Hard overrides for tiles (exact behavior)
  if (env_cstr("SGEMM_MC") || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")) {
    TileParams t{};
    t.MC = env_int_default("SGEMM_MC", 256);
    t.KC = env_int_default("SGEMM_KC", 512);
    t.NC = env_int_default("SGEMM_NC", N);
    t.MC = clamp_int(round_down_multiple(std::max(MR, t.MC), MR), MR, std::max(MR, M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR, t.NC), NR), NR, std::max(NR, N));
    return t;
  }

  // Cache model (detected), plus tunables
  auto dc = detect_caches();
  const double alpha   = env_double_default("SGEMM_ALPHA",   1.20);  // weight C in L2 pressure (higher: penalize large MC*NC)
  const double beta    = env_double_default("SGEMM_BETA",    0.55);  // allowed fraction of L2 for A + alpha*C
  const double gamma   = env_double_default("SGEMM_GAMMA",   0.20);  // allowed fraction of (effective) LLC for B panel
  const double llc_eff = env_double_default("SGEMM_LLC_EFF", 1.00);  // share of LLC this thread can use
  const double eta_c   = env_double_default("SGEMM_C_L2_FRAC", 0.40); // direct cap on raw C tile vs L2 (write-allocate guard)

  // Base cost weights (can be tweaked at runtime)
  const double w_pk    = env_double_default("SGEMM_W_PK", 10.0); // weight for K panels (B pack cost)
  const double w_pi    = env_double_default("SGEMM_W_PI",  4.0); // weight for M panels (A pack cost)
  const double w_pj    = env_double_default("SGEMM_W_PJ",  3.0); // weight for N panels
  const double wL2     = env_double_default("SGEMM_W_L2", 1000.0);
  const double wL3     = env_double_default("SGEMM_W_L3",  200.0);

  const double L2_budget = beta * (double)dc.L2;
  const double L3_budget = gamma * (llc_eff * (double)dc.L3);

  auto rd_nr = [](int x){ return round_down_multiple(x, NR); };
  auto rd_mr = [](int x){ return round_down_multiple(x, MR); };

  // KC candidates (bounded by K)
  const int kc_cands_raw[] = {384, 512, 640, 768, 896, 1024, 1152, 1280, 1536};
  const int num_kc = (int)(sizeof(kc_cands_raw)/sizeof(kc_cands_raw[0]));

  // Helpful NC “anchors”
  auto add_nc = [&](std::vector<int>& v, int nc){
    nc = clamp_int(rd_nr(nc), NR, std::max(NR, N));
    if (std::find(v.begin(), v.end(), nc) == v.end()) v.push_back(nc);
  };

  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {256, 512, std::min(N, rd_nr(768))};

  for (int ii=0; ii<num_kc; ++ii) {
    int KC = kc_cands_raw[ii];
    if (KC > K) KC = K;
    if (KC <= 0) continue;

    // LLC-bound NC (upper) and some smaller pragmatic options
    double max_nc_llc = (L3_budget > 0.0) ? (L3_budget / ( (double)KC * dtype_bytes )) : (double)N;
    int bound_nc = clamp_int((int)std::floor(max_nc_llc), NR, std::max(NR, N));

    std::vector<int> nc_cands;
    add_nc(nc_cands, N);                 // pack whole N if feasible
    add_nc(nc_cands, bound_nc);
    add_nc(nc_cands, (int)(0.75 * bound_nc));
    add_nc(nc_cands, (int)(0.50 * bound_nc));
    // Practical sweet spots around 10–12×NR
    add_nc(nc_cands, 10*NR);  // 480
    add_nc(nc_cands, 11*NR);  // 528
    add_nc(nc_cands, 12*NR);  // 576

    for (int NC : nc_cands) {
      if (NC > N) NC = rd_nr(N);

      // Predefine MC candidates (friendly sizes) and cap by M
      int mc_pref[] = {64,80,96,112,128,160,192,224,256};
      for (int mci=0; mci<(int)(sizeof(mc_pref)/sizeof(mc_pref[0])); ++mci) {
        int MC = mc_pref[mci];
        if (MC > M) continue;
        MC = rd_mr(std::max(MR, MC));

        // Compute “pressure”
        double A_bytes = (double)MC * KC * dtype_bytes;
        double C_raw   = (double)MC * NC * dtype_bytes;
        double C_bytes = alpha * C_raw;
        double l2_ratio = (L2_budget > 0.0) ? ( (A_bytes + C_bytes) / L2_budget ) : 0.0;

        // Hard guard: avoid massive C tiles that cause write-allocate churn
        bool c_too_big = (C_raw > eta_c * (double)dc.L2);

        // L3 (B panel) pressure
        double Bpanel_bytes = (double)KC * NC * dtype_bytes;
        double l3_ratio = (L3_budget > 0.0) ? (Bpanel_bytes / L3_budget) : 0.0;

        // Panel counts (rough overhead model)
        double Pi = ceil_div((double)M, (double)MC);
        double Pj = ceil_div((double)N, (double)NC);
        double Pk = ceil_div((double)K, (double)KC);

        // Base cost: pack-B per (jc,k), pack-A per (ic,k), loop overhead per tiles
        double base = w_pk*Pk + w_pi*Pi + w_pj*Pj;

        // Penalties: emphasize L2 overuse; mild L3 overuse penalty
        double penL2 = (l2_ratio > 1.0 ? (l2_ratio - 1.0) : 0.0);
        double penL3 = (l3_ratio > 1.0 ? (l3_ratio - 1.0) : 0.0);

        // Extra penalty if C tile is too big even before weighting
        double penC  = c_too_big ? 1.0 : 0.0;

        double score = base + wL2*penL2 + wL3*penL3 + (wL2*0.25)*penC;

        if (score < best_score) {
          best_score = score;
          best.MC = MC;
          best.KC = KC;
          best.NC = NC;
        }
      }
    }
  }

  // Final sanitize
  best.MC = clamp_int(round_down_multiple(std::max(MR, best.MC), MR), MR, std::max(MR, M));
  best.KC = std::max(1, best.KC);
  best.NC = clamp_int(round_down_multiple(std::max(NR, best.NC), NR), NR, std::max(NR, N));
  return best;
}

} // namespace gemm
