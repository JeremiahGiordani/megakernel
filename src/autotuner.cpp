#include "autotuner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace gemm {

static constexpr int MR = 8;   // must match kernel
static constexpr int NR = 48;  // must match kernel

// ---------- helpers ----------
static inline const char* env_cstr(const char* k){ const char* s=getenv(k); return (s && *s)? s: nullptr; }
static inline int env_int(const char* k, int d){ if(const char* s=env_cstr(k)) return std::atoi(s); return d; }
static inline double env_double(const char* k, double d){ if(const char* s=env_cstr(k)) return std::atof(s); return d; }

static inline int round_down_multiple(int x, int m){ if(m<=0) return x; int r=x-(x%m); return std::max(m,r); }
static inline int clamp_int(int v, int lo, int hi){ return std::max(lo, std::min(hi, v)); }

static inline bool read_file(const std::string& p, std::string& out){
  std::ifstream ifs(p); if(!ifs) return false; std::ostringstream ss; ss<<ifs.rdbuf(); out=ss.str(); return true;
}
static inline void trim(std::string& s){
  while(!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
  while(!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
}
static inline size_t parse_size_token(const std::string& s){
  if(s.empty()) return 0;
  size_t i=0; while(i<s.size() && (std::isdigit((unsigned char)s[i]) || s[i]=='.')) ++i;
  if(i==0) return 0;
  double v = std::atof(s.substr(0,i).c_str());
  double m = 1.0;
  if(i<s.size()){
    char c=s[i];
    if(c=='K'||c=='k') m=1024.0;
    else if(c=='M'||c=='m') m=1024.0*1024.0;
    else if(c=='G'||c=='g') m=1024.0*1024.0*1024.0;
  }
  double bytes=v*m; return bytes>0.0? (size_t)(bytes+0.5):0;
}

// Linux sysfs cache detection for cpu0
struct Caches { size_t L1d=0, L2=0, L3=0; };
static Caches detect_caches_linux() {
  Caches c{};
  // L1d (level 1, Data)
  // L2 (level 2, Data/Unified)
  // L3 (level 3, Unified)
  for(int idx=0; idx<10; ++idx){
    std::string base="/sys/devices/system/cpu/cpu0/cache/index"+std::to_string(idx)+"/";
    std::string slev, styp, ssiz;
    if(!read_file(base+"level", slev)) continue;
    if(!read_file(base+"type",  styp)) continue;
    if(!read_file(base+"size",  ssiz)) continue;
    trim(slev); trim(styp); trim(ssiz);
    int level = std::atoi(slev.c_str());
    size_t bytes = parse_size_token(ssiz);
    if(!bytes) continue;
    if(level==1 && styp=="Data")         { c.L1d = bytes; }
    else if(level==2 && (styp=="Data" || styp=="Unified")) { c.L2 = bytes; }
    else if(level==3 && styp=="Unified") { c.L3 = bytes; }
  }
  // Conservative fallbacks
  if(!c.L1d) c.L1d = 32*1024;
  if(!c.L2)  c.L2  = 1u*1024u*1024u;   // ~1 MiB per core typical
  if(!c.L3)  c.L3  = 32u*1024u*1024u;  // ~32 MiB socket
  return c;
}

// ---------- the heuristic ----------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes){
  // Hard overrides (same behavior you had)
  if(env_cstr("SGEMM_MC") || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")){
    TileParams t{ env_int("SGEMM_MC",256), env_int("SGEMM_KC",512), env_int("SGEMM_NC",N) };
    t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
    return t;
  }

  // Cache + threading info
  Caches c = detect_caches_linux();
  const int threads = env_int("OMP_NUM_THREADS", 1);

  // Tunable constants (picked to match your two boxes at 8T)
  const double ALPHA = env_double("SGEMM_ALPHA", 0.60);  // C write-alloc weight in L2 guard
  const double BETA  = env_double("SGEMM_BETA",  0.75);  // fraction of L2 allowed for A + ALPHA*C
  const double GAMMA = env_double("SGEMM_GAMMA", 0.40);  // fraction of per-thread LLC allowed for B panel

  const double L2_target = BETA * (double)c.L2;
  const double L3_per_thread = (threads>0) ? ((double)c.L3 / (double)threads) : (double)c.L3;

  // KC candidates (clamped to K)
  std::vector<int> kc_cands = {512, 768, 1024, 1536, 2048, 3072};
  for(int& v : kc_cands){ if(v > K) v = K; }
  kc_cands.erase(std::remove_if(kc_cands.begin(), kc_cands.end(), [](int x){ return x<=0; }), kc_cands.end());

  // Preferred NC values (multiples of NR)
  std::vector<int> nc_pref = { 336, 480, 528, 576, 672, 1008, 1152, 1344 };

  // MC candidates (largest first)
  std::vector<int> mc_cands = {128, 96, 64};

  // Scoring weights: light, just to break ties sensibly
  const double W_KPAN = 8.0;  // fewer K panels (pack B less) is good
  const double W_NTIL = 3.0;  // fewer N tiles is good
  const double W_MTIL = 2.0;  // fewer M tiles is good
  const double W_L2   = 2.0;  // small L2 pressure bias

  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {64, 512, std::min(N, 576)};

  for(int KC : kc_cands){
    // LLC bound for NC (bytes: KC*NC*dtype <= GAMMA * L3_per_thread)
    double max_nc_llc = (GAMMA * L3_per_thread) / ( (double)KC * (double)dtype_bytes );
    int nc_llc = (int)std::floor(max_nc_llc);

    // Build NC candidate list for this KC
    std::vector<int> nc_cands;
    for(int nc : nc_pref){
      if(nc <= N && nc <= nc_llc) nc_cands.push_back(nc);
    }
    // Ensure we have *something*: if none pass the LLC bound, take the largest NR-multiple that does.
    if(nc_cands.empty()){
      int nc = round_down_multiple(std::max(NR, std::min(N, nc_llc)), NR);
      if(nc >= NR) nc_cands.push_back(nc);
    }

    for(int NC : nc_cands){
      // L2 guard: pick largest MC in {128,96,64} that satisfies
      // (MC*(KC + ALPHA*NC))*dtype <= L2_target
      int chosen_MC = MR;
      for(int MC : mc_cands){
        double bytes = (double)MC * ((double)KC + ALPHA*(double)NC) * (double)dtype_bytes;
        if(bytes <= L2_target){ chosen_MC = MC; break; }
      }
      if(chosen_MC < 64) continue; // too much pressure; skip this KC/NC

      // Tile counts
      auto ceil_div = [](int a, int b){ return (a + b - 1)/b; };
      int Pi = ceil_div(std::max(MR, M), std::max(MR, chosen_MC));
      int Pj = ceil_div(std::max(NR, N), std::max(NR, NC));
      int Pk = ceil_div(std::max(1,  K), std::max(1,  KC));

      // Small bias against large MC near L2 limit (keeps MC in 64–96 region if similar)
      double A_bytes = (double)chosen_MC * (double)KC * (double)dtype_bytes;
      double C_bytes = (double)chosen_MC * (double)NC * (double)dtype_bytes;
      double l2_bias = (A_bytes + 0.5*C_bytes) / std::max(1.0, (double)c.L2);

      double score = W_KPAN*Pk + W_NTIL*Pj + W_MTIL*Pi + W_L2*l2_bias;

      // Light tie-break to prefer "nice" NCs you liked (528/576/1008/1152)
      if(NC==528 || NC==576 || NC==1008 || NC==1152) score *= 0.995;

      if(score < best_score){
        best_score = score;
        best.MC = chosen_MC;
        best.KC = KC;
        best.NC = NC;
      }
    }
  }

  // Final sanitize and clamp to problem size / MR/NR
  best.MC = clamp_int(round_down_multiple(std::max(MR, best.MC), MR), MR, std::max(MR, M));
  best.KC = std::max(1, std::min(best.KC, K));
  best.NC = clamp_int(round_down_multiple(std::max(NR, best.NC), NR), NR, std::max(NR, N));
  return best;
}

} // namespace gemm
