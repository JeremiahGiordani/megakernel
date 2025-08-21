#include "autotuner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace gemm {

static constexpr int MR = 8;   // must match your microkernel
static constexpr int NR = 48;  // must match your microkernel

// --------------------- small utils ---------------------
static inline const char* env_cstr(const char* k){ const char* s=getenv(k); return (s&&*s)? s:nullptr; }
static inline int env_int(const char* k, int d){ if(const char* s=env_cstr(k)) return std::atoi(s); return d; }
static inline double env_double(const char* k, double d){ if(const char* s=env_cstr(k)) return std::atof(s); return d; }

static inline int round_down_multiple(int x, int m){ if(m<=0) return x; int r=x-(x%m); return std::max(m,r); }
static inline int clamp_int(int v, int lo, int hi){ return std::max(lo, std::min(hi, v)); }

static inline bool read_text(const std::string& p, std::string& out){
  std::ifstream ifs(p); if(!ifs) return false;
  std::ostringstream ss; ss<<ifs.rdbuf(); out=ss.str(); return true;
}
static inline void trim(std::string& s){
  while(!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
  while(!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
}
static inline size_t parse_size_token(const std::string& s){
  std::string t=s; trim(t); if(t.empty()) return 0;
  size_t i=0; while(i<t.size() && (std::isdigit((unsigned char)t[i]) || t[i]=='.')) ++i;
  if(i==0) return 0;
  double v = std::atof(t.substr(0,i).c_str());
  double m = 1.0;
  if(i<t.size()){
    char c=t[i];
    if(c=='K'||c=='k') m=1024.0;
    else if(c=='M'||c=='m') m=1024.0*1024.0;
    else if(c=='G'||c=='g') m=1024.0*1024.0*1024.0;
  }
  double bytes=v*m; return bytes>0.0? (size_t)(bytes+0.5):0;
}

// --------------------- Linux cache detection ---------------------
struct Caches { size_t L1d=0, L2_per_core=0, L3_total=0; };

static std::vector<std::string> list_dir(const std::string& path){
  std::vector<std::string> names;
  DIR* d = opendir(path.c_str()); if(!d) return names;
  if (dirent* ent; true) {
    while((ent=readdir(d))){
      if(ent->d_name[0]=='.') continue;
      names.emplace_back(ent->d_name);
    }
  }
  closedir(d);
  return names;
}

static Caches detect_caches_linux(){
  Caches c{};

  // L1d/L2 from cpu0
  {
    std::string base="/sys/devices/system/cpu/cpu0/cache/";
    for(const auto& idx : list_dir(base)){
      std::string p = base + idx + "/";
      std::string slev, styp, ssiz;
      if(!read_text(p+"level", slev)) continue;
      if(!read_text(p+"type",  styp)) continue;
      if(!read_text(p+"size",  ssiz)) continue;
      trim(slev); trim(styp); trim(ssiz);
      int level = std::atoi(slev.c_str());
      size_t bytes = parse_size_token(ssiz);
      if(level==1 && styp=="Data") c.L1d = bytes;
      if(level==2 && (styp=="Data" || styp=="Unified")) c.L2_per_core = bytes;
    }
    if(!c.L1d) c.L1d = 32*1024;
    if(!c.L2_per_core) c.L2_per_core = 1u*1024u*1024u;
  }

  // L3 total: sum unique LLCs by deduping shared_cpu_list
  {
    std::set<std::string> seen_groups;
    size_t total = 0;
    std::string cpuroot="/sys/devices/system/cpu/";
    for(const auto& cpu : list_dir(cpuroot)){
      if(cpu.rfind("cpu",0)!=0) continue;
      std::string cpath = cpuroot + cpu + "/cache/";
      for(const auto& idx : list_dir(cpath)){
        std::string base = cpath + idx + "/";
        std::string slev, styp, ssiz, sgroup;
        if(!read_text(base+"level", slev)) continue;
        if(!read_text(base+"type",  styp)) continue;
        if(!read_text(base+"size",  ssiz)) continue;
        trim(slev); trim(styp); trim(ssiz);
        int level = std::atoi(slev.c_str());
        if(level!=3 || styp!="Unified") continue;
        if(!read_text(base+"shared_cpu_list", sgroup)) continue;
        trim(sgroup);
        if(seen_groups.insert(sgroup).second){
          size_t bytes = parse_size_token(ssiz);
          total += bytes;
        }
      }
    }
    if(!total){
      // fallback: try cpu0 once
      std::string base="/sys/devices/system/cpu/cpu0/cache/";
      for(const auto& idx : list_dir(base)){
        std::string p = base + idx + "/";
        std::string slev, styp, ssiz;
        if(!read_text(p+"level", slev)) continue;
        if(!read_text(p+"type",  styp)) continue;
        if(!read_text(p+"size",  ssiz)) continue;
        trim(slev); trim(styp); trim(ssiz);
        int level = std::atoi(slev.c_str());
        if(level==3 && styp=="Unified"){ total = parse_size_token(ssiz); break; }
      }
    }
    if(!total) total = 32u*1024u*1024u;
    c.L3_total = total;
  }
  return c;
}

// --------------------- Heuristic tuned to your two boxes (8T) ---------------------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes){
  // Hard overrides (unchanged)
  if(env_cstr("SGEMM_MC") || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")){
    TileParams t{ env_int("SGEMM_MC",256), env_int("SGEMM_KC",512), env_int("SGEMM_NC",N) };
    t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
    return t;
  }

  const Caches c      = detect_caches_linux();
  const int threads   = std::max(1, env_int("OMP_NUM_THREADS", 1));
  const double L3_pt  = (double)c.L3_total / (double)threads; // per-thread LLC share
  const double L2_tgt = env_double("SGEMM_BETA", 0.60) * (double)c.L2_per_core;
  const double ALPHA  = env_double("SGEMM_ALPHA", 0.75);       // C weight in L2 guard
  const double GAMMA  = env_double("SGEMM_GAMMA", 0.50);       // fraction of per-thread LLC for KC*NC panel

  // For large-N problems, we'd like NC >= this floor if LLC allows.
  const int NC_HEALTHY = env_int("SGEMM_NC_HEALTHY", 480);     // ~10*NR
  const int NC_MIN_BIG = env_int("SGEMM_NC_MIN_BIG", 336);     // ~7*NR

  // Candidate ladders
  std::vector<int> kc_cands = {384, 512, 768, 1024, 1536, 2048, 3072};
  std::vector<int> nc_pref  = {336, 480, 528, 576, 672, 864, 1008, 1152, 1344};
  std::vector<int> mc_cands = {96, 64, 128}; // prefer 96/64 first; 128 only if roomy

  // Clamp KC by K and drop zeros
  for(int& v : kc_cands){ if(v > K) v = K; }
  kc_cands.erase(std::remove_if(kc_cands.begin(), kc_cands.end(), [](int x){ return x<=0; }), kc_cands.end());

  auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };

  // We will first see which KCs can support at least a healthy NC under the LLC bound.
  std::vector<int> kc_ok;
  for(int KC : kc_cands){
    double nc_max_f = (GAMMA * L3_pt) / ((double)KC * (double)dtype_bytes);
    int nc_llc = (int)std::floor(nc_max_f);
    if (nc_llc >= NC_HEALTHY) kc_ok.push_back(KC);
  }
  // If at least one KC can support NC>=healthy, prefer evaluating only those KCs.
  const bool enforce_healthy = !kc_ok.empty();
  const std::vector<int>& kc_eval = enforce_healthy ? kc_ok : kc_cands;

  // Scoring: increase N-tiling weight and penalize too-small NC
  const double W_KPAN = 8.0;   // fewer K panels is good, but not at the cost of tiny NC
  const double W_NTIL = 8.0;   // fewer N tiles matters a lot in your data
  const double W_MTIL = 2.0;   // fewer M tiles (pack A cost)
  const double W_L2   = 4.0;   // bias away from high L2 pressure

  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {64, 512, std::min(N, 576)};

  for(int KC : kc_eval){
    // LLC bound for NC: KC*NC*dtype <= GAMMA * L3_per_thread
    double nc_max_f = (GAMMA * L3_pt) / ((double)KC * (double)dtype_bytes);
    int nc_llc = (int)std::floor(nc_max_f);

    // Build NC list for this KC under the bound
    std::vector<int> nc_cands;
    for(int nc : nc_pref){
      if(nc <= N && nc <= nc_llc) nc_cands.push_back(nc);
    }
    if(nc_cands.empty()){
      // Take the largest NR-multiple allowed (if any)
      int nc = round_down_multiple(std::max(NR, std::min(N, nc_llc)), NR);
      if(nc >= NR) nc_cands.push_back(nc);
    }
    if(nc_cands.empty()) continue; // nothing feasible for this KC

    for(int NC : nc_cands){
      // L2 guard → pick largest MC in preferred order that satisfies:
      // (MC*(KC + ALPHA*NC))*dtype <= L2_tgt
      int chosen_MC = 0;
      for(int MC : mc_cands){
        double bytes = (double)MC * ((double)KC + ALPHA*(double)NC) * (double)dtype_bytes;
        if(bytes <= L2_tgt){ chosen_MC = MC; break; }
      }
      if(chosen_MC == 0) continue;

      // Tile counts
      int Pi = ceil_div(std::max(MR, M), std::max(MR, chosen_MC));
      int Pj = ceil_div(std::max(NR, N), std::max(NR, NC));
      int Pk = ceil_div(std::max(1,  K), std::max(1,  KC));

      // L2 pressure bias
      double A_bytes = (double)chosen_MC * (double)KC * (double)dtype_bytes;
      double C_bytes = (double)chosen_MC * (double)NC * (double)dtype_bytes;
      double l2_bias = (A_bytes + 0.5*C_bytes) / std::max(1.0, (double)c.L2_per_core);

      // Score
      double score = W_KPAN*Pk + W_NTIL*Pj + W_MTIL*Pi + W_L2*l2_bias;

      // Penalty if NC is too small for big-N (LLC forced us down): prefer ≥ 480 if possible
      if (N >= 3*NR && NC < NC_HEALTHY) score *= 1.10;   // +10% penalty
      if (N >= 3*NR && NC < NC_MIN_BIG)  score *= 1.20;   // stronger penalty below 336 (should rarely happen)

      // Nice NC tie-breakers
      if(NC==528 || NC==576 || NC==1008 || NC==1152 || NC==1344) score *= 0.995;
      // Prefer MC=64/96 slightly
      if(chosen_MC==64 || chosen_MC==96) score *= 0.997;

      if(score < best_score){
        best_score = score;
        best.MC = chosen_MC;
        best.KC = KC;
        best.NC = NC;
      }
    }
  }

  // Final sanitize & clamp
  best.MC = clamp_int(round_down_multiple(std::max(MR, best.MC), MR), MR, std::max(MR, M));
  best.KC = std::max(1, std::min(best.KC, K));
  best.NC = clamp_int(round_down_multiple(std::max(NR, best.NC), NR), NR, std::max(NR, N));
  return best;
}

} // namespace gemm
