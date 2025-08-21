#include "autotuner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <iostream>

namespace gemm {

static constexpr int MR = 8;   // microkernel rows
static constexpr int NR = 48;  // microkernel cols

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
  dirent* ent;
  while((ent=readdir(d))){
    if(ent->d_name[0]=='.') continue;
    names.emplace_back(ent->d_name);
  }
  closedir(d);
  return names;
}

static Caches detect_caches_linux(){
  Caches c{};
  // L2 per-core from cpu0
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
    if(!c.L2_per_core) c.L2_per_core = 1u*1024u*1024u; // fallback
  }
  // L3 total by deduping shared_cpu_list across all cpu*/cache/index*
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
      // fallback to cpu0 L3 if nothing found
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

// --------------------- Persistent cache ---------------------
// Keyed by the "hardware+threads profile".
struct CacheKey {
  size_t L2; size_t L3; int threads; int dtype; int mr; int nr;
  bool operator<(const CacheKey& o) const {
    if (L2!=o.L2) return L2<o.L2;
    if (L3!=o.L3) return L3<o.L3;
    if (threads!=o.threads) return threads<o.threads;
    if (dtype!=o.dtype) return dtype<o.dtype;
    if (mr!=o.mr) return mr<o.mr;
    return nr<o.nr;
  }
};

struct CacheVal { int MC; int KC; int NC; };

static std::string default_cache_path() {
  const char* envp = env_cstr("SGEMM_CACHE_FILE");
  if (envp) return std::string(envp);
  const char* home = getenv("HOME");
  std::string dir = home ? (std::string(home) + "/.cache") : std::string("/tmp");
  // ensure directory exists (best-effort)
  ::mkdir(dir.c_str(), 0755);
  return dir + "/sgemm_tiles_v1.txt";
}

static std::mutex g_cache_mu;
static std::map<CacheKey, CacheVal> g_cache_mem;
static bool g_cache_loaded = false;

static void load_cache_file_locked(const std::string& path){
  if (g_cache_loaded) return;
  std::ifstream ifs(path);
  if (!ifs) { g_cache_loaded = true; return; }
  CacheKey k; CacheVal v;
  while (ifs >> k.L2 >> k.L3 >> k.threads >> k.dtype >> k.mr >> k.nr >> v.MC >> v.KC >> v.NC) {
    g_cache_mem[k] = v;
  }
  g_cache_loaded = true;
}

static void save_cache_file_locked(const std::string& path){
  std::ofstream ofs(path, std::ios::trunc);
  for (const auto& kv : g_cache_mem) {
    const auto& k = kv.first; const auto& v = kv.second;
    ofs << k.L2 << " " << k.L3 << " " << k.threads << " " << k.dtype << " "
        << k.mr << " " << k.nr << " "
        << v.MC << " " << v.KC << " " << v.NC << "\n";
  }
}

// --------------------- Heuristic (as before) ---------------------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes){
  // Hard overrides remain
  if(env_cstr("SGEMM_MC") || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")){
    TileParams t{ env_int("SGEMM_MC",256), env_int("SGEMM_KC",512), env_int("SGEMM_NC",N) };
    t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
    return t;
  }

  const Caches c = detect_caches_linux();

  std::cout << "cache L1d" << c.L1d << std::endl;
  std::cout << "cache L2_per_core" << c.L2_per_core << std::endl;
  std::cout << "cache L3_total" << c.L3_total << std::endl;
  const int threads = std::max(1, env_int("OMP_NUM_THREADS", 1));

  // -- CACHE: try load
  if (!env_int("SGEMM_CACHE_DISABLE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    load_cache_file_locked(default_cache_path());
    CacheKey key{c.L2_per_core, c.L3_total, threads, dtype_bytes, MR, NR};
    auto it = g_cache_mem.find(key);
    if (it != g_cache_mem.end()) {
      // Clamp to current problem and return
      TileParams t{it->second.MC, it->second.KC, it->second.NC};
      t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
      t.KC = std::max(1, std::min(t.KC, K));
      t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
      std::cout << "Using cache" << std::endl;
      return t;
    }
  }
  std::cout << "NOT using cache" << std::endl;

  // Tunables (chosen to reproduce your winners; override with env if needed)
  const double ALPHA = env_double("SGEMM_ALPHA", 0.75);
  const double BETA  = env_double("SGEMM_BETA",  0.60);
  const double GAMMA = env_double("SGEMM_GAMMA", 0.50);

  const double L2_target = BETA * (double)c.L2_per_core;
  const double L3_per_thread = (double)c.L3_total / (double)threads;

  // Candidate ladders
  std::vector<int> kc_cands = {384, 512, 768, 1024, 1536, 2048, 3072};
  std::vector<int> nc_pref  = {336, 480, 528, 576, 672, 864, 1008, 1152, 1344};
  std::vector<int> mc_cands = {96, 64, 128}; // prefer 96/64; allow 128 if roomy

  // Clamp KC by K and drop zeros
  for(int& v : kc_cands){ if(v > K) v = K; }
  kc_cands.erase(std::remove_if(kc_cands.begin(), kc_cands.end(), [](int x){ return x<=0; }), kc_cands.end());

  auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };

  const double W_KPAN = 12.0;  // prefer fewer K panels
  const double W_NTIL = 4.0;   // fewer N tiles
  const double W_MTIL = 2.0;   // fewer M tiles
  const double W_L2   = 4.0;   // bias away from high L2 pressure

  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {64, 512, std::min(N, 576)};

  for(int KC : kc_cands){
    // LLC bound for NC
    double nc_max_f = (GAMMA * L3_per_thread) / ((double)KC * (double)dtype_bytes);
    int nc_llc = (int)std::floor(nc_max_f);
    int min_reasonable_nc = (N >= 3*NR) ? 336 : NR;
    if(nc_llc < min_reasonable_nc) nc_llc = min_reasonable_nc;

    std::vector<int> nc_cands;
    for(int nc : nc_pref){
      if(nc <= N && nc <= nc_llc) nc_cands.push_back(nc);
    }
    if(nc_cands.empty()){
      int nc = round_down_multiple(std::max(NR, std::min(N, nc_llc)), NR);
      if(nc >= NR) nc_cands.push_back(nc);
    }

    for(int NC : nc_cands){
      // L2 guard for MC
      int chosen_MC = 0;
      for(int MC : mc_cands){
        double bytes = (double)MC * ((double)KC + ALPHA*(double)NC) * (double)dtype_bytes;
        if(bytes <= L2_target){ chosen_MC = MC; break; }
      }
      if(chosen_MC == 0) continue;

      int Pi = ceil_div(std::max(MR, M), std::max(MR, chosen_MC));
      int Pj = ceil_div(std::max(NR, N), std::max(NR, NC));
      int Pk = ceil_div(std::max(1,  K), std::max(1,  KC));

      double A_bytes = (double)chosen_MC * (double)KC * (double)dtype_bytes;
      double C_bytes = (double)chosen_MC * (double)NC * (double)dtype_bytes;
      double l2_bias = (A_bytes + 0.5*C_bytes) / std::max(1.0, (double)c.L2_per_core);

      double score = W_KPAN*Pk + W_NTIL*Pj + W_MTIL*Pi + W_L2*l2_bias;

      if(NC==528 || NC==576 || NC==1008 || NC==1152 || NC==1344) score *= 0.995;
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

  // -- CACHE: save
  if (!env_int("SGEMM_CACHE_DISABLE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    load_cache_file_locked(default_cache_path());
    CacheKey key{c.L2_per_core, c.L3_total, threads, dtype_bytes, MR, NR};
    g_cache_mem[key] = CacheVal{best.MC, best.KC, best.NC};

    std::string path = default_cache_path();
    if (env_int("SGEMM_CACHE_CLEAR", 0)) {
      // clear file before writing
      std::ofstream(path, std::ios::trunc).close();
    }
    save_cache_file_locked(path);
  }

  return best;
}

} // namespace gemm
