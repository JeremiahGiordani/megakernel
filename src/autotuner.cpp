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

namespace gemm {

static constexpr int MR = 8;
static constexpr int NR = 48;

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
    if(!c.L2_per_core) c.L2_per_core = 1u*1024u*1024u;
  }
  // L3 total by deduping shared_cpu_list across all cpus
  {
    std::set<std::string> seen;
    size_t total = 0;
    std::string cpuroot="/sys/devices/system/cpu/";
    for(const auto& cpu : list_dir(cpuroot)){
      if(cpu.rfind("cpu",0)!=0) continue;
      std::string cpath = cpuroot + cpu + "/cache/";
      for(const auto& idx : list_dir(cpath)){
        std::string base = cpath + idx + "/";
        std::string slev, styp, ssiz, sgrp;
        if(!read_text(base+"level", slev)) continue;
        if(!read_text(base+"type",  styp)) continue;
        if(!read_text(base+"size",  ssiz)) continue;
        trim(slev); trim(styp); trim(ssiz);
        if(std::atoi(slev.c_str())!=3 || styp!="Unified") continue;
        if(!read_text(base+"shared_cpu_list", sgrp)) continue; trim(sgrp);
        if(seen.insert(sgrp).second) total += parse_size_token(ssiz);
      }
    }
    if(!total){
      std::string base="/sys/devices/system/cpu/cpu0/cache/";
      for(const auto& idx : list_dir(base)){
        std::string p = base + idx + "/";
        std::string slev, styp, ssiz;
        if(!read_text(p+"level", slev)) continue;
        if(!read_text(p+"type",  styp)) continue;
        if(!read_text(p+"size",  ssiz)) continue;
        trim(slev); trim(styp); trim(ssiz);
        if(std::atoi(slev.c_str())==3 && styp=="Unified"){ c.L3_total = parse_size_token(ssiz); break; }
      }
    } else {
      c.L3_total = total;
    }
    if(!c.L3_total) c.L3_total = 32u*1024u*1024u;
  }
  return c;
}

// --------------------- persistent caches (versioned) ---------------------
struct CacheKey {
  size_t L2; size_t L3; int threads; int dtype; int mr; int nr; int ver;
  bool operator<(const CacheKey& o) const {
    if (L2!=o.L2) return L2<o.L2;
    if (L3!=o.L3) return L3<o.L3;
    if (threads!=o.threads) return threads<o.threads;
    if (dtype!=o.dtype) return dtype<o.dtype;
    if (mr!=o.mr) return mr<o.mr;
    if (nr!=o.nr) return nr<o.nr;
    return ver<o.ver;
  }
};
struct CacheVal { int MC; int KC; int NC; };

static int cache_version() {
  return env_int("SGEMM_CACHE_VERSION", 2);
}
static std::string default_cache_path() {
  if (const char* p = env_cstr("SGEMM_CACHE_FILE")) return std::string(p);
  const char* home = getenv("HOME");
  std::string dir = home ? (std::string(home) + "/.cache") : std::string("/tmp");
  ::mkdir(dir.c_str(), 0755);
  return dir + "/sgemm_tiles_v2.txt";
}

// --- FAST host+threads cache (skips detection entirely when present)
struct HostKey {
  std::string host; int threads; int dtype; int mr; int nr; int ver;
  bool operator<(const HostKey& o) const {
    if (host!=o.host) return host<o.host;
    if (threads!=o.threads) return threads<o.threads;
    if (dtype!=o.dtype) return dtype<o.dtype;
    if (mr!=o.mr) return mr<o.mr;
    if (nr!=o.nr) return nr<o.nr;
    return ver<o.ver;
  }
};

static std::string fast_cache_path() {
  if (const char* p = env_cstr("SGEMM_FAST_CACHE_FILE")) return std::string(p);
  const char* home = getenv("HOME");
  std::string dir = home ? (std::string(home) + "/.cache") : std::string("/tmp");
  ::mkdir(dir.c_str(), 0755);
  return dir + "/sgemm_tiles_fast_v2.txt";
}
static std::string hostname_str() {
  char buf[256]; buf[0]='\0';
  if (gethostname(buf, sizeof(buf)-1)==0) { buf[sizeof(buf)-1]='\0'; return std::string(buf); }
  return std::string("unknown-host");
}

static std::mutex g_cache_mu;
static std::map<CacheKey, CacheVal> g_cache_mem;
static bool g_cache_loaded=false;
static std::map<HostKey, CacheVal> g_fast_mem;
static bool g_fast_loaded=false;

static void load_cache_file_locked(const std::string& path){
  if (g_cache_loaded) return;
  std::ifstream ifs(path); if(!ifs){ g_cache_loaded=true; return; }
  CacheKey k; CacheVal v;
  while (ifs >> k.L2 >> k.L3 >> k.threads >> k.dtype >> k.mr >> k.nr >> k.ver
             >> v.MC >> v.KC >> v.NC) {
    if (k.ver == cache_version()) g_cache_mem[k]=v;
  }
  g_cache_loaded=true;
}
static void save_cache_file_locked(const std::string& path){
  std::ofstream ofs(path, std::ios::trunc);
  for (const auto& kv : g_cache_mem) {
    const auto& k=kv.first; const auto& v=kv.second;
    ofs << k.L2 << " " << k.L3 << " " << k.threads << " " << k.dtype << " "
        << k.mr << " " << k.nr << " " << k.ver << " "
        << v.MC << " " << v.KC << " " << v.NC << "\n";
  }
}

static void load_fast_file_locked(const std::string& path){
  if (g_fast_loaded) return;
  std::ifstream ifs(path); if(!ifs){ g_fast_loaded=true; return; }
  HostKey k; CacheVal v;
  while (ifs >> k.host >> k.threads >> k.dtype >> k.mr >> k.nr >> k.ver
             >> v.MC >> v.KC >> v.NC) {
    if (k.ver == cache_version()) g_fast_mem[k]=v;
  }
  g_fast_loaded=true;
}
static void save_fast_file_locked(const std::string& path){
  std::ofstream ofs(path, std::ios::trunc);
  for (const auto& kv : g_fast_mem) {
    const auto& k=kv.first; const auto& v=kv.second;
    ofs << k.host << " " << k.threads << " " << k.dtype << " "
        << k.mr << " " << k.nr << " " << k.ver << " "
        << v.MC << " " << v.KC << " " << v.NC << "\n";
  }
}

// --------------------- tile picker ---------------------
TileParams pick_tiles_avx512(int M, int N, int K, int dtype_bytes){
  // Hard overrides
  if(env_cstr("SGEMM_MC") || env_cstr("SGEMM_KC") || env_cstr("SGEMM_NC")){
    TileParams t{ env_int("SGEMM_MC",256), env_int("SGEMM_KC",512), env_int("SGEMM_NC",N) };
    t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
    t.KC = std::max(1, t.KC);
    t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
    return t;
  }

  const int threads = std::max(1, env_int("OMP_NUM_THREADS", 1));
  const int ver = cache_version();

  // ---------- FAST PATH: host+threads cache (no detection) ----------
  if (!env_int("SGEMM_CACHE_DISABLE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    load_fast_file_locked(fast_cache_path());
    HostKey hk{hostname_str(), threads, dtype_bytes, MR, NR, ver};
    auto it = g_fast_mem.find(hk);
    if (it != g_fast_mem.end()) {
      TileParams t{it->second.MC, it->second.KC, it->second.NC};
      t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
      t.KC = std::max(1, std::min(t.KC, K));
      t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
      return t;
    }
  }

  // ---------- SLOW PATH: detect caches → read keyed cache → compute ----------
  const Caches c = detect_caches_linux();

  // Try full cache keyed by L2/L3
  if (!env_int("SGEMM_CACHE_DISABLE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    load_cache_file_locked(default_cache_path());
    CacheKey key{c.L2_per_core, c.L3_total, threads, dtype_bytes, MR, NR, ver};
    auto it = g_cache_mem.find(key);
    if (it != g_cache_mem.end()) {
      TileParams t{it->second.MC, it->second.KC, it->second.NC};
      t.MC = clamp_int(round_down_multiple(std::max(MR,t.MC), MR), MR, std::max(MR,M));
      t.KC = std::max(1, std::min(t.KC, K));
      t.NC = clamp_int(round_down_multiple(std::max(NR,t.NC), NR), NR, std::max(NR,N));
      // also populate fast cache for next time
      HostKey hk{hostname_str(), threads, dtype_bytes, MR, NR, ver};
      g_fast_mem[hk] = CacheVal{t.MC, t.KC, t.NC};
      save_fast_file_locked(fast_cache_path());
      return t;
    }
  }

  // ---- Heuristic (unchanged from last version) ----
  const double ALPHA = env_double("SGEMM_ALPHA", 0.75);
  const double BETA  = env_double("SGEMM_BETA",  0.60);
  const double GAMMA = env_double("SGEMM_GAMMA", 0.50);

  const double L2_target = BETA * (double)c.L2_per_core;
  const double L3_per_thread = (double)c.L3_total / (double)threads;

  std::vector<int> kc_cands = {384, 512, 768, 1024, 1536, 2048, 3072};
  std::vector<int> nc_pref  = {336, 480, 528, 576, 672, 864, 1008, 1152, 1344};
  std::vector<int> mc_cands = {96, 64, 128};

  for(int& v : kc_cands){ if(v > K) v = K; }
  kc_cands.erase(std::remove_if(kc_cands.begin(), kc_cands.end(), [](int x){ return x<=0; }), kc_cands.end());
  auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };

  const double W_KPAN = 12.0;
  const double W_NTIL = 4.0;
  const double W_MTIL = 2.0;
  const double W_L2   = 4.0;
  const double W_SMALLNC = 8.0;

  double best_score = std::numeric_limits<double>::infinity();
  TileParams best {64, 512, std::min(N, 576)};

  const int NC_TARGET_MIN = env_int("SGEMM_NC_TARGET_MIN", 480);
  std::vector<int> kc_pruned;
  kc_pruned.reserve(kc_cands.size());
  for (int KC : kc_cands) {
    double kc_nc_bytes = (double)KC * (double)NC_TARGET_MIN * (double)dtype_bytes;
    if (kc_nc_bytes <= GAMMA * L3_per_thread) kc_pruned.push_back(KC);
  }
  if (kc_pruned.empty()) kc_pruned = kc_cands;

  for(int KC : kc_pruned){
    double nc_max_f = (GAMMA * L3_per_thread) / ((double)KC * (double)dtype_bytes);
    int nc_llc = (int)std::floor(nc_max_f);

    const int MIN_NC_BIGN = (N >= 3*NR) ? 336 : NR;
    if (nc_llc < MIN_NC_BIGN) continue;

    std::vector<int> nc_cands;
    for(int nc : nc_pref){
      if(nc <= N && nc <= nc_llc) nc_cands.push_back(nc);
    }
    if(nc_cands.empty()){
      int nc = round_down_multiple(std::max(NR, std::min(N, nc_llc)), NR);
      if(nc >= NR) nc_cands.push_back(nc);
    }

    for(int NC : nc_cands){
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
      if (N >= 3*NR && NC < 480) score += W_SMALLNC;
      if(chosen_MC==64 || chosen_MC==96) score *= 0.997;

      if(score < best_score){
        best_score = score;
        best.MC = chosen_MC;
        best.KC = KC;
        best.NC = NC;
      }
    }
  }

  best.MC = clamp_int(round_down_multiple(std::max(MR, best.MC), MR), MR, std::max(MR, M));
  best.KC = std::max(1, std::min(best.KC, K));
  best.NC = clamp_int(round_down_multiple(std::max(NR, best.NC), NR), NR, std::max(NR, N));

  // Save to both caches for next time (so we’ll hit the fast path)
  if (!env_int("SGEMM_CACHE_DISABLE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    load_cache_file_locked(default_cache_path());
    CacheKey key{c.L2_per_core, c.L3_total, threads, dtype_bytes, MR, NR, ver};
    g_cache_mem[key] = CacheVal{best.MC, best.KC, best.NC};
    save_cache_file_locked(default_cache_path());

    load_fast_file_locked(fast_cache_path());
    HostKey hk{hostname_str(), threads, dtype_bytes, MR, NR, ver};
    g_fast_mem[hk] = CacheVal{best.MC, best.KC, best.NC};
    save_fast_file_locked(fast_cache_path());
  }

  return best;
}

} // namespace gemm
