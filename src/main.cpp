// src/main.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

namespace gemm {
  void sgemm_blocked(const float* A, int M, int K,
                     const float* B, int N,
                     float* C,
                     float alpha, float beta);
}

static float* aligned_alloc64(size_t n_floats){
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats*sizeof(float), 64));
#else
  void* p=nullptr; if (posix_memalign(&p,64,n_floats*sizeof(float))!=0) return nullptr; return (float*)p;
#endif
}
static void aligned_free64(float* p){
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

static void usage(const char* prog){
  std::fprintf(stderr,
    "Usage: %s --bench gemm --m M --n N --k K [--alpha A] [--beta B] [--repeats R] [--warmups W] [--threads T]\n",
    prog);
}

int main(int argc, char** argv){
  int M=2048, N=960, K=1280;
  float alpha=1.f, beta=0.f;
  int repeats=20, warmups=3;        // match Python defaults
  int threads=-1;

  for(int i=1;i<argc;i++){
    auto eq=[&](const char* a,const char* b){ return std::strcmp(a,b)==0; };
    if(eq(argv[i],"--bench")) { ++i; continue; }
    if(eq(argv[i],"--m") && i+1<argc){ M=std::atoi(argv[++i]); continue; }
    if(eq(argv[i],"--n") && i+1<argc){ N=std::atoi(argv[++i]); continue; }
    if(eq(argv[i],"--k") && i+1<argc){ K=std::atoi(argv[++i]); continue; }
    if(eq(argv[i],"--alpha") && i+1<argc){ alpha=std::atof(argv[++i]); continue; }
    if(eq(argv[i],"--beta")  && i+1<argc){ beta =std::atof(argv[++i]); continue; }
    if(eq(argv[i],"--repeats") && i+1<argc){ repeats=std::max(1,std::atoi(argv[++i])); continue; }
    if(eq(argv[i],"--warmups") && i+1<argc){ warmups=std::max(0,std::atoi(argv[++i])); continue; }
    if(eq(argv[i],"--threads") && i+1<argc){ threads=std::atoi(argv[++i]); continue; }
    usage(argv[0]); return 1;
  }

  if(threads>0) omp_set_num_threads(threads);

  const size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
  float* A = aligned_alloc64(szA);
  float* B = aligned_alloc64(szB);
  float* C = aligned_alloc64(szC);
  if(!A || !B || !C){ std::fprintf(stderr,"alloc failed\n"); return 2; }

  // Thread-safe init: thread-local RNG seeded differently per thread
#pragma omp parallel
  {
    std::seed_seq seed{12345, omp_get_thread_num()};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

#pragma omp for
    for (long long i=0; i<(long long)szA; ++i) A[i] = dist(rng);
#pragma omp for
    for (long long i=0; i<(long long)szB; ++i) B[i] = dist(rng);
  }
  std::memset(C, 0, szC*sizeof(float));

  // Warmups (kick unroll picker, fill caches)
  for(int w=0; w<warmups; ++w){
    gemm::sgemm_blocked(A, M, K, B, N, C, alpha, 0.0f);
  }

  double best_ms=1e100, sum_ms=0.0;
  for(int r=0; r<repeats; ++r){
    if (beta==0.0f) std::memset(C, 0, szC*sizeof(float));
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm::sgemm_blocked(A, M, K, B, N, C, alpha, beta);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best_ms = std::min(best_ms, ms);
    sum_ms += ms;
  }
  double mean_ms = sum_ms / repeats;
  double gflops = (2.0 * (double)M * (double)N * (double)K) / (mean_ms/1000.0) / 1e9;

  std::printf("[GEMM]       %.3f ms\n", mean_ms);
  std::printf("best: %.3f ms   GFLOP/s: %.2f   (M=%d N=%d K=%d, repeats=%d threads=%d)\n",
              best_ms, gflops, M, N, K, repeats,
              threads>0 ? threads : omp_get_max_threads());

  aligned_free64(A); aligned_free64(B); aligned_free64(C);
  return 0;
}
