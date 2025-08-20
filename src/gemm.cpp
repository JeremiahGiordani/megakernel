// sgemm_blocked.cpp — AVX-512 SGEMM with classic 3-level blocking (MC, KC, NC)
// Updated:
//  - Parallel B-panel pack (no OpenMP tasks, no serialized single-thread pack)
//  - NC defaults to N (pack whole N by default; override via SGEMM_NC)
//  - Keeps templated UNROLL, masked tails, pointer-bump, tuned prefetch
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <omp.h>
#include <iostream>

namespace gemm {

static constexpr int MR = 8;    // micro rows
static constexpr int NR = 48;   // micro cols (3× zmm)

// ---------------- helpers ----------------
static inline float* aligned_alloc64(size_t n_floats){
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats*sizeof(float), 64));
#else
  void* p=nullptr; if (posix_memalign(&p,64,n_floats*sizeof(float))!=0) return nullptr; return (float*)p;
#endif
}
static inline void aligned_free64(float* p){
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}
static inline bool aligned64(const void* p){ return ((reinterpret_cast<uintptr_t>(p)&63u)==0u); }
static inline int  env_int (const char* name, int dflt){ const char* s=getenv(name); return s? std::atoi(s):dflt; }
static inline bool env_bool(const char* name, bool dflt){ const char* s=getenv(name); return s? (std::atoi(s)!=0):dflt; }

static inline int pick_prefetch_distance(int K){
  int pf = env_int("SGEMM_PFD", -1);
  if (pf >= 0) return pf;
  if (K < 64) return 0;
  int x = K >> 3;
  if (x < 16)  x = 16;
  if (x > 128) x = 128;
  return x;
}

// ---------------- packing ----------------
// A: pack MR rows across Kc (k-major), fuse alpha
static inline void pack_A_tile_mrK(const float* __restrict A, int ldA,
                                   int mr_eff, int Kc, float alpha,
                                   float* __restrict Ap){
  for(int k=0;k<Kc;++k){
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k*MR;
    int r=0; for(; r<mr_eff; ++r) dst[r] = a_col[(size_t)r*ldA]*alpha;
            for(; r<MR;     ++r) dst[r] = 0.0f;
  }
}

// Pack A block [ic:ic+mc) × [k0:k0+Kc)
static inline void pack_A_block_mcxKc(const float* __restrict A, int ldA,
                                      int mc, int Kc, float alpha,
                                      float* __restrict Ap_blk){
  const int MT = (mc + MR - 1)/MR;
  for(int tm=0; tm<MT; ++tm){
    int r0 = tm*MR;
    int mr_eff = std::min(MR, mc - r0);
    const float* A_src = A + (size_t)r0*ldA;
    float*       A_dst = Ap_blk + (size_t)tm*Kc*MR;
    pack_A_tile_mrK(A_src, ldA, mr_eff, Kc, alpha, A_dst);
  }
}

// B: pack NR cols across Kc (k-major)
static inline void pack_B_tile_Knr(const float* __restrict B, int ldB,
                                   int Kc, int nr_eff,
                                   float* __restrict Bp){
  for(int k=0;k<Kc;++k){
    const float* b_row = B + (size_t)k*ldB;
    float* dst = Bp + (size_t)k*NR;
    if(nr_eff==NR){
      __m512 x0=_mm512_loadu_ps(b_row+ 0);
      __m512 x1=_mm512_loadu_ps(b_row+16);
      __m512 x2=_mm512_loadu_ps(b_row+32);
      _mm512_store_ps(dst+ 0, x0);
      _mm512_store_ps(dst+16, x1);
      _mm512_store_ps(dst+32, x2);
    }else{
      int j=0; for(; j<nr_eff; ++j) dst[j]=b_row[j];
               for(; j<NR;     ++j) dst[j]=0.0f;
    }
  }
}

// ---------------- masks ----------------
static inline void nr_masks(int nr_eff, __mmask16 &m0, __mmask16 &m1, __mmask16 &m2){
  int c0 = std::min(16, nr_eff);
  int c1 = std::max(0, std::min(16, nr_eff - 16));
  int c2 = std::max(0, std::min(16, nr_eff - 32));
  m0 = (c0==16) ? 0xFFFF : ((__mmask16)((1u<<c0)-1u));
  m1 = (c1==16) ? 0xFFFF : ((__mmask16)((1u<<c1)-1u));
  m2 = (c2==16) ? 0xFFFF : ((__mmask16)((1u<<c2)-1u));
}

// ---------------- micro-kernels (templated UNROLL) ----------------
template<int UNROLL>
static inline void micro_8x48_overwrite_u(const float* __restrict Ap,
                                          const float* __restrict Bp,
                                          float* __restrict C, int ldc, int Kc,
                                          int mr_eff, int nr_eff, bool stream_last_panel)
{
  static_assert(UNROLL>=1 && UNROLL<=8, "UNROLL in [1,8]");
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  const int PFD = pick_prefetch_distance(Kc);
  const float* a_ptr = Ap;
  const float* b_ptr = Bp;

  int k=0, kend = Kc - (Kc % UNROLL);
  auto k_step = [&](const float* a, const float* b){
    __m512 b0=_mm512_load_ps(b+ 0);
    __m512 b1=_mm512_load_ps(b+16);
    __m512 b2=_mm512_load_ps(b+32);
#pragma unroll
    for(int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  };
  for(; k<kend; k+=UNROLL){
    if(PFD>0){
      int kp=k+PFD; if(kp<Kc){
        _mm_prefetch((const char*)(Ap+(size_t)kp*MR), _MM_HINT_T0);
        _mm_prefetch((const char*)(Bp+(size_t)kp*NR), _MM_HINT_T0);
      }
    }
#pragma unroll
    for(int u=0; u<UNROLL; ++u){
      k_step(a_ptr + (size_t)u*MR, b_ptr + (size_t)u*NR);
    }
    a_ptr += (size_t)UNROLL*MR;
    b_ptr += (size_t)UNROLL*NR;
  }
  for(; k<Kc; ++k){
    k_step(a_ptr, b_ptr);
    a_ptr += MR; b_ptr += NR;
  }

  if(mr_eff==MR && nr_eff==NR){
#pragma unroll
    for(int r=0;r<MR;++r){
      float* c = C + (size_t)r*ldc;
      if(stream_last_panel && aligned64(c)){
        _mm512_stream_ps(c+ 0, acc0[r]);
        _mm512_stream_ps(c+16, acc1[r]);
        _mm512_stream_ps(c+32, acc2[r]);
      }else{
        if(aligned64(c)){
          _mm512_store_ps (c+ 0, acc0[r]);
          _mm512_store_ps (c+16, acc1[r]);
          _mm512_store_ps (c+32, acc2[r]);
        }else{
          _mm512_storeu_ps(c+ 0, acc0[r]);
          _mm512_storeu_ps(c+16, acc1[r]);
          _mm512_storeu_ps(c+32, acc2[r]);
        }
      }
    }
  }else{
    __mmask16 m0,m1,m2; nr_masks(nr_eff,m0,m1,m2);
    for(int r=0;r<mr_eff;++r){
      float* c = C + (size_t)r*ldc;
      _mm512_mask_storeu_ps(c+ 0, m0, acc0[r]);
      _mm512_mask_storeu_ps(c+16, m1, acc1[r]);
      _mm512_mask_storeu_ps(c+32, m2, acc2[r]);
    }
  }
}

template<int UNROLL>
static inline void micro_8x48_accum_u(const float* __restrict Ap,
                                      const float* __restrict Bp,
                                      float* __restrict C, int ldc, int Kc,
                                      int mr_eff, int nr_eff)
{
  static_assert(UNROLL>=1 && UNROLL<=8, "UNROLL in [1,8]");
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  const int PFD = pick_prefetch_distance(Kc);
  const float* a_ptr = Ap;
  const float* b_ptr = Bp;

  int k=0, kend = Kc - (Kc % UNROLL);
  auto k_step = [&](const float* a, const float* b){
    __m512 b0=_mm512_load_ps(b+ 0);
    __m512 b1=_mm512_load_ps(b+16);
    __m512 b2=_mm512_load_ps(b+32);
#pragma unroll
    for(int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  };
  for(; k<kend; k+=UNROLL){
    if(PFD>0){
      int kp=k+PFD; if(kp<Kc){
        _mm_prefetch((const char*)(Ap+(size_t)kp*MR), _MM_HINT_T0);
        _mm_prefetch((const char*)(Bp+(size_t)kp*NR), _MM_HINT_T0);
      }
    }
#pragma unroll
    for(int u=0; u<UNROLL; ++u){
      k_step(a_ptr + (size_t)u*MR, b_ptr + (size_t)u*NR);
    }
    a_ptr += (size_t)UNROLL*MR;
    b_ptr += (size_t)UNROLL*NR;
  }
  for(; k<Kc; ++k){
    k_step(a_ptr, b_ptr);
    a_ptr += MR; b_ptr += NR;
  }

  if(mr_eff==MR && nr_eff==NR){
#pragma unroll
    for(int r=0;r<MR;++r){
      float* c = C + (size_t)r*ldc;
      __m512 c0 = aligned64(c)? _mm512_load_ps (c+ 0) : _mm512_loadu_ps(c+ 0);
      __m512 c1 = aligned64(c)? _mm512_load_ps (c+16) : _mm512_loadu_ps(c+16);
      __m512 c2 = aligned64(c)? _mm512_load_ps (c+32) : _mm512_loadu_ps(c+32);
      c0 = _mm512_add_ps(c0, acc0[r]);
      c1 = _mm512_add_ps(c1, acc1[r]);
      c2 = _mm512_add_ps(c2, acc2[r]);
      if(aligned64(c)){
        _mm512_store_ps (c+ 0, c0);
        _mm512_store_ps (c+16, c1);
        _mm512_store_ps (c+32, c2);
      }else{
        _mm512_storeu_ps(c+ 0, c0);
        _mm512_storeu_ps(c+16, c1);
        _mm512_storeu_ps(c+32, c2);
      }
    }
  }else{
    __mmask16 m0,m1,m2; nr_masks(nr_eff,m0,m1,m2);
    for(int r=0;r<mr_eff;++r){
      float* c = C + (size_t)r*ldc;
      __m512 c0 = _mm512_maskz_loadu_ps(m0, c+ 0);
      __m512 c1 = _mm512_maskz_loadu_ps(m1, c+16);
      __m512 c2 = _mm512_maskz_loadu_ps(m2, c+32);
      c0 = _mm512_add_ps(c0, acc0[r]);
      c1 = _mm512_add_ps(c1, acc1[r]);
      c2 = _mm512_add_ps(c2, acc2[r]);
      _mm512_mask_storeu_ps(c+ 0, m0, c0);
      _mm512_mask_storeu_ps(c+16, m1, c1);
      _mm512_mask_storeu_ps(c+32, m2, c2);
    }
  }
}

// dispatch tables
using MicroOverwrite = void(*)(const float*, const float*, float*, int, int, int, int, bool);
using MicroAccum    = void(*)(const float*, const float*, float*, int, int, int, int);
template<int U> static inline void micro_overwrite_entry(const float* a,const float* b,float* c,int ldc,int Kc,int mr,int nr,bool stream){
  micro_8x48_overwrite_u<U>(a,b,c,ldc,Kc,mr,nr,stream);
}
template<int U> static inline void micro_accum_entry(const float* a,const float* b,float* c,int ldc,int Kc,int mr,int nr){
  micro_8x48_accum_u<U>(a,b,c,ldc,Kc,mr,nr);
}
static constexpr int kNumCand = 6;
static constexpr int kCand[kNumCand] = {1,2,3,4,6,8};
static MicroOverwrite kOwTable[kNumCand] = {
  micro_overwrite_entry<1>, micro_overwrite_entry<2>, micro_overwrite_entry<3>,
  micro_overwrite_entry<4>, micro_overwrite_entry<6>, micro_overwrite_entry<8>
};
static MicroAccum kAcTable[kNumCand] = {
  micro_accum_entry<1>, micro_accum_entry<2>, micro_accum_entry<3>,
  micro_accum_entry<4>, micro_accum_entry<6>, micro_accum_entry<8>
};

// one-shot UNROLL picker (same safe buffer sizing/indexing)
static std::atomic<int> g_unroll_idx{-1};
static inline int pick_unroll_once_idx(int Kc_for_test){
  int tune   = env_int("SGEMM_TUNE", 1);
  int forceU = env_int("SGEMM_U", 0);
  if(tune==0 && forceU>0){
    for(int i=0;i<kNumCand;++i) if(kCand[i]==forceU){ g_unroll_idx.store(i); return i; }
  }
  int already = g_unroll_idx.load();
  if(already>=0) return already;

  const int test_MT = 6, test_NT = 2, ldc = test_NT*NR;
  float* Ap = aligned_alloc64((size_t)test_MT * Kc_for_test * MR);
  float* Bp = aligned_alloc64((size_t)test_NT * Kc_for_test * NR);
  float* C  = aligned_alloc64((size_t)test_MT * MR * ldc);
  std::memset(Ap, 0, sizeof(float)*(size_t)test_MT*Kc_for_test*MR);
  std::memset(Bp, 0, sizeof(float)*(size_t)test_NT*Kc_for_test*NR);
  std::memset(C , 0, sizeof(float)*(size_t)test_MT*MR*ldc);

  auto bench = [&](int idx){
    auto fn = kOwTable[idx];
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int tm=0; tm<test_MT; ++tm){
      const float* Ap_t = Ap + (size_t)tm*Kc_for_test*MR;
      for(int tn=0; tn<test_NT; ++tn){
        const float* Bp_t = Bp + (size_t)tn*Kc_for_test*NR;
        float* C_t = C + (size_t)tm*MR*ldc + tn*NR;
        fn(Ap_t, Bp_t, C_t, ldc, Kc_for_test, MR, NR, false);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1-t0).count();
  };

  int best_i = 0; double best_t = 1e100;
  for(int i=0;i<kNumCand;++i){
    double t = bench(i);
    if(t < best_t){ best_t = t; best_i = i; }
  }
  aligned_free64(Ap); aligned_free64(Bp); aligned_free64(C);
  g_unroll_idx.store(best_i);
  return best_i;
}

// ---------------- top-level SGEMM (3-level blocking) ----------------
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta)
{
  if(M<=0 || N<=0 || K<=0) return;

  const int ldA = K, ldB = N, ldC = N;

  if(alpha==0.0f){
#pragma omp parallel for schedule(static)
    for(int i=0;i<M;++i){
      float* Crow = C + (size_t)i*ldC;
      if(beta==0.0f) std::memset(Crow, 0, sizeof(float)*N);
      else if(beta!=1.0f) for(int j=0;j<N;++j) Crow[j]*=beta;
    }
    return;
  }

  if(beta!=0.0f){
#pragma omp parallel for schedule(static)
    for(int i=0;i<M;++i){
      for(int j=0;j<N;++j){
        float acc=0.f; for(int k=0;k<K;++k) acc += A[(size_t)i*ldA+k] * B[(size_t)k*ldB+j];
        C[(size_t)i*ldC + j] = alpha*acc + beta*C[(size_t)i*ldC + j];
      }
    }
    return;
  }

  // ---- Tunables (env) ----
  int MC = env_int("SGEMM_MC", 256);
  int KC = env_int("SGEMM_KC", 1024);
  int NC = env_int("SGEMM_NC", N);   // default: pack whole N (matches your best results)

  MC = std::max(MR, MC);
  KC = std::max(1,  KC);
  NC = std::max(NR, std::min(NC, N));

  // Pick UNROLL once based on representative Kc
  int kc_test = std::min(KC, K);
  int uidx = pick_unroll_once_idx(kc_test);
  // std::cout << "Unroll idx: " << uidx << std::endl;
  MicroOverwrite micro_overwrite = kOwTable[uidx];
  MicroAccum    micro_accum     = kAcTable[uidx];

  // Shared B-pack buffers (two are fine though we use one per panel now)
  const int NC_round_cols = ((NC + NR - 1)/NR)*NR;
  float* Bp_buf = nullptr;

#pragma omp parallel
  {
    // Per-thread A pack buffer: max MC×KC floats
    float* Ap_thr = aligned_alloc64((size_t)MC * (size_t)KC);

#pragma omp single
    {
      Bp_buf = aligned_alloc64((size_t)KC * (size_t)NC_round_cols);
    }
#pragma omp barrier

    for(int jc=0; jc<N; jc+=NC){
      const int nc = std::min(NC, N - jc);
      const int NT = (nc + NR - 1)/NR;

      for(int k0=0; k0<K; k0+=KC){
        const int Kc = std::min(KC, K - k0);
        const bool is_last_panel = (k0 + Kc >= K);

        // ---- Parallel B-pack for this (jc,k0) panel ----
#pragma omp for schedule(static)
        for(int tn=0; tn<NT; ++tn){
          const int j0     = jc + tn*NR;
          const int nr_eff = std::min(NR, N - j0);
          const float* B_src = B + (size_t)k0*ldB + j0;
          float*       B_dst = Bp_buf + (size_t)tn*Kc*NR;
          pack_B_tile_Knr(B_src, ldB, Kc, nr_eff, B_dst);
        }

        // ---- Compute over IC blocks (each thread packs A(ic,k0) once) ----
#pragma omp for schedule(static)
        for(int ic=0; ic<M; ic+=MC){
          const int mc = std::min(MC, M - ic);
          const int MT = (mc + MR - 1)/MR;

          pack_A_block_mcxKc(A + (size_t)ic*ldA + k0, ldA, mc, Kc, alpha, Ap_thr);

          for(int tm=0; tm<MT; ++tm){
            const int r_off   = ic + tm*MR;
            const int mr_here = std::min(MR, M - r_off);
            const float* Ap_t = Ap_thr + (size_t)tm*Kc*MR;

            for(int tn=0; tn<NT; ++tn){
              const int j0      = jc + tn*NR;
              const int nr_eff  = std::min(NR, N - j0);
              const float* Bp_t = Bp_buf + (size_t)tn*Kc*NR;
              float* C_tile     = C + (size_t)r_off*ldC + j0;

              if(k0==0){
                const bool stream_ok = (is_last_panel && mr_here==MR && nr_eff==NR && aligned64(C_tile));
                micro_overwrite(Ap_t, Bp_t, C_tile, ldC, Kc, mr_here, nr_eff, stream_ok);
              }else{
                micro_accum(Ap_t, Bp_t, C_tile, ldC, Kc, mr_here, nr_eff);
              }
            }
          }
        } // ic
      } // k0
    }   // jc

    aligned_free64(Ap_thr);
#pragma omp single
    {
      aligned_free64(Bp_buf);
    }
  } // omp parallel
}

} // namespace gemm
