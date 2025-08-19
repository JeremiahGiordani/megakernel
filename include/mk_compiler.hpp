#pragma once

#if defined(__clang__) || defined(__GNUC__)
  #define MK_RESTRICT __restrict__
  #define MK_PRAGMA(x) _Pragma(#x)
  #define MK_UNROLL_16 MK_PRAGMA(unroll 16)
  #define MK_UNROLL_9  MK_PRAGMA(unroll 9)
  #define MK_UNROLL_3  MK_PRAGMA(unroll 3)
#else
  #define MK_RESTRICT
  #define MK_UNROLL_16
  #define MK_UNROLL_9
  #define MK_UNROLL_3
#endif