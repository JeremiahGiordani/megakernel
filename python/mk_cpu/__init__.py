# src/attn_cpu/__init__.py
from typing import Literal
import numpy as np
from ._attn_cpu import (  # type: ignore,
    mha_block_dense as _mha_block_dense,
    gemm as _gemm,
    gemm_jit as _gemm_jit
)

def mha_block_dense(x: np.ndarray, W_in: np.ndarray, b_in: np.ndarray, W_out: np.ndarray, b_out: np.ndarray, num_heads: int, causal: bool = False) -> np.ndarray:
    return _mha_block_dense(x, W_in, b_in, W_out, b_out, num_heads, causal)


# float alpha = 1.0f,
#     float beta  = 0.0f,
#     int Mb = 256, int Nb = 96, int Kb = 288,
#     int mr = 16, int nr = 24, int ku = 4
# 
# 
# 2048×1280×960
def gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # return _gemm(A, B)
    return _gemm(A, B)

def gemm_jit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # return _gemm(A, B)
    return _gemm_jit(A, B)


