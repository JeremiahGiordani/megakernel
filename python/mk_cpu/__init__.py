# src/mk_cpu/__init__.py
from typing import Literal
import numpy as np
from ._mk_cpu import (  # type: ignore
    gemm as _gemm
)

def gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _gemm(A, B)


