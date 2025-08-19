import pytest
import numpy as np
import torch

from mk_cpu import gemm

# Shapes to test: square, rectangular, and irregular
SHAPES = [
    # Small / trivial
    (1, 1, 1),
    (1, 5, 7),
    (7, 1, 13),
    (3, 3, 3),

    # Small oddballs
    (5, 7, 11),
    (13, 17, 19),
    (31, 47, 29),

    # Degenerate wide/tall
    (1, 1024, 1),
    (1024, 1, 1),
    (1, 1, 1024),
    (2, 4096, 3),
    (4096, 2, 7),

    # Medium, irregular
    (33, 65, 129),
    (127, 255, 63),
    (257, 129, 65),
    (123, 456, 789),
    (111, 222, 333),

    # Misaligned vs. blocking
    (15, 15, 15),
    (17, 33, 65),
    (29, 58, 87),
    (30, 45, 60),
    (63, 127, 255),

    # Powers of 2 + near powers
    (64, 64, 64),
    (65, 65, 65),
    (128, 256, 512),
    (255, 255, 255),
    (256, 255, 257),

    # Large realistic but irregular
    (512, 513, 514),
    (1023, 1025, 1027),
    (999, 1001, 1003),

    # Benchmark-style
    (2048, 1280, 960),
    (2048, 1280, 1920),
    (4096, 4096, 4096),

    # Extreme skinny/fat
    (8192, 8, 16),
    (16, 8192, 8),
    (8, 16, 8192),

    # Mega oddball primes
    (101, 103, 107),
    (211, 223, 227),
    (401, 409, 419),
]


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_gemm_matches_torch(M, K, N):
    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate random inputs
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # Torch reference
    A_t = torch.tensor(A)
    B_t = torch.tensor(B)
    ref = torch.matmul(A_t, B_t).numpy()

    # Custom GEMM
    out = gemm(A, B)

    # Compare
    np.testing.assert_allclose(
        out, ref, rtol=1e-3, atol=1e-3,
        err_msg=f"GEMM mismatch for shape {(M,K,N)}"
    )
