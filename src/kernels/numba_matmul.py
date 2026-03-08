"""
Matrix multiplication using Numba CUDA JIT.

Provides a Python JIT-compiled CUDA kernel for comparison
with CuPy RawKernel and PyTorch implementations.
"""

import numpy as np
from numba import cuda


@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0

        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]

        C[row, col] = tmp


def run_numba_kernel(n: int):
    """
    Executes matrix multiplication using Numba CUDA.

    Args:
        n (int): matrix dimension

    Returns:
        elapsed_time (float)
        result_sample (np.ndarray)
    """

    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    C = np.zeros((n, n), dtype=np.float32)

    threads = (16, 16)
    blocks = (
        (n + threads[0] - 1) // threads[0],
        (n + threads[1] - 1) // threads[1],
    )

    # Warmup run (JIT compilation)
    matmul_kernel[blocks, threads](A, B, C)
    cuda.synchronize()

    C[:] = 0

    start = cuda.event()
    end = cuda.event()

    start.record()

    matmul_kernel[blocks, threads](A, B, C)

    end.record()
    end.synchronize()

    elapsed = cuda.event_elapsed_time(start, end) / 1000.0

    return elapsed, C[:2, :2]


if __name__ == "__main__":
    size = 1024

    t, sample = run_numba_kernel(size)

    print(f"Size: {size}x{size}")
    print(f"Execution time: {t:.6f} seconds")
    print("Sample output:\n", sample)
