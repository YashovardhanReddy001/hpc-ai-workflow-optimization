"""
Custom CUDA matrix multiplication using CuPy RawKernel.

Provides low-level GPU kernel implementation for benchmarking
against high-level frameworks like PyTorch.
"""

import cupy as cp


kernel_code = r"""
extern "C" __global__
void matmul(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
"""


def run_cupy_kernel(n: int):
    """
    Runs NxN matrix multiplication using custom CUDA kernel.

    Args:
        n (int): Matrix dimension

    Returns:
        elapsed_time (float): Execution time in seconds
        result_sample (cp.ndarray): Top-left 2x2 block
    """

    A = cp.random.randn(n, n, dtype=cp.float32)
    B = cp.random.randn(n, n, dtype=cp.float32)
    C = cp.zeros((n, n), dtype=cp.float32)

    kernel = cp.RawKernel(kernel_code, "matmul")

    block = (16, 16)
    grid = ((n + 15) // 16, (n + 15) // 16)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    kernel(grid, block, (A, B, C, n))

    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0

    return elapsed, C[:2, :2]


if __name__ == "__main__":
    size = 1024
    t, sample = run_cupy_kernel(size)

    print(f"Size: {size}x{size}")
    print(f"Execution time: {t:.6f} seconds")
    print("Sample output:\n", sample)
