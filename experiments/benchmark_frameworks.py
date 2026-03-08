"""
Benchmark harness for comparing matrix multiplication backends.

Backends:
- PyTorch
- CuPy RawKernel
- Numba CUDA
"""

from src.baseline.pytorch_matmul import run_pytorch_matmul
from src.kernels.cupy_rawkernel_matmul import run_cupy_kernel
from src.kernels.numba_matmul import run_numba_kernel


def run_benchmarks(size_list):
    results = []

    for n in size_list:
        print(f"\nRunning benchmarks for size {n}x{n}")

        pytorch_time, _ = run_pytorch_matmul(n)
        print(f"PyTorch: {pytorch_time:.6f}s")

        cupy_time, _ = run_cupy_kernel(n)
        print(f"CuPy RawKernel: {cupy_time:.6f}s")

        numba_time, _ = run_numba_kernel(n)
        print(f"Numba CUDA: {numba_time:.6f}s")

        results.append({
            "size": n,
            "pytorch": pytorch_time,
            "cupy": cupy_time,
            "numba": numba_time
        })

    return results


if __name__ == "__main__":
    matrix_sizes = [256, 512, 1024]

    results = run_benchmarks(matrix_sizes)

    print("\nBenchmark Results:")
    for r in results:
        print(r)
