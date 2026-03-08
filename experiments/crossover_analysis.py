"""
CPU vs GPU crossover analysis for matrix multiplication workloads.
"""

from src.baseline.pytorch_matmul import run_pytorch_matmul
import numpy as np


def run_crossover_analysis(sizes):
    results = []

    print("\nRunning CPU vs GPU crossover analysis\n")

    for n in sizes:
        print(f"Matrix size: {n}x{n}")

        cpu_time, _ = run_pytorch_matmul(n, device="cpu")
        gpu_time, _ = run_pytorch_matmul(n, device="cuda")

        print(f"CPU time: {cpu_time:.6f}s")
        print(f"GPU time: {gpu_time:.6f}s")

        results.append({
            "size": n,
            "cpu": cpu_time,
            "gpu": gpu_time
        })

    return results


if __name__ == "__main__":

    sizes = [128, 256, 512, 1024, 2048]

    data = run_crossover_analysis(sizes)

    print("\nResults:")
    for r in data:
        print(r)
