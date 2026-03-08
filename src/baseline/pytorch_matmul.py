"""
PyTorch GPU/CPU baseline matrix multiplication.

Provides clean benchmarking with proper CUDA event timing.
"""

import torch


def run_pytorch_matmul(n: int, device: str = None):
    """
    Runs NxN matrix multiplication using PyTorch.

    Args:
        n (int): Matrix dimension.
        device (str): "cuda" or "cpu". Auto-detects if None.

    Returns:
        elapsed_time (float): Execution time in seconds.
        result_sample (torch.Tensor): Top-left 2x2 block.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    A = torch.randn((n, n), dtype=torch.float32, device=device)
    B = torch.randn((n, n), dtype=torch.float32, device=device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        C = torch.matmul(A, B)

        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end) / 1000.0  # ms → seconds
    else:
        import time
        t0 = time.time()
        C = torch.matmul(A, B)
        elapsed = time.time() - t0

    return elapsed, C[:2, :2]


if __name__ == "__main__":
    size = 1024
    time_taken, sample = run_pytorch_matmul(size)
    print(f"Size: {size}x{size}")
    print(f"Execution time: {time_taken:.6f} seconds")
    print("Sample output:\n", sample)
