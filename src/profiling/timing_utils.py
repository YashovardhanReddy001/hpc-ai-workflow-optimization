"""
Reusable GPU/CPU timing utilities.
"""

import time
import torch


def cpu_timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result


def gpu_timer(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = func(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0

    return elapsed, result
