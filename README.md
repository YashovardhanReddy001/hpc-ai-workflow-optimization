# HPC–AI Workflow Optimization Framework

## Motivation
Investigate CPU–GPU crossover behavior for large-scale matrix workloads and design a regression-based execution cost model.

## Problem Statement
Matrix multiplication workloads exhibit varying performance characteristics depending on memory size, compute intensity, and device architecture. This project explores performance tradeoffs and builds a learned cost model to predict optimal execution placement (CPU vs GPU).

## Components
- NumPy CPU baseline
- CuPy (cuBLAS-backed) GPU acceleration
- FP16 mixed precision optimization
- Regression-based CPU/GPU cost modeling
- torch.profiler kernel-level analysis
- Docker-based reproducible benchmarking

## Key Results
- 8.9× end-to-end speedup at 4000×4000 matrix scale
- 50% memory reduction via FP16
- Identified crossover threshold between CPU and GPU execution

## Experimental Setup
- Tesla T4 GPU (Google Colab)
- Python 3.x
- PyTorch + CuPy

## Future Work
- Multi-feature cost modeling
- Auto-tuning block sizes
- Distributed scaling benchmarks
