# HPC–AI Workflow Optimization Framework

## Overview
This project investigates CPU–GPU execution tradeoffs for large-scale matrix workloads and develops a cost-aware optimization pipeline integrating custom CUDA kernels, high-level ML frameworks, and mixed-precision inference.

The goal is to:
- Analyze performance crossover behavior across execution backends
- Benchmark multiple GPU programming approaches
- Integrate AI-based surrogate modeling
- Evaluate mixed-precision acceleration strategies

---

## Research Objectives

1. Characterize CPU vs GPU execution crossover behavior
2. Compare custom CUDA kernels against high-level frameworks
3. Measure compute–memory tradeoffs
4. Integrate regression-based execution cost modeling
5. Explore FP16 mixed precision acceleration in AI pipelines

---

## Implemented Backends

### Custom CUDA Kernels
- PyCUDA (SourceModule)
- CuPy RawKernel
- Numba CUDA JIT

### Framework Baselines
- PyTorch (GPU + CUDA events)
- JAX (JIT-compiled matmul)
- NumPy (CPU baseline)

---

## Performance Methodology

- Matrix sizes tested: 128 → 4096
- GPU: NVIDIA Tesla T4 (Colab)
- Timing via CUDA events and synchronized measurements
- JIT warmup runs excluded
- Multiple-run averaging
- Memory footprint analysis

---

## Mixed Precision Optimization

- Integrated FP16 inference using PyTorch AMP
- Measured ~50% memory reduction
- Evaluated precision-performance tradeoffs
- Reduced bandwidth pressure in downstream AI stages

---

## AI Surrogate Modeling

- Implemented deep neural surrogate model for workload transformation
- Integrated tensor conversion pipeline (HPC → Torch → HPC)
- Structured execution pipeline with monitoring utilities
- Designed for future regression-based device selection modeling

---

## Experimental Components

- Modular kernel implementations
- Benchmark harness for framework comparison
- Execution monitoring utilities
- Device-aware execution logic
- Reproducible benchmarking setup

---

## Key Findings

- Custom RawKernel implementations approach framework-level performance for moderate sizes
- Framework-level optimizations (cuBLAS-backed ops) dominate for large matrices
- Mixed precision reduces memory overhead (~50%) with minimal inference degradation
- GPU acceleration benefit depends strongly on workload size and transfer overhead

---

## Future Work

- Train regression-based device selection model
- Extend to distributed MPI workloads
- Kernel-level memory bandwidth profiling
- Auto-tuned block size optimization
- Structured performance dataset logging

---

## Repository Structure

```
hpc-ai-workflow-optimization/
│
├── README.md
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── kernels/
│   │   ├── pycuda_matmul.py
│   │   ├── cupy_rawkernel_matmul.py
│   │   ├── numba_matmul.py
│   │   └── jax_matmul.py
│   │
│   ├── baseline/
│   │   ├── pytorch_matmul.py
│   │   └── numpy_matmul.py
│   │
│   ├── ai/
│   │   ├── deep_model.py
│   │   ├── data_transformer.py
│   │   └── amp_inference.py
│   │
│   ├── profiling/
│   │   ├── timing_utils.py
│   │   └── profiler_analysis.py
│   │
│   ├── cost_model/
│   │   └── regression_cost_model.py
│   │
│   └── main.py
│
├── experiments/
│   ├── benchmark_frameworks.py
│   ├── crossover_analysis.py
│   └── mixed_precision_study.py
│
└── results/
    ├── speedup_plot.png
    ├── framework_comparison.png
    └── crossover_curve.png
```
   
---

## Focus

Designing intelligent systems that optimize, adapt, and reason under computational constraints.
