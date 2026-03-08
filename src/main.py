"""
Main execution entrypoint for HPC–AI workflow experiments.
"""

from experiments.benchmark_frameworks import run_benchmarks
from experiments.crossover_analysis import run_crossover_analysis


def main():

    print("\nRunning backend benchmark experiment\n")

    sizes = [256, 512, 1024]
    run_benchmarks(sizes)

    print("\nRunning CPU–GPU crossover experiment\n")

    crossover_sizes = [128, 256, 512, 1024, 2048]
    run_crossover_analysis(crossover_sizes)


if __name__ == "__main__":
    main()
