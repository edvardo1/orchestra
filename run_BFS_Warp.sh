#!/bin/bash

# Check if the user provided N and benchmark name
if [ $# -ne 2 ]; then
    echo "Usage: $0 <N> <benchmark_file>"
    echo "Example: $0 10 facebook_combined.dat"
    exit 1
fi

N=$1
BENCHMARK=$2

# Validate if N is a positive integer
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: N must be a positive integer."
    exit 1
fi

# Benchmark path
BENCHMARK_PATH="benchmarks/cooperative/bfs/$BENCHMARK"

# Check if benchmark file exists
if [ ! -f "$BENCHMARK_PATH" ]; then
    echo "Error: Benchmark file '$BENCHMARK_PATH' not found."
    exit 1
fi

# Execute the command N times - No GPU
echo "Running BFS-Warp benchmark WITHOUT GPU acceleration..."
echo "Benchmark: $BENCHMARK"
echo "----------------------------------------"

for ((i=1; i<=N; i++)); do
    mix run benchmarks/cooperative/bfs/BFS-Warp.exs \
        "$BENCHMARK_PATH" \
        5000 | grep "BFS-Warp took:"
done

echo ""

# Execute the command N times - With GPU
echo "Running BFS-Warp benchmark USING GPU acceleration..."
echo "Benchmark: $BENCHMARK"
echo "----------------------------------------"

for ((i=1; i<=N; i++)); do
    mix run benchmarks/cooperative/bfs/BFS-Warp.exs \
        "$BENCHMARK_PATH" \
        1024 | grep "BFS-Warp took:"
done