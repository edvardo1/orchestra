#!/bin/bash

# Check if the user provided N
if [ $# -ne 1 ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

N=$1

# Validate if N is a positive integer
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: N must be a positive integer."
    exit 1
fi

# Execute the command N times - No GPU
echo "Running BFS benchmark WITHOUT GPU acceleration..."

for ((i=1; i<=N; i++)); do
    mix run benchmarks/cooperative/bfs/BFS.exs \
        benchmarks/cooperative/bfs/NYR_input.dat \
        4096 | grep "BFS took:"
done

# Execute the command N times - With GPU
echo "Running BFS benchmark USING GPU acceleration..."


for ((i=1; i<=N; i++)); do
    mix run benchmarks/cooperative/bfs/BFS.exs \
        benchmarks/cooperative/bfs/NYR_input.dat \
        1024 | grep "BFS took:"
done