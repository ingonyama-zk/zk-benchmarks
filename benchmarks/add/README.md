# Icicle benchmark: add operation

The benchmark measures the runtime of the vector operation $c[i] = a[i] + N*b[i]$, where $N$ is sufficiently large and we can ignore the memory access times.

## Best-Practices

We recommend to run the benchmarks in [ZK-containers](../ZK-containers.md) to save your time and mental energy.

## Run benchmark

Inside the container, run

```sh
rm -rf build
mkdir -p build
cmake -S . -B build
cmake --build build
./build/benchmark
```
