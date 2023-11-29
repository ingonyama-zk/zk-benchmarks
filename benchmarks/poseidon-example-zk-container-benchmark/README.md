# Icicle benchmark: Poseidon hash

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

