# Icicle benchmark: add operation

The benchmark measures the runtime of the vector operation $c[i] = a[i] + operation_factor*b[i]$, where $operation_factor$ is sufficiently large and we can ignore the memory access times.

## Best-Practices

We recommend to run the benchmarks in [ZK-containers](../ZK-containers.md) to save your time and mental energy.

## Run benchmark

Inside the container,

```sh
./compile.sh
./run.sh
```

