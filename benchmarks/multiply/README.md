# Icicle benchmark: multiply operation

The benchmark measures the runtime of the vector operation $c[i] = a[i] * b[i]^n$, where $n$ is sufficiently large and we can ignore the memory access times.

## Best-Practices

We recommend to run the benchmarks in [ZK-containers](../ZK-containers.md) to save your time and mental energy.

## Targets

We designed the benchmark to estimate how many operations per second a given GPU can sustain.

## Run benchmark

Inside the container,

```sh
./compile.sh
./run.sh
```


