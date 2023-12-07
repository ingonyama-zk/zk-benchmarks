#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <nvml.h>
#include <benchmark/benchmark.h>
#include "icicle/primitives/field.cuh"
#include "icicle/utils/storage.cuh"
#include "icicle/primitives/projective.cuh"
#include "icicle/curves/bn254/curve_config.cuh"
#include "ve_mod_mult.cuh"
 
using namespace BN254;

typedef projective_t T;
const unsigned nof_add = 100;
unsigned nof_elements;
T *vec_a;
T *vec_b;
T *d_vec_b;
T *d_vec_a, *d_result;
nvmlDevice_t device;

static void BM_add(benchmark::State& state) {  
  for (auto _ : state) {
    vector_add<T, T, nof_add>(d_vec_a, d_vec_b, d_result, nof_elements);
    cudaDeviceSynchronize();
  }
  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}

BENCHMARK(BM_add)->MinTime(30.);

int main(int argc, char** argv) {
  cudaDeviceReset();
  cudaDeviceProp deviceProperties;
  int deviceId=0;
  cudaGetDeviceProperties(&deviceProperties, deviceId);
  std::string gpu_full_name = deviceProperties.name;
  std::cout << gpu_full_name << std::endl;
  std::string gpu_name;
  if (gpu_full_name.find("3090") != std::string::npos) {
    gpu_name = "RTX 3090";
  } else if (gpu_name.find("4090") != std::string::npos) {
    gpu_name = "RTX 4090";
  } else {
        std::cout << "unrecognized GPU" << std::endl;
  }
  int gpu_clock_mhz = deviceProperties.clockRate/1000.;

  nvmlInit();
  nvmlDeviceGetHandleByIndex(0, &device);  // for GPU 0

  std::cout << "Setting host data" << std::endl;
  nof_elements = 1 << 25;  
  vec_a = (T*)malloc(sizeof(T) * nof_elements);
  vec_b = (T*)malloc(sizeof(T) * nof_elements);
  for (unsigned i = 0; i < (1 << 10); i++) {
    vec_a[i] = T::rand_host();
    vec_b[i] = T::rand_host();
  }
  for (unsigned i = 1; i < (nof_elements >> 10); i++) {
    memcpy((void *)(vec_a + (i << 10)), (void *)(vec_a + ((i-1) << 10)), sizeof(T) << 10);
    memcpy((void *)(vec_b + (i << 10)), (void *)(vec_b + ((i-1) << 10)), sizeof(T) << 10);
  }
  // Allocate memory on the device for the input vectors, the output vector, and the modulus
  std::cout << "Moving data to device" << std::endl;
  cudaMalloc(&d_vec_a, nof_elements * sizeof(T));
  cudaMalloc(&d_vec_b, nof_elements * sizeof(T));
  cudaMalloc(&d_result, nof_elements * sizeof(T));

  // Copy the input vectors and the modulus from the host to the device
  cudaMemcpy(d_vec_a, vec_a, nof_elements * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec_b, vec_b, nof_elements * sizeof(T), cudaMemcpyHostToDevice);
  std::cout << "Running benchmark" << std::endl;

  // Run all benchmarks
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("project", "ICICLE");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", "BN254");
  ::benchmark::AddCustomContext("comment", "on-device API");
  ::benchmark::RunSpecifiedBenchmarks();


  cudaFree(d_vec_a);
  cudaFree(d_vec_b);
  cudaFree(d_result);
  free(vec_a);
  free(vec_b);
}
