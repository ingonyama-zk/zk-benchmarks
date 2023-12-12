#define CURVE_BN254     1
#define CURVE_BLS12_381 2
#define CURVE_BLS12_377 3

#define CURVE CURVE_BN254

#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>
#include <benchmark/benchmark.h>
#include "icicle/primitives/field.cuh"
#include "icicle/utils/storage.cuh"
#include "icicle/primitives/projective.cuh"

#include "icicle/appUtils/msm/msm.cu"

#if CURVE == CURVE_BN254

#include "icicle/curves/bn254/curve_config.cuh"    
using namespace BN254;
const std::string curve = "BN254";

#elif CURVE == CURVE_BLS12_381

#include "icicle/curves/bls12_381/curve_config.cuh"
using namespace BLS12_381;
const std::string curve = "BLS12-381";

#elif CURVE == CURVE_BLS12_377

#include "icicle/curves/bls12_377/curve_config.cuh"
using namespace BLS12_377;
const std::string curve = "BLS12-377";
    
#endif

const unsigned max_msm_size = 1<<22;
unsigned bucket_factor = 12;

// on-host data
scalar_t* scalars;
affine_t* points;
projective_t result;

// on-device data
scalar_t* scalars_d;
affine_t* points_d;
projective_t* result_d;

nvmlDevice_t device;
cudaStream_t stream;

static void BM_msm(benchmark::State& state) {
  const uint32_t msm_size=state.range(0);  
  bool on_device = true;
  bool big_triangle = false;
  for (auto _ : state) {
    large_msm<scalar_t, projective_t, affine_t>(scalars_d, points_d, msm_size, result_d, on_device, big_triangle, bucket_factor, stream);
    // cudaDeviceSynchronize();
  }
  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}

BENCHMARK(BM_msm)->MinTime(30.)
  ->Arg(1<<10)
  ->Arg(1<<11)
  ->Arg(1<<12)
  ->Arg(1<<13)
  ->Arg(1<<14)
  ->Arg(1<<15)
  ->Arg(1<<16)
  ->Arg(1<<17)
  ->Arg(1<<18)
  ->Arg(1<<19)
  ->Arg(1<<20)
  ->Arg(1<<21)
  ->Arg(1<<22);

int main(int argc, char** argv) {
  cudaDeviceReset();
  cudaDeviceProp deviceProperties;
  int deviceId=0;
  cudaGetDeviceProperties(&deviceProperties, deviceId);
  std::string gpu_full_name = deviceProperties.name;
  std::cout << gpu_full_name << std::endl;
  std::string gpu_name = gpu_full_name;
  int gpu_clock_mhz = deviceProperties.clockRate/1000.;

  nvmlInit();
  nvmlDeviceGetHandleByIndex(0, &device);  // for GPU 0

  std::cout << "Setting host data" << std::endl;
  
  scalars = (scalar_t*) malloc(sizeof(scalar_t) * max_msm_size);
  points = (affine_t*)malloc(sizeof(affine_t) * max_msm_size);
  for (unsigned i = 0; i < max_msm_size; i++) {
    points[i] = (i % max_msm_size < 10) ? projective_t::to_affine(projective_t::rand_host()) : points[i - 10];
    scalars[i] = scalar_t::rand_host();
  }

  std::cout << "Moving data to device" << std::endl;

  cudaMalloc(&scalars_d, sizeof(scalar_t) * max_msm_size);
  cudaMalloc(&points_d, sizeof(affine_t) * max_msm_size);
  cudaMalloc(&result_d, sizeof(projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * max_msm_size, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_t) * max_msm_size, cudaMemcpyHostToDevice);


  std::cout << "Running benchmark" << std::endl;

  cudaStreamCreate(&stream);

  // Run all benchmarks 
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("team", "Ingonyama");
  ::benchmark::AddCustomContext("project", "Icicle");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", curve);
  ::benchmark::AddCustomContext("comment", "on-device API");
  ::benchmark::AddCustomContext("coefficient_C", std::to_string(bucket_factor));
  ::benchmark::RunSpecifiedBenchmarks();

  cudaFree(scalars_d);
  cudaFree(points_d);
  cudaFree(result_d);
  free(scalars);
  free(points);
  return 0;
}
