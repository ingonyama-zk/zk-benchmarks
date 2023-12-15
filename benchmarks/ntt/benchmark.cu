#define CURVE_BN254     1
#define CURVE_BLS12_381 2
#define CURVE_BLS12_377 3


// CURVE_BLS12_381 seems to be hardcoded in some ntt, at least in dev branch
#define CURVE CURVE_BLS12_381

#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>
#include <benchmark/benchmark.h>
#include "icicle/appUtils/ntt/ntt.cuh"

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

// Operate on scalars
typedef scalar_t S;
typedef scalar_t E;

const unsigned nof_ntts = 1;

// on-host data
E* elements;
  
void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
  for (unsigned i = 0; i < ntt_size * nof_ntts; i++) {
    elements[i] = E::rand_host();
  }
}

// on-device data

S* d_twiddles;
E* d_elements;

nvmlDevice_t device;
cudaStream_t stream;

static void DoSetup(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  unsigned ntt_size = 1 << log_ntt_size;
  // std::cout << "DoSetup: " << log_ntt_size << std::endl;
  d_twiddles = fill_twiddle_factors_array((uint32_t) ntt_size, S::omega(log_ntt_size), stream); // Scalars
  elements = (E*) malloc(sizeof(E) * ntt_size);
  initialize_input(ntt_size, 1, elements );
  cudaMallocAsync(&d_elements, sizeof(E) * ntt_size, stream);
  cudaMemcpyAsync(d_elements, elements, sizeof(E) * ntt_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}

static void DoTeardown(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  // std::cout << "DoTeardown: " << log_ntt_size << std::endl;
  cudaFreeAsync(d_elements, stream);
  cudaFreeAsync(d_twiddles, stream);
  cudaStreamSynchronize(stream);
  free(elements);
}

static void BM_ntt(benchmark::State& state) {
  const unsigned log_ntt_size = state.range(0);  
  const unsigned ntt_size = 1 << log_ntt_size;
  // const unsigned batch_size = 1 * ntt_size;
  bool inverse = false;
  S* _null = nullptr;
  
  for (auto _ : state) {
    // for (unsigned i = 0; i < nof_ntts; i++) {
    ntt_inplace_batch_template(d_elements, d_twiddles, ntt_size, 1, inverse, false, _null, stream, false);
    // }
    cudaStreamSynchronize(stream);
  }

  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}

BENCHMARK(BM_ntt) //->MinTime(30.)
  ->Arg(10)
  ->Arg(11)
  ->Arg(12)
  ->Arg(13)
  ->Arg(14)
  ->Arg(15)
  ->Arg(16)
  ->Arg(17)
  ->Arg(18)
  ->Arg(19)
  ->Arg(20)
  ->Arg(21)
  ->Arg(22)
  ->Arg(23)
  ->Arg(24)
  ->Arg(25)
  ->Arg(26)
  ->Arg(27)
  ->Setup(DoSetup)->Teardown(DoTeardown)
  ;
  

  
 

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

  cudaStreamCreate(&stream);
  
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("team", "Ingonyama");
  ::benchmark::AddCustomContext("project", "Icicle");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", curve);
  ::benchmark::AddCustomContext("comment", "on-device API");
  // ::benchmark::AddCustomContext("coefficient_C", std::to_string(bucket_factor));
  ::benchmark::RunSpecifiedBenchmarks();

  cudaStreamDestroy(stream);
  return 0;
}
