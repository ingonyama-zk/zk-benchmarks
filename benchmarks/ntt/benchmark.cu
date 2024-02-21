#include <stdio.h>
#include <iostream>
#include <string>
#include <nvml.h>
#include <benchmark/benchmark.h>


// select the curve
#define CURVE_ID 1
// include NTT template
#include "appUtils/ntt/ntt.cu"
#include "appUtils/ntt/kernel_ntt.cu"
using namespace curve_config;
using namespace ntt;

#if CURVE_ID == BN254
const std::string curve = "BN254";
#elif CURVE_ID == BLS12_381
const std::string curve = "BLS12-381";
#elif CURVE_ID == BLS12_377
const std::string curve = "BLS12-377";
#endif

typedef scalar_t S;
typedef scalar_t E;

const unsigned nof_ntts = 1;

// on-host data
E* elements;

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
  E::RandHostMany(elements, ntt_size * nof_ntts);
}

// on-device data
E* d_input;
E* d_output;

nvmlDevice_t device;
NTTConfig<S> config = DefaultNTTConfig<S>();

static void DoSetup(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  unsigned ntt_size = 1 << log_ntt_size;
  config.ntt_algorithm = NttAlgorithm::MixedRadix; 
  config.batch_size = nof_ntts;
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  elements = (E*) malloc(sizeof(E) * ntt_size);
  initialize_input(ntt_size, nof_ntts, elements );
  cudaMallocAsync(&d_input, sizeof(E) * ntt_size, config.ctx.stream);
  cudaMallocAsync(&d_output, sizeof(E) * ntt_size, config.ctx.stream);
  cudaMemcpyAsync(d_input, elements, sizeof(E) * ntt_size, cudaMemcpyHostToDevice, config.ctx.stream);
  cudaStreamSynchronize(config.ctx.stream);
}

static void DoTeardown(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  cudaFreeAsync(d_input, config.ctx.stream);
  cudaFreeAsync(d_output, config.ctx.stream);
  cudaStreamSynchronize(config.ctx.stream);
  free(elements);
}

static void BM_ntt(benchmark::State& state) {
  const unsigned log_ntt_size = state.range(0);  
  const unsigned ntt_size = 1 << log_ntt_size;
  
  for (auto _ : state) {
    cudaError_t err = NTT<S, E>(d_input, ntt_size, NTTDir::kForward, config, d_output);
  }

  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}


const unsigned max_log_ntt_size = 27;
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
  const S basic_root = S::omega(max_log_ntt_size);
  InitDomain(basic_root, config.ctx);
  
  config.batch_size = nof_ntts;
  // all data is on device, blocking calls
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  config.is_async = false;
  
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("team", "Ingonyama");
  ::benchmark::AddCustomContext("project", "Icicle");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", curve);
  ::benchmark::AddCustomContext("comment", "on-device MixedRadix");
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
