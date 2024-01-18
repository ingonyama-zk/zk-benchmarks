#include <stdio.h>
#include <iostream>
#include <string>
#include <nvml.h>
#include <benchmark/benchmark.h>

// select the curve
#define CURVE_ID 1
// include NTT template
#include "/opt/icicle/icicle/appUtils/ntt/ntt.cu"
using namespace curve_config;

#if CURVE_ID == BN254
const std::string curve = "BN254";
#elif CURVE_ID == BLS12_381
const std::string curve = "BLS12-381";
#elif CURVE_ID == BLS12_377
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

E* d_input;
E* d_output;

nvmlDevice_t device;
auto ctx = device_context::get_default_device_context();
ntt::NTTConfig<S> config=ntt::GetDefaultNTTConfig();

static void DoSetup(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  unsigned ntt_size = 1 << log_ntt_size;
  elements = (E*) malloc(sizeof(E) * ntt_size);
  initialize_input(ntt_size, nof_ntts, elements );
  cudaMallocAsync(&d_input, sizeof(E) * ntt_size, ctx.stream);
  cudaMallocAsync(&d_output, sizeof(E) * ntt_size, ctx.stream);
  cudaMemcpyAsync(d_input, elements, sizeof(E) * ntt_size, cudaMemcpyHostToDevice, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
}

static void DoTeardown(const benchmark::State& state) {
  unsigned log_ntt_size = state.range(0);  
  cudaFreeAsync(d_input, ctx.stream);
  cudaFreeAsync(d_output, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
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
    // ntt_inplace_batch_template(d_elements, d_twiddles, ntt_size, 1, inverse, false, _null, stream, false);
    cudaError_t err = ntt::NTT<S, E>(d_input, ntt_size, ntt::NTTDir::kForward, config, d_output);
    // }
    cudaStreamSynchronize(ctx.stream);
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

  cudaStreamCreate(&ctx.stream);
  // the next line is valid only for CURVE_ID 1 (will add support for other curves soon)
#if CURVE_ID == BN254  
  scalar_t rou = scalar_t{ {0x53337857, 0x53422da9, 0xdbed349f, 0xac616632, 0x6d1e303, 0x27508aba, 0xa0ed063, 0x26125da1} };
#else
#error "Unsupported curve"
#endif  
  ntt::InitDomain(rou, ctx);
  
  config.batch_size = nof_ntts;
  config.ctx.stream = ctx.stream;
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
  ::benchmark::AddCustomContext("comment", "on-device API");
  ::benchmark::RunSpecifiedBenchmarks();

  cudaStreamDestroy(ctx.stream);
  return 0;
}
