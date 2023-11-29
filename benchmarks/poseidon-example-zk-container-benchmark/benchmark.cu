#include <vector>
#include <string>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include "curves/bls12_381/curve_config.cuh"
#include "curves/bls12_381/poseidon.cu"


const uint32_t arity = 4;
const uint32_t max_tree_height = 20;
const uint32_t max_nof_blocks = 1 << max_tree_height;
BLS12_381::scalar_t* blocks;
BLS12_381::scalar_t* hashes;

nvmlDevice_t device;

cudaStream_t stream;
static Poseidon<BLS12_381::scalar_t> poseidon(arity, stream);

static void BM_Poseidon(benchmark::State& state) {  
  const uint32_t tree_height=state.range(0);
  const uint32_t nof_blocks = 1 << tree_height;
  for (auto _ : state) {
    poseidon.hash_blocks(blocks, nof_blocks, hashes, Poseidon<BLS12_381::scalar_t>::HashType::MerkleTree, stream);
  }
  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}
BENCHMARK(BM_Poseidon)->MinTime(1.)
  ->Arg(8)
  ->Arg(9)
  ->Arg(10)
  ->Arg(11)
  ->Arg(12)
  ->Arg(13)
  ->Arg(14)
  ->Arg(15);

int main(int argc, char** argv) {
  cudaDeviceProp deviceProperties;
  int deviceId=0;
  cudaGetDeviceProperties(&deviceProperties, deviceId);
  
  std::string gpu_full_name = deviceProperties.name;
  std::string gpu_name;
  if (gpu_full_name.find("3090") != std::string::npos) {
    gpu_name = "RTX 3090";
  } else if (gpu_name.find("4090") != std::string::npos) {
    gpu_name = "RTX 4090";
  } else {
        std::cout << "unrecognized GPU" << std::endl;
  }
  int gpu_clock_mhz = deviceProperties.clockRate/1000.;

  // custom counter for device temperature and power

  nvmlInit();
  nvmlDeviceGetHandleByIndex(0, &device);  // for GPU 0

  // Initialization code here
  cudaStreamCreate(&stream);
  blocks = static_cast<BLS12_381::scalar_t*>(malloc(max_nof_blocks * arity * sizeof(BLS12_381::scalar_t)));
  hashes = static_cast<BLS12_381::scalar_t*>(malloc(max_nof_blocks * sizeof(BLS12_381::scalar_t)));  
  BLS12_381::scalar_t d = BLS12_381::scalar_t::zero();
  for (uint32_t i = 0; i < max_nof_blocks * arity; i++) {
    blocks[i] = d;
    d = d + BLS12_381::scalar_t::one();
  }
       
  // Run all benchmarks
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("project", "ICICLE");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", "BLS12-381");
  ::benchmark::AddCustomContext("comment", "on-host API");

  
  
  
  ::benchmark::RunSpecifiedBenchmarks();

  
  // Cleanup code here
  cudaStreamDestroy(stream);
  free(hashes);
  free(blocks);
  nvmlShutdown();
  return 0;
}