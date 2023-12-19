#define CURVE_BN254     1
#define CURVE_BLS12_381 2
#define CURVE_BLS12_377 3

#define CURVE CURVE_BLS12_377

#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <nvml.h>
#include </opt/benchmark/include/benchmark/benchmark.h>
#include "/icicle/icicle/primitives/field.cuh"

#if CURVE == CURVE_BN254

#include "/icicle/icicle/curves/bn254/curve_config.cuh"    
using namespace BN254;
const std::string curve = "BN254";

#elif CURVE == CURVE_BLS12_381

#include "/icicle/icicle/curves/bls12_381/curve_config.cuh"
using namespace BLS12_381;
const std::string curve = "BLS12-381";

#elif CURVE == CURVE_BLS12_377

#include "/icicle/icicle/curves/bls12_377/curve_config.cuh"
using namespace BLS12_377;
const std::string curve = "BLS12-377";
    
#endif


#include "/icicle/icicle/appUtils/vector_manipulation/ve_mod_mult.cuh"

template <typename S, int N>
__global__ void vectorMult(S *vec_a, S *vec_b, S *vec_r, size_t n_elments)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elments)
    {
        const S b = vec_b[tid];
        S r = vec_a[tid];
        // #pragma unroll
        for (int i = 0; i < N; i++)
            r = r * b;
        vec_r[tid] = r;
    }
}

template <typename S, int N = 10>
int vector_mult(S *vec_b, S *vec_a, S *vec_result, size_t n_elments)
{
    // Set the grid and block dimensions
    int num_blocks = (int)ceil((float)n_elments / MAX_THREADS_PER_BLOCK);
    int threads_per_block = MAX_THREADS_PER_BLOCK;

    // Call the kernel to perform element-wise modular multiplication
    vectorMult<S, N><<<num_blocks, threads_per_block>>>(vec_a, vec_b, vec_result, n_elments);
    return 0;
}

// typedef scalar_t S;
typedef point_field_t S;

const unsigned nof_mult = 100;
unsigned nof_elements = 1 << 25;  
S *vec_a;
S *vec_b;
S *d_vec_b;
S *d_vec_a, *d_result;
nvmlDevice_t device;

static void BM_mult(benchmark::State& state) {  
  for (auto _ : state) {
    vector_mult<S, nof_mult>(d_vec_a, d_vec_b, d_result, nof_elements);
    cudaDeviceSynchronize();
  }
  unsigned int power;
  nvmlDeviceGetPowerUsage(device, &power);
  state.counters["PowerUsage"] = int(1.0e-3 * power);
  unsigned int temperature;
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  state.counters["Temperature"] = int(temperature);
}

BENCHMARK(BM_mult)->MinTime(60.);

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
  
  vec_a = (S*)malloc(sizeof(S) * nof_elements);
  vec_b = (S*)malloc(sizeof(S) * nof_elements);
  for (unsigned i = 0; i < (1 << 10); i++) {
    vec_a[i] = S::rand_host();
    vec_b[i] = S::rand_host();
  }
  for (unsigned i = 1; i < (nof_elements >> 10); i++) {
    memcpy((void *)(vec_a + (i << 10)), (void *)(vec_a + ((i-1) << 10)), sizeof(S) << 10);
    memcpy((void *)(vec_b + (i << 10)), (void *)(vec_b + ((i-1) << 10)), sizeof(S) << 10);
  }
  // Allocate memory on the device for the input vectors, the output vector, and the modulus
  std::cout << "Moving data to device" << std::endl;
  cudaMalloc(&d_vec_a, nof_elements * sizeof(S));
  cudaMalloc(&d_vec_b, nof_elements * sizeof(S));
  cudaMalloc(&d_result, nof_elements * sizeof(S));

  // Copy the input vectors and the modulus from the host to the device
  cudaMemcpy(d_vec_a, vec_a, nof_elements * sizeof(S), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec_b, vec_b, nof_elements * sizeof(S), cudaMemcpyHostToDevice);
  std::cout << "Running benchmark" << std::endl;

  // Run all benchmarks 
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("team", "Ingonyama");
  ::benchmark::AddCustomContext("project", "Icicle");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  ::benchmark::AddCustomContext("uses", curve);
  ::benchmark::AddCustomContext("comment", "on-device API");
  ::benchmark::AddCustomContext("operation_factor", std::to_string(nof_mult));
  ::benchmark::AddCustomContext("vector_size", std::to_string(nof_elements));
  ::benchmark::RunSpecifiedBenchmarks();

  cudaFree(d_vec_a);
  cudaFree(d_vec_b);
  cudaFree(d_result);
  free(vec_a);
  free(vec_b);
}
