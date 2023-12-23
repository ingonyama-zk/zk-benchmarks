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
#include "/icicle/icicle/curves/bn254_params.cuh"
#include "/icicle/icicle/curves/bls12_381_params.cuh"
#include "/icicle/icicle/curves/bls12_377_params.cuh"
#include "/icicle/icicle/curves/bw6_761_params.cuh"
typedef Field<bn254::fp_config> bn254_scalar_t;
typedef Field<bn254::fq_config> bn254_point_field_t;
typedef Field<bls12_381::fp_config> bls12_381_scalar_t;
typedef Field<bls12_381::fq_config> bls12_381_point_field_t;
typedef Field<bls12_377::fp_config> bls12_377_scalar_t;
typedef Field<bls12_377::fq_config> bls12_377_point_field_t;
// typedef Field<bw6_761::fp_config> bw6_761_scalar_t;


// const std::string curve = "BLS12-377";



#define MAX_THREADS_PER_BLOCK 256

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

const std::string multiply_type="point_field";
typedef bls12_377_scalar_t S;
//typedef point_field_t S;

const unsigned nof_mult = 100;
unsigned nof_elements = 1 << 25;  

nvmlDevice_t device;


// best-practice to reduce redundant code:
// https://github.com/google/benchmark/issues/698

template <typename T>
class MyTemplatedFixture : public benchmark::Fixture {
  public:
    // host data
    T *h_vec_a;
    T *h_vec_b;
    // device data
    T *d_vec_a;
    T *d_vec_b;
    T *d_result;
    int* array;
    size_t size;

    void SetUp(const ::benchmark::State& state) override {
        size = static_cast<size_t>(state.range(0));
        // expect size is a power of 2 and size > 10
        array = new int[size];
        h_vec_a = new T[size];
        h_vec_b = new T[size];
        // Initialize the arrays with random data
        for (unsigned i = 0; i < (1 << 10); i++) {
          h_vec_a[i] = T::rand_host();
          h_vec_b[i] = T::rand_host();
        }
        for (unsigned i = 1; i < (size >> 10); i++) {
          memcpy((void *)(h_vec_a + (i << 10)), (void *)(h_vec_a + ((i-1) << 10)), sizeof(T) << 10);
          memcpy((void *)(h_vec_b + (i << 10)), (void *)(h_vec_b + ((i-1) << 10)), sizeof(T) << 10);
        }
        // Allocate memory on the device 
        // std::cout << "Moving data to device" << std::endl;
        cudaMalloc(&d_vec_a, size * sizeof(T));
        cudaMalloc(&d_vec_b, size * sizeof(T));
        cudaMalloc(&d_result, size * sizeof(T));
        // Copy the data to the device
        cudaMemcpy(d_vec_a, h_vec_a, size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec_b, h_vec_b, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void TearDown(const ::benchmark::State& state) override {
        delete[] array;
        cudaFree(d_vec_a);
        cudaFree(d_vec_b);
        cudaFree(d_result);
        delete[] h_vec_a;
        delete[] h_vec_b;
    }
  protected:
    virtual void BenchmarkCase(benchmark::State& state) override {
      for (auto _ : state) {
          // Benchmark code using the array of type T
          vector_mult<T, 100>(d_vec_a, d_vec_b, d_result, size);
          cudaError_t error = cudaDeviceSynchronize();
          if (error != cudaSuccess) {
           fprintf(stderr, "CUDA Error after cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
            // Handle the error
          }
      }
      unsigned int power;
      nvmlDeviceGetPowerUsage(device, &power);
      state.counters["PowerUsage"] = int(1.0e-3 * power);
      unsigned int temperature;
      nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
      state.counters["Temperature"] = int(temperature);
    }
};

// bn254
BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test1, bn254_scalar_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test1)->Name("BN254:scalar")->MinTime(30.)->Arg(nof_elements);

BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test2, bn254_point_field_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test2)->Name("BN254:point_field")->MinTime(30.)->Arg(nof_elements);

// bls12_381
BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test3, bls12_381_scalar_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test3)->Name("BLS12_381:scalar")->MinTime(30.)->Arg(nof_elements);

BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test4, bls12_381_point_field_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test4)->Name("BLS12_381:point_field")->MinTime(30.)->Arg(nof_elements);

// bls12_377
BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test5, bls12_377_scalar_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test5)->Name("BLS12_377:scalar")->MinTime(30.)->Arg(nof_elements);

BENCHMARK_TEMPLATE_DEFINE_F(MyTemplatedFixture, Test6, bls12_377_point_field_t)(benchmark::State& state){MyTemplatedFixture::BenchmarkCase(state);}
BENCHMARK_REGISTER_F(MyTemplatedFixture, Test6)->Name("BLS12_377:point_field")->MinTime(30.)->Arg(nof_elements);


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

  // Run all benchmarks 
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::AddCustomContext("team", "Ingonyama");
  ::benchmark::AddCustomContext("git_repository", "https://github.com/ingonyama-zk/icicle.git");
  ::benchmark::AddCustomContext("project", "Icicle");
  ::benchmark::AddCustomContext("runs_on", gpu_name);
  ::benchmark::AddCustomContext("frequency_MHz", std::to_string(gpu_clock_mhz));
  // ::benchmark::AddCustomContext("uses", curve);
  ::benchmark::AddCustomContext("comment", "on-device API");
  ::benchmark::AddCustomContext("operation_factor", std::to_string(nof_mult));
  ::benchmark::AddCustomContext("vector_size", std::to_string(nof_elements));
  ::benchmark::AddCustomContext("multiply", multiply_type);
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
