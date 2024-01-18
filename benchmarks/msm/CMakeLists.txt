cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
endif ()
project(zk-benchmarks LANGUAGES CUDA CXX)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#set(CMAKE_CUDA_FLAGS_RELEASE "")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

include_directories("/opt/icicle")
include_directories("/opt/benchmark/include")

add_executable(
  benchmark
  benchmark.cu
)

find_library(BENCHMARK_LIBRARY benchmark PATHS /usr/local/lib)
find_library(NVML_LIBRARY nvidia-ml PATHS /usr/local/cuda/targets/x86_64-linux/lib/stubs/ )
target_link_libraries(benchmark ${BENCHMARK_LIBRARY} ${NVML_LIBRARY})
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

