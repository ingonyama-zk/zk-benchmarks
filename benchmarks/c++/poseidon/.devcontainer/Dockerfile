# Make sure NVIDIA Container Toolkit is installed on your host

# Use the specified base image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    protobuf-compiler \
    curl \
    build-essential \
    git \
    libboost-all-dev \
    jq \
    postgresql-client \
    python3-pip \    
    && rm -rf /var/lib/apt/lists/*

# Clone Icicle from a GitHub repository
RUN git clone https://github.com/ingonyama-zk/icicle.git  /icicle

# Benchmarking in C++
RUN git clone https://github.com/google/benchmark.git /opt/benchmark \
    && cd /opt/benchmark \
    && cmake -E make_directory "build" \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -S . -B "build" \
    && cmake --build "build" --config Release \
    && cmake --build "build" --config Release --target install

# Install Python dependencies
# RUN pip install poseidon-hash

# Set the working directory in the container
WORKDIR /icicle-benchmark
 COPY . .
 RUN mkdir -p build && \
     cmake -S . -B build && \
     cmake --build build

# Specify the default command for the container
CMD ["/bin/bash"]
