# Use Ubuntu as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# nsight-systems-12.2 \
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    cmake \
    protobuf-compiler \
    build-essential \
    libboost-all-dev \
    jq \
    postgresql-client \
    pkg-config \
    libssl-dev \
    llvm \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for your app and data
RUN mkdir /app && chmod 755 /app

# Set a working directory (optional)
WORKDIR /app

# Benchmarking in C++
# RUN git clone https://github.com/google/benchmark.git /opt/benchmark \
#     && cd /opt/benchmark \
#     && cmake -E make_directory "build" \
#     && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -S . -B "build" \
#     && cmake --build "build" --config Release \
#     && cmake --build "build" --config Release --target install

# Create a new user "runner" with sudo privileges
#RUN useradd -m runner && echo "runner:runner" | chpasswd && adduser runner sudo

# Switch to the new user
# USER runner

# Now all commands will be run as "runner"

ENV BENCHMARK_REPO=/app/icicle/

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
# extend cargo with command criterion and enable export from criterion to json
#RUN cargo install cargo-criterion

RUN git clone https://github.com/ingonyama-zk/icicle.git  /app/icicle
# RUN cd /home/runner/icicle/wrappers/rust && cargo build 

# clone https://github.com/ingonyama-zk/zk-benchmarks, branch CI-initial to /app/zk-benchmarks
RUN git clone -b CI-initial https://github.com/ingonyama-zk/zk-benchmarks /app/zk-benchmarks

# Set the entry point
#WORKDIR /home/runner
#CMD ["/bin/bash"]
#CMD ["ls -la"]




