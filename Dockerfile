# Use Ubuntu as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install necessary packages

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

# Clone Icicle from a GitHub repository
#RUN git clone https://github.com/ingonyama-zk/icicle.git  /opt/icicle 

# Benchmarking in C++
RUN git clone https://github.com/google/benchmark.git /opt/benchmark \
    && cd /opt/benchmark \
    && cmake -E make_directory "build" \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -S . -B "build" \
    && cmake --build "build" --config Release \
    && cmake --build "build" --config Release --target install

# Create a new user "runner" with sudo privileges
RUN useradd -m runner && echo "runner:runner" | chpasswd && adduser runner sudo

# Switch to the new user
USER runner

# Now all commands will be run as "runner"

ENV BENCHMARK_REPO=/home/runner/icicle/

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/runner/.cargo/bin:${PATH}"
RUN cargo install cargo-criterion


RUN git clone https://github.com/ingonyama-zk/icicle.git  /home/runner/icicle 
RUN cd /home/runner/icicle/wrappers/rust && cargo build 

#RUN git config --global --add safe.directory /opt/icicle

# Set the entry point
WORKDIR /home/runner
CMD ["/bin/bash"]
#CMD ["ls -la"]




