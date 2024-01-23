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
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Icicle from a GitHub repository
RUN git clone https://github.com/ingonyama-zk/icicle.git  /opt/icicle 

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

RUN git config --global --add safe.directory /opt/icicle

# Define a build-time variable for the runner version
ARG RUNNER_VERSION=2.311.0
ARG RUNNER_TOKEN=NEEDTOSET

# Download & install the GitHub Actions runner
RUN cd /home/runner && \
     curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
     tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
     rm ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
     ./config.sh --url https://github.com/ingonyama-zk/zk-benchmarks --token ${RUNNER_TOKEN} --name "docker-runner" --unattended --replace

# Set the entry point
WORKDIR /home/runner
#ENTRYPOINT [".run.sh --once"]
CMD ["/bin/bash"]




