# Use an official CUDA base image with Ubuntu
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (match CUDA version to the base image)
# For official binaries, see https://pytorch.org/get-started/locally/
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Create a working directory
WORKDIR /app

# Copy everything into /app (including your .py files)
COPY . /app

# Example command to run the benchmark script
CMD ["python3", "benchmark.py"]
