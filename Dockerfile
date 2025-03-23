# Use an official CUDA base image with Ubuntu
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (match CUDA version to the base image).
# The cu118 wheel is currently the latest for many PyTorch releases,
# but confirm it aligns with your container CUDA version or use a local wheel.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install Flask for the REST API
RUN pip3 install --no-cache-dir tornado
RUN pip3 install --no-cache-dir joblib
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir eth_account

# Create a working directory
WORKDIR /app

# Copy everything (including .py files) into /app
COPY ./prover.py /app
COPY ./common.py /app
COPY ./verifier_service.py /app

# Expose port 12121, 14141 for the servers
EXPOSE 12121
EXPOSE 14141

# Run the Prover application by default
CMD ["python3", "prover.py"]
