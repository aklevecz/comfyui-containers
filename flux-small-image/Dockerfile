FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a directory for ComfyUI
WORKDIR /workspace
RUN mkdir -p ComfyUI
WORKDIR /workspace/ComfyUI

# Copy entrypoint script
COPY entrypoint.sh /workspace/
RUN chmod +x /workspace/entrypoint.sh && dos2unix /workspace/entrypoint.sh

# Expose port for web UI
EXPOSE 8188

# Set the entrypoint to our startup script
ENTRYPOINT ["/workspace/entrypoint.sh"]