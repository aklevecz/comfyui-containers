FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for ComfyUI
WORKDIR /workspace
RUN mkdir -p ComfyUI
WORKDIR /workspace/ComfyUI

# Clone the latest ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

# Install PyTorch 2.7 with CUDA 12.8 support
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI dependencies
RUN pip install -r requirements.txt

# Create startup script
COPY startup.sh /workspace/ComfyUI/
RUN chmod +x /workspace/ComfyUI/startup.sh

# Expose port for web UI
EXPOSE 8188

# Set the entrypoint to our startup script
ENTRYPOINT ["/workspace/ComfyUI/startup.sh"]