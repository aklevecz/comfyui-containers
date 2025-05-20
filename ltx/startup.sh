#!/bin/bash

# Download base model if it doesn't exist
if [ ! -f "/workspace/ComfyUI/models/checkpoints/ltxv-13b-0.9.7-dev.safetensors" ]; then
    echo "Downloading LTXV base model..."
    wget -O /workspace/ComfyUI/models/checkpoints/ltxv-13b-0.9.7-dev.safetensors https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev.safetensors
fi

# Download VAE if it doesn't exist
if [ ! -f "/workspace/ComfyUI/models/upscale_models/ltxv-spatial-upscaler-0.9.7.safetensors" ]; then
    echo "Downloading LTX Upscaler"
    wget -O /workspace/ComfyUI/models/upscale_models/ltxv-spatial-upscaler-0.9.7.safetensors https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7/resolve/main/latent_upsampler/diffusion_pytorch_model.safetensors
fi

# Download ControlNet model if it doesn't exist
if [ ! -f "/workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors" ]; then
    echo "Downloading Flux text encoder model..."
    wget -O /workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
fi

# Install ComfyUI-Manager if it doesn't exist
if [ ! -d "/workspace/ComfyUI/custom_nodes/comfyui-manager" ]; then
    echo "Installing ComfyUI-Manager..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
    cd /workspace/ComfyUI
fi

#install ComyUI-LTXVideo node
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo" ]; then
    echo "Installing ComfyUI-LTXVideo..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
    cd ComfyUI-LTXVideo && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#install ComfyUI-KJNodes node if does not exist https://github.com/kijai/ComfyUI-KJNodes.git
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes" ]; then
    echo "Installing ComfyUI-KJNodes..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
    cd ComfyUI-KJNodes && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#install COMFYUI-VideoHelperSuite nodes if do not exist https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" ]; then
    echo "Installing COMFYUI-VideoHelperSuite..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

# Start ComfyUI
echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188