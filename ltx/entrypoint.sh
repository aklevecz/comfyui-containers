#!/bin/bash

# Setup ComfyUI
setup_comfyui() {
    echo "Setting up ComfyUI..."

    # Check if ComfyUI is already cloned
    if [ ! -d "/workspace/ComfyUI/.git" ]; then
        echo "Cloning ComfyUI repository..."
        cd /workspace
        rm -rf ComfyUI/*
        git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
        cd ComfyUI
    else
        echo "ComfyUI repository already exists"
        cd /workspace/ComfyUI
    fi

    # Install PyTorch 2.7.0 with CUDA 12.8.1 support (matching RunPod template)
    if ! pip list | grep -q "torch"; then
        echo "Installing PyTorch 2.7.0 with CUDA 12.8 support..."
        pip install --upgrade pip
        pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    else
        echo "PyTorch already installed"
    fi

    # Install ComfyUI dependencies
    echo "Installing ComfyUI dependencies..."
    pip install -r requirements.txt
    pip install -U accelerate==0.32.0
}

# Function for downloading models
download_model() {
    local target_path=$1
    local url=$2
    local name=$3

    if [ ! -f "$target_path" ]; then
        echo "Downloading $name..."
        wget -O "$target_path" "$url" || echo "Warning: Failed to download $name"
    else
        echo "$name already exists, skipping download"
    fi
}

# Function for installing custom nodes
install_node() {
    local target_dir=$1
    local repo_url=$2
    local name=$3

    if [ ! -d "$target_dir" ]; then
        echo "Installing $name..."
        (cd /workspace/ComfyUI/custom_nodes &&
         git clone "$repo_url" "$(basename "$target_dir")" &&
         cd "$(basename "$target_dir")" &&
         if [ -f "requirements.txt" ]; then
             pip install -r requirements.txt
         fi) || echo "Warning: Failed to install $name"
    else
        echo "$name already exists, skipping installation"
    fi
}

# Setup ComfyUI first (needs to happen before other installations)
setup_comfyui

echo "Starting post-installation setup..."

# Download models in the background
(
    download_model "/workspace/ComfyUI/models/checkpoints/ltxv-13b-0.9.7-dev.safetensors" \
                  "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev.safetensors" \
                  "LTXV base model"

    download_model "/workspace/ComfyUI/models/upscale_models/ltxv-spatial-upscaler-0.9.7.safetensors" \
                  "https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7/resolve/main/latent_upsampler/diffusion_pytorch_model.safetensors" \
                  "LTX Upscaler"

    download_model "/workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors" \
                  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
                  "Flux text encoder model"
) &

# Install ComfyUI-Manager
(
    install_node "/workspace/ComfyUI/custom_nodes/comfyui-manager" \
                "https://github.com/ltdrdata/ComfyUI-Manager" \
                "ComfyUI-Manager"
) &

#install ComyUI-LTXVideo node
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo" \
                "https://github.com/Lightricks/ComfyUI-LTXVideo.git" \
                "ComfyUI-LTXVideo"
) &

#install ComfyUI-KJNodes node if does not exist https://github.com/kijai/ComfyUI-KJNodes.git
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes" \
                "https://github.com/kijai/ComfyUI-KJNodes.git" \
                "ComfyUI-KJNodes"
) &

#install COMFYUI-VideoHelperSuite nodes if do not exist https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" \
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" \
                "COMFYUI-VideoHelperSuite"
) &

# Start ComfyUI
echo "Starting ComfyUI..."
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188