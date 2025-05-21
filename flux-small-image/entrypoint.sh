#!/bin/bash

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

    # Create necessary directories
    mkdir -p /workspace/ComfyUI/models/checkpoints
    mkdir -p /workspace/ComfyUI/models/vae
    mkdir -p /workspace/ComfyUI/models/unet
    mkdir -p /workspace/ComfyUI/models/style_models
    mkdir -p /workspace/ComfyUI/models/diffusion_models
    mkdir -p /workspace/ComfyUI/models/text_encoders
    mkdir -p /workspace/ComfyUI/custom_nodes
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
    download_model "/workspace/ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors" \
                  "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
                  "Flux fp8 merged model"
    
    download_model "/workspace/ComfyUI/models/checkpoints/realisticVisionV51_v51VAE.safetensors" \
                  "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors" \
                  "Realistic Vision v5"
    
    download_model "/workspace/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors" \
                  "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
                  "VAE"

    download_model "/workspace/ComfyUI/models/vae/ae.safetensors" \
                  "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors" \
                  "Flux VAE"
    
    download_model "/workspace/ComfyUI/models/unet/iclight_sd15_fc.safetensors" \
                  "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors" \
                  "IC Light"
    
    download_model "/workspace/ComfyUI/models/style_models/flux1-redux-dev.safetensors" \
                  "https://huggingface.co/Runware/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors" \
                  "Flux redux model"
    
    download_model "/workspace/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors" \
                  "https://huggingface.co/mp3pintyo/FLUX.1/resolve/main/flux1-fill-dev.safetensors" \
                  "Flux fill model"
    
    download_model "/workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors" \
                  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
                  "Flux text encoder model"
    
    download_model "/workspace/ComfyUI/models/text_encoders/clip_l.safetensors" \
                  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
                  "Flux CLIP text encoder model"

    download_model "/workspace/ComfyUI/models/text_encoders/sigclip_vision_patch14_384.safetensors" \
                  "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors" \
                  "SigClip Vision model"
) &

# Start installing custom nodes in the background
(
    install_node "/workspace/ComfyUI/custom_nodes/comfyui-manager" \
                "https://github.com/ltdrdata/ComfyUI-Manager" \
                "ComfyUI-Manager"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle" \
                "https://github.com/chflame163/ComfyUI_LayerStyle.git" \
                "ComfyUI_LayerStyle"
    
    install_node "/workspace/ComfyUI/custom_nodes/comfyui-mixlab-nodes" \
                "https://github.com/MixLabPro/comfyui-mixlab-nodes.git" \
                "comfyui-mixlab-nodes"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes" \
                "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git" \
                "ComfyUI_Comfyroll_CustomNodes"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-BrushNet" \
                "https://github.com/nullquant/ComfyUI-BrushNet.git" \
                "ComfyUI-BrushNet"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-IC-Light" \
                "https://github.com/kijai/ComfyUI-IC-Light.git" \
                "ComfyUI-IC-Light"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_essentials" \
                "https://github.com/cubiq/ComfyUI_essentials.git" \
                "ComfyUI_essentials"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle_Advance" \
                "https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git" \
                "ComfyUI_LayerStyle_Advance"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes" \
                "https://github.com/kijai/ComfyUI-KJNodes.git" \
                "ComfyUI-KJNodes"
    
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" \
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" \
                "ComfyUI-VideoHelperSuite"
) &

# Start ComfyUI
echo "Starting ComfyUI..."
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188