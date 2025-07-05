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
    # NORMAL WAN MODEL - FLF2V
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-FLF2V-14B-720P_fp16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/checkpoints/Wan2_1-FLF2V-14B-720P_fp16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-FLF2V-14B-720P_fp16.safetensors?download=true" \
    #               "Wan Video base model"

    # WAN VACE MODEL 14B
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors
    download_model "/workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors" \
                  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors" \
                  "Wan Video Vace model"

    # WAN VACE BF16
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/diffusion_models/Wan2_1-VACE_module_14B_bf16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors?download=true" \
    #               "Wan Video Vace model"

    # WAN I2V FP16
    # https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp16.safetensors" \
    #               "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors?download=true" \
    #               "Wan Video I2V model" 

    # WAN I2V FP8
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e5m2.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/diffusion_models/Wan2_1-T2V-14B_fp8_e5m2.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e5m2.safetensors?download=true" \
    #               "Wan Video I2V model"

    # Wan Vace FusionX
    # https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE/resolve/main/Wan2.1_T2V_14B_FusionX_VACE-FP16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/diffusion_models/Wan2.1_T2V_14B_FusionX_VACE-FP16.safetensors" \
    #               "https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE/resolve/main/Wan2.1_T2V_14B_FusionX_VACE-FP16.safetensors?download=true" \
    #               "Wan Video Vace FusionX model"

    # Wan Vace 1.3B
    # https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors
    # download_model "/workspace/ComfyUI/models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" \
    #               "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" \
    #               "Wan Video Vace 1.3B model"

    # Wan Vace 1.3B FusionX
    # Wan Text Encoder bf16
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true" \
    #               "Wan Video text encoder model"

    # Wan Video text encoder
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true
    download_model "/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors" \
                  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors?download=true" \
                  "Wan Video text encoder model"

    # BF16
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true" \
    #               "Wan Video text encoder model"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/text_encoders/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors?download=true" \
    #               "Wan Video visual encoder model"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true
    # download_model "/workspace/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors" \
    #               "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true" \
    #               "Wan Video VAE model"

    # https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
    download_model "/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors" \
                  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
                  "Wan Video VAE model"
) &

# Install ComfyUI-Manager
(
    install_node "/workspace/ComfyUI/custom_nodes/comfyui-manager" \
                "https://github.com/ltdrdata/ComfyUI-Manager" \
                "ComfyUI-Manager"
) &

#install ComyUI-LTXVideo node
(
    #https://github.com/kijai/ComfyUI-WanVideoWrapper
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper" \
                "https://github.com/kijai/ComfyUI-WanVideoWrapper" \
                "ComfyUI-WanVideoWrapper"
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

# https://github.com/cubiq/ComfyUI_essentials
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_essentials" \
                "https://github.com/cubiq/ComfyUI_essentials.git" \
                "ComfyUI_essentials"
) &

# https://github.com/pythongosssss/ComfyUI-Custom-Scripts
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts" \
                "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git" \
                "ComfyUI-Custom-Scripts"
) &

# https://github.com/yolain/ComfyUI-Easy-Use
(
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use" \
                "https://github.com/yolain/ComfyUI-Easy-Use.git" \
                "ComfyUI-Easy-Use"
) &

# https://github.com/BadCafeCode/masquerade-nodes-comfyui
(
    install_node "/workspace/ComfyUI/custom_nodes/masquerade-nodes-comfyui" \
                "https://github.com/BadCafeCode/masquerade-nodes-comfyui.git" \
                "masquerade-nodes-comfyui"
) 

# ( 
#     #https://github.com/Fannovel16/comfyui_controlnet_aux
#     install_node "/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux" \
#                 "https://github.com/Fannovel16/comfyui_controlnet_aux" \
#                 "comfyui_controlnet_aux"
# ) & 

# (
#     #https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
#     install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation" \
#                 "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation" \
#                 "ComfyUI-Frame-Interpolation"
# )
# Start ComfyUI
echo "Starting ComfyUI..."
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188

# python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header "*"
#
