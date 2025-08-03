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
    # WAN VACE f8
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors
    download_model "/workspace/ComfyUI/models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" \
                  "Wan Video Vace model"

    # WAN VACE BF16
    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors?download=true
    download_model "/workspace/ComfyUI/models/diffusion_models/Wan2_1-VACE_module_14B_bf16.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors?download=true" \
                  "Wan Video Vace model"


    # Wan Text Encoder bf16
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true
    download_model "/workspace/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors?download=true" \
                  "Wan Video text encoder model"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true
    download_model "/workspace/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true" \
                  "Wan Video VAE model"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors?download=true
    download_model "/workspace/ComfyUI/models/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors?download=true" \
                  "Wan Video Lightx2v Lora"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors?download=true
    download_model "/workspace/ComfyUI/models/loras/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors?download=true" \
                  "Wan Video CausVid Lora"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors?download=true
    download_model "/workspace/ComfyUI/models/loras/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors?download=true" \
                  "Wan Video AccVid Lora"

    # https://huggingface.co/aklevecz/wan/resolve/main/detailz-wan.safetensors?download=true
    # Detailz-Wan Lora
    download_model "/workspace/ComfyUI/models/loras/detailz-wan.safetensors" \
                  "https://huggingface.co/aklevecz/wan/resolve/main/detailz-wan.safetensors?download=true" \
                  "Detailz-Wan Lora"

    # https://huggingface.co/aklevecz/wan/resolve/main/sh4rpn3ss_v2_e56.safetensors?download=true
    # sh4rpn3ss_v2_e56 Lora
    download_model "/workspace/ComfyUI/models/loras/sh4rpn3ss_v2_e56.safetensors" \
                  "https://huggingface.co/aklevecz/wan/resolve/main/sh4rpn3ss_v2_e56.safetensors?download=true" \
                  "sh4rpn3ss_v2_e56 Lora"

    # https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors
    download_model "/workspace/ComfyUI/models/loras/Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors" \
                  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors?download=true" \
                  "Wan Video MoviiGen Lora"

    # https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors
    download_model "/workspace/ComfyUI/models/loras/Wan2.1-Fun-14B-InP-MPS.safetensors" \
                  "https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors" \
                  "Wan Video Fun Lora"
    
    # https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/Wan14B_RealismBoost.safetensors?download=true
    download_model "/workspace/ComfyUI/models/loras/Wan14B_RealismBoost.safetensors" \
                  "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/Wan14B_RealismBoost.safetensors?download=true" \
                  "Wan Video RealismBoost Lora"

    # https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/DetailEnhancerV1.safetensors?download=true
    download_model "/workspace/ComfyUI/models/loras/DetailEnhancerV1.safetensors" \
                  "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/DetailEnhancerV1.safetensors?download=true" \
                  "Wan Video DetailEnhancerV1 Lora"

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

( 
    #https://github.com/Fannovel16/comfyui_controlnet_aux
    install_node "/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux" \
                "https://github.com/Fannovel16/comfyui_controlnet_aux" \
                "comfyui_controlnet_aux"
) & 

(
    #https://github.com/crystian/ComfyUI-Crystools.git
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI-Crystools" \
                "https://github.com/crystian/ComfyUI-Crystools.git" \
                "ComfyUI-Crystools"
) &

(
    #https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes" \
                "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git" \
                "ComfyUI_Comfyroll_CustomNodes"
) &

(
    #https://github.com/chrisgoringe/cg-use-everywhere.git
    install_node "/workspace/ComfyUI/custom_nodes/cg-use-everywhere" \
                "https://github.com/chrisgoringe/cg-use-everywhere.git" \
                "cg-use-everywhere"
) &

(
    #https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git
    install_node "/workspace/ComfyUI/custom_nodes/ComfyUI_tinyterraNodes" \
                "https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git" \
                "ComfyUI_tinyterraNodes"
)


# Start ComfyUI
echo "Starting ComfyUI..."
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188

# python3 main.py --listen 0.0.0.0 --port 8188 --enable-cors-header "*"
# 

# 50GB
# june3 2025
