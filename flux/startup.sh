#!/bin/bash

# 
if [ ! -f "/workspace/ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors" ]; then
    echo "Downloading Flux fp8 merged model..."
    wget -O /workspace/ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
fi

#https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors
if [ ! -f "/workspace/ComfyUI/models/checkpoints/realisticVisionV51_v51VAE.safetensors" ]; then
    echo "Downloading Realistic Vision v5..."
    wget -O /workspace/ComfyUI/models/checkpoints/realisticVisionV51_v51VAE.safetensors https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors
fi

#https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?
if [ ! -f "/workspace/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors" ]; then
    echo "Downloading VAE..."
    wget -O /workspace/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
fi

#https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors
if [ ! -f "/workspace/ComfyUI/models/unet/iclight_sd15_fc.safetensors" ]; then
    echo "Downloading IC Light..."
    wget -O /workspace/ComfyUI/models/unet/iclight_sd15_fc.safetensors https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors
fi

# GET BETTER LINK FOR FLUX REDUX OR MAKE OWN
if [ ! -f "/workspace/ComfyUI/models/style_models/flux1-redux-dev" ]; then
    echo "Downloading Flux redux model...."
    wget -O /workspace/ComfyUI/models/style_models/flux1-redux-dev https://huggingface.co/Runware/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors
fi

# https://huggingface.co/mp3pintyo/FLUX.1/resolve/main/flux1-fill-dev.safetensors
if [ ! -f "/workspace/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors" ]; then
    echo "Downloading Flux fill model...."
    wget -O /workspace/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors https://huggingface.co/mp3pintyo/FLUX.1/resolve/main/flux1-fill-dev.safetensors
fi


# Download ControlNet model if it doesn't exist
if [ ! -f "/workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors" ]; then
    echo "Downloading Flux text encoder model..."
    wget -O /workspace/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
fi

#https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
if [ ! -f "/workspace/ComfyUI/models/text_encoders/clip_l.safetensors" ]; then
    echo "Downloading Flux text encoder model..."
    wget -O /workspace/ComfyUI/models/text_encoders/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
fi

# Install ComfyUI-Manager if it doesn't exist
if [ ! -d "/workspace/ComfyUI/custom_nodes/comfyui-manager" ]; then
    echo "Installing ComfyUI-Manager..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
    cd /workspace/ComfyUI
fi

# https://github.com/chflame163/ComfyUI_LayerStyle
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle" ]; then
    echo "Installing ComfyUI_LayerStyle..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
    cd ComfyUI_LayerStyle && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/MixLabPro/comfyui-mixlab-nodes
if [ ! -d "/workspace/ComfyUI/custom_nodes/comfyui-mixlab-nodes" ]; then
    echo "Installing comfyui-mixlab-nodes..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/MixLabPro/comfyui-mixlab-nodes.git
    cd comfyui-mixlab-nodes && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes" ]; then
    echo "Installing ComfyUI_Comfyroll_CustomNodes..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
    cd ComfyUI_Comfyroll_CustomNodes && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/nullquant/ComfyUI-BrushNet
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI-BrushNet" ]; then
    echo "Installing ComfyUI-BrushNet..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/nullquant/ComfyUI-BrushNet.git
    cd ComfyUI-BrushNet && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/kijai/ComfyUI-IC-Light
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI-IC-Light" ]; then
    echo "Installing ComfyUI-IC-Light..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/kijai/ComfyUI-IC-Light.git
    cd ComfyUI-IC-Light && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/cubiq/ComfyUI_essentials
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI_essentials" ]; then
    echo "Installing ComfyUI_essentials..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/cubiq/ComfyUI_essentials.git
    cd ComfyUI_essentials && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

#https://github.com/chflame163/ComfyUI_LayerStyle_Advance
if [ ! -d "/workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle_Advance" ]; then
    echo "Installing ComfyUI_LayerStyle_Advance..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git
    cd ComfyUI_LayerStyle_Advance && pip install -r requirements.txt
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
    echo "Installing ComfyUI-VideoHelperSuite..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt
    cd /workspace/ComfyUI
fi

# Start ComfyUI
echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188