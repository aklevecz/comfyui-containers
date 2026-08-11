# syntax=docker/dockerfile:1
# Pinned frontend for the heredoc RUN below. Must stay on line 1 -- a parser
# directive after any comment is silently ignored.
#
# Wan on RTX PRO 6000 Blackwell.
#
# - cu128 base, not 12.4: Blackwell is sm_120. A torch built against CUDA 12.4
#   has no sm_120 kernels, so it either refuses the device or silently falls
#   back and runs slower than the local Windows box. This is the whole reason
#   this template is separate from audio-boy/runpod (which pins torch 2.4/cu124).
# - ubuntu24.04 for python3.12 final -- same reasoning as reactor-celery, where
#   22.04's python3.11.0rc1 segfaulted torch.jit on the kornia import.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    COMFY_ROOT=/opt/ComfyUI

# devel (not runtime) base: some WanVideoWrapper extras compile CUDA kernels.
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    dos2unix \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Baked at /opt so a /workspace volume mount can never shadow it -- the failure
# mode that crash-looped the earlier images in this repo.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# torch first, from the cu128 index, so the custom-node requirements below
# resolve against an already-correct torch instead of pulling a CPU wheel.
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install -r /opt/ComfyUI/requirements.txt

WORKDIR /opt/ComfyUI/custom_nodes

# The full node set the flower graphs need. Comfyroll and Easy-Use are in here
# deliberately: they are used by the extend graph's path/counter arithmetic and
# had gone missing from the local install, so the graph would not even load.
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
 && git clone --depth 1 https://github.com/BadCafeCode/masquerade-nodes-comfyui.git \
 && git clone --depth 1 https://github.com/crystian/ComfyUI-Crystools.git \
 && git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git \
 && git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
 && git clone --depth 1 https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git \
 && git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
 && git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git \
 && git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git

# Pin the torch trio so no node pack can replace it. This is not theoretical:
# ComfyUI_essentials pulls pixeloe, which asks for a newer torch off default
# PyPI, and pip is one backtrack away from honouring it. A constraint turns
# that into a loud failure of one pack's requirements instead of a silent swap
# of the cu128 wheel this whole template exists for.
RUN python3 -m pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /opt/torch-constraint.txt \
 && cat /opt/torch-constraint.txt

# Install each pack's requirements independently. One pack with a broken pin
# should not take down the image build -- ComfyUI reports a failed node pack
# clearly at startup, which is easier to debug than a dead build.
RUN for d in /opt/ComfyUI/custom_nodes/*/; do \
      if [ -f "$d/requirements.txt" ]; then \
        echo "=== installing requirements for $(basename $d)"; \
        PIP_CONSTRAINT=/opt/torch-constraint.txt \
          python3 -m pip install -r "$d/requirements.txt" || echo "!!! FAILED: $(basename $d)"; \
      fi; \
    done

# Fail the build if a custom-node requirements.txt pinned torch and pip swapped
# our cu128 wheel for a CPU or older-CUDA one. That clobber is silent: the image
# builds, the pod boots, ComfyUI starts, and then the card either refuses the
# device or falls back to something slower than the local Windows box. sm_120 in
# the arch list is the actual contract this template exists to satisfy, and both
# checks work on a CPU-only runner.
RUN python3 - <<'PY'
import sys, torch
# NOT torch.cuda.get_arch_list(): it early-returns [] when no CUDA device is
# visible, and the builder is a CPU-only runner, so it reports "no sm_120" for
# a perfectly good wheel. _cuda_getArchFlags() reads the compile-time arch
# string out of the extension and does not care whether a GPU is present.
v = torch.__version__
flags = getattr(torch._C, "_cuda_getArchFlags", lambda: None)()
print("torch %s\narch flags %s" % (v, flags))
if "+cu128" not in v:
    sys.exit("FATAL: torch is %s, not a cu128 build -- a custom node pinned over it" % v)
if not flags:
    sys.exit("FATAL: torch reports no CUDA arch flags -- this is a CPU-only build")
if "sm_120" not in flags.split():
    sys.exit("FATAL: torch has no sm_120 kernels (%s); Blackwell will not run this" % flags)
PY

# ImageTrimBatch: a one-off node written for this project, never published.
# The extend graph uses it to strip the 15 overlap frames off each render.
# Without it the graph will not load at all.
COPY TrimImageBatch.py /opt/ComfyUI/custom_nodes/TrimImageBatch.py

# Best-effort SageAttention. Worth a large speedup on Blackwell, but it builds
# CUDA kernels and is the most fragile thing here, so a failure must not break
# the image. The workflows ship with attention_mode "sdpa" and keep working
# either way; flip them to sageattn only after confirming the import succeeds.
RUN python3 -m pip install sageattention || echo "sageattention unavailable -- workflows stay on sdpa"

# ReActor face swap, for the celery_man app's dancer workflows. It registers the
# class name those graphs call, ReActorFaceSwap, so no workflow edits are needed.
#
# This template's README used to argue ReActor belonged on the cheap tier, since
# ONNX face swap loads no diffusion checkpoint and a 96 GB card buys it nothing.
# That reasoning was about whether to rent a big GPU *for* ReActor. It does not
# apply to riding along on a card already rented for Wan, and the app wants one
# endpoint rather than two hosts, so it goes here.
#
# onnxruntime-gpu is the load-bearing part: without cuDNN present the CUDA
# execution provider silently falls back to CPU and swaps crawl. This base is
# already cudnn-devel, which is why it works here at all.
# insightface is NOT in ReActor's requirements.txt any more -- it moved into
# the pack's install.py, which only runs under ComfyUI-Manager, and there is no
# Manager on this pod. Installing it explicitly is what makes the node import;
# without it ReActor is present on disk but never registers ReActorFaceSwap,
# which looks identical to "not installed" from the client's side.
# It builds from source, hence cython and the devel base's build-essential.
RUN PIP_CONSTRAINT=/opt/torch-constraint.txt python3 -m pip install cython \
 && PIP_CONSTRAINT=/opt/torch-constraint.txt python3 -m pip install insightface==0.7.3

RUN PIP_CONSTRAINT=/opt/torch-constraint.txt python3 -m pip install onnxruntime-gpu

# Two failure modes that only surface as a slow or missing face swap on the pod,
# so assert them here instead. ComfyUI-Easy-Use pulls CPU onnxruntime, and if it
# ends up shadowing onnxruntime-gpu the CUDA provider silently disappears and
# swaps fall back to CPU. get_available_providers() reports the compiled-in
# providers, so this works on a GPU-less builder.
RUN python3 - <<'PY'
import sys
import insightface, onnxruntime
print("insightface", insightface.__version__)
provs = onnxruntime.get_available_providers()
print("onnxruntime", onnxruntime.__version__, provs)
if "CUDAExecutionProvider" not in provs:
    sys.exit("FATAL: no CUDAExecutionProvider -- CPU onnxruntime shadowed the GPU build")
PY

# ~1.2 GB, baked rather than fetched at boot. entrypoint.sh copies models/ onto
# the volume before symlinking, with cp -rn, so these land next to the Wan
# payload on first boot without clobbering anything already there.
#  - inswapper_128.onnx        the swap model
#  - buffalo_l                 insightface detection/recognition pack
#  - detection_Resnet50_Final  retinaface_resnet50 face detector
RUN mkdir -p /opt/ComfyUI/models/insightface/models/buffalo_l /opt/ComfyUI/models/facedetection \
    && wget -q -O /opt/ComfyUI/models/insightface/inswapper_128.onnx \
       "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx" \
    && wget -q -O /tmp/buffalo_l.zip \
       "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
    && unzip -q -j /tmp/buffalo_l.zip -d /opt/ComfyUI/models/insightface/models/buffalo_l \
    && rm /tmp/buffalo_l.zip \
    && wget -q -O /opt/ComfyUI/models/facedetection/detection_Resnet50_Final.pth \
       "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"

# Face restorers (~1.2 GB). Without these the face_restore_model dropdown offers
# only "none", which is the state the celery_man output is stuck in.
#
# Why they matter more than the GPU: inswapper_128 emits a 128x128 face patch,
# full stop. On a 768x768 frame the face lands around 250-350px, so it is being
# upscaled ~2.5x from 128 no matter how much VRAM is available -- there is no
# public 256 or 512 inswapper. A restorer is the only way past that ceiling.
#
# GPEN-BFR-2048 is the one worth having on a 96 GB card: it restores at
# 2048x2048 where GFPGAN and CodeFormer work at 512. 1024 is kept as the
# middle setting, and GFPGAN/CodeFormer because they are gentler -- this
# workflow deliberately wants some of the VHS softness left intact.
RUN mkdir -p /opt/ComfyUI/models/facerestore_models \
    && for m in GFPGANv1.4.pth codeformer-v0.1.0.pth GPEN-BFR-1024.onnx GPEN-BFR-2048.onnx; do \
         wget -q -O "/opt/ComfyUI/models/facerestore_models/$m" \
           "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/$m" \
         || exit 1; \
       done \
    && ls -la /opt/ComfyUI/models/facerestore_models

COPY download_models.py /opt/download_models.py
COPY workflows/ /opt/workflows/
COPY entrypoint.sh /opt/entrypoint.sh
RUN dos2unix /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
