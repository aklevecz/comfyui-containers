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
 && git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git

# Install each pack's requirements independently. One pack with a broken pin
# should not take down the image build -- ComfyUI reports a failed node pack
# clearly at startup, which is easier to debug than a dead build.
RUN for d in /opt/ComfyUI/custom_nodes/*/; do \
      if [ -f "$d/requirements.txt" ]; then \
        echo "=== installing requirements for $(basename $d)"; \
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
v, archs = torch.__version__, torch.cuda.get_arch_list()
print("torch %s\narchs %s" % (v, archs))
if "+cu128" not in v:
    sys.exit("FATAL: torch is %s, not a cu128 build -- a custom node pinned over it" % v)
if "sm_120" not in archs:
    sys.exit("FATAL: torch has no sm_120 kernels; Blackwell will not run this")
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

COPY download_models.py /opt/download_models.py
COPY workflows/ /opt/workflows/
COPY entrypoint.sh /opt/entrypoint.sh
RUN dos2unix /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188

ENTRYPOINT ["/opt/entrypoint.sh"]
