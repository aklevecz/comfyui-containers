#!/bin/bash
# Same shape as reactor-celery's entrypoint, with one important difference:
# the model payload here is ~31 GB, far too big to bake. So models live on the
# network volume and download in the BACKGROUND -- ComfyUI comes up immediately
# and you can watch the download progress in the pod logs rather than staring
# at a dead port for twenty minutes.

COMFY=${COMFY_ROOT:-/opt/ComfyUI}

link_to_volume() {
    local sub=$1
    mkdir -p "/workspace/$sub"
    if [ -d "$COMFY/$sub" ] && [ ! -L "$COMFY/$sub" ]; then
        cp -rn "$COMFY/$sub/." "/workspace/$sub/" 2>/dev/null || true
        rm -rf "$COMFY/$sub"
    fi
    ln -sfn "/workspace/$sub" "$COMFY/$sub"
}

if [ -d /workspace ]; then
    echo "network volume detected -- linking models/output/input/user"
    for d in models output input user; do
        link_to_volume "$d"
    done
else
    echo "NO network volume mounted."
    echo "The ~31 GB model payload will download into the container and be lost"
    echo "when the pod stops. Attach a volume at /workspace unless this is a test."
fi

# Ship the workflows into the user's workflow browser on every boot, so an
# image update actually reaches the UI. Never clobber an edited copy.
mkdir -p "$COMFY/user/default/workflows"
for f in /opt/workflows/*.json; do
    base=$(basename "$f")
    if [ ! -f "$COMFY/user/default/workflows/$base" ]; then
        cp "$f" "$COMFY/user/default/workflows/$base"
    fi
done

# Background so the UI is reachable now. Log lands in the pod log and on volume.
# Fall back to /tmp when there is no volume: tee against a missing /workspace
# fails, and the downloader then dies on the broken pipe instead of running.
LOG=/workspace/model-download.log
[ -d /workspace ] || LOG=/tmp/model-download.log
echo "starting model download in background (see $LOG)"
python3 /opt/download_models.py 2>&1 | tee "$LOG" &

cd "$COMFY"
# --listen for the RunPod HTTP proxy.
# NOTE: do NOT add --disable-smart-memory / dynamic-vram flags here. On a 96 GB
# card the whole point is to let ComfyUI keep the model resident between passes;
# the extend chain queues many short jobs back to back and reloading the 14B
# model each time would erase the speedup this template exists for.

ARGS=(--listen 0.0.0.0 --port 8188)

# CORS is ON by default so the React client works against a fresh pod with no
# configuration. The cost is real and worth stating once: RunPod proxy URLs are
# public and ComfyUI has no authentication, so with '*' any page loaded in any
# browser can queue work on this GPU. Two ways to narrow that without giving up
# the client -- set CORS_ORIGIN to the exact origin serving it, or ENABLE_CORS=0
# to turn it off entirely.
#
# CORS_ORIGIN must stay quoted. Its default is a literal '*', and unquoted it
# would glob against $COMFY and hand ComfyUI a filename as the allowed origin.
case "${ENABLE_CORS:-1}" in
    0|false|no|off)
        echo "CORS disabled (ENABLE_CORS=${ENABLE_CORS})"
        ;;
    *)
        ARGS+=(--enable-cors-header "${CORS_ORIGIN:-*}")
        echo "CORS enabled for origin: ${CORS_ORIGIN:-*}"
        ;;
esac

# COMFY_ARGS is a passthrough so the pod can be reconfigured without a rebuild,
# which costs 12 uncached minutes. Unquoted on purpose -- word splitting is how
# multiple flags get through -- so globbing is disabled first to stop a stray
# '*' in it from expanding against the working directory.
set -f
echo "Starting ComfyUI${COMFY_ARGS:+ with extra args: $COMFY_ARGS}"
exec python3 main.py "${ARGS[@]}" $COMFY_ARGS
