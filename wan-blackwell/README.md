# wan-blackwell

ComfyUI + Wan 2.1 on an RTX PRO 6000 Blackwell, carrying two workflows that
share the same model stack:

- **wan-flower** — the audio-reactive VACE chain (`wan_audio_reactive_fast_*`)
- **celery-man v2v** — the Wan video-to-video re-render (`celery_man_wan_v2v_api`)

They ride in one image because they load the *same* base: `Wan2_1-T2V-14B_fp8`,
`Wan2_1_VAE_bf16`, `umt5-xxl-enc-bf16`, and `lightx2v`. Only the LoRA stack on
top differs, so a single ~31 GB volume serves both.

The ReActor Celery Man is **not** here — see "Which Celery Man goes where" below.

## Why this is a separate template

`audio-boy/runpod` pins `torch 2.4 / cu12.4`. Blackwell is **sm_120**, which
that build has no kernels for. This image is cu128, matching `reactor-celery`.

## Build and push

CI does it. `.github/workflows/wan-blackwell.yml` builds on a GitHub runner and
pushes to **`ghcr.io/aklevecz/comfyui-wan-blackwell:latest`** (plus a
commit-SHA tag) on any push to `main` touching `wan-blackwell/**`. Trigger a
rebuild by hand with:

```sh
gh workflow run wan-blackwell.yml
gh run watch
```

There is deliberately no build cache — the GHA cache is capped at 10 GB per repo
and the torch layer alone exceeds that, so it would evict itself every run. A
cold build is the only kind there is.

**Not RunPod's GitHub integration.** That path looks like the obvious fit and
isn't: it is Serverless-only, it requires a runpod handler function (this image
serves an interactive ComfyUI UI on 8188, so there is nothing to hand off to),
and images it builds are locked to RunPod's own infrastructure — they cannot be
pulled for a Pod.

Building locally instead is a fallback, not the default: it's ~25 GB of image
plus a ~10 GB push over a home connection.

```sh
cd wan-blackwell && docker build -t ghcr.io/aklevecz/comfyui-wan-blackwell:latest .
```

## RunPod setup

- Container image: **`ghcr.io/aklevecz/comfyui-wan-blackwell:latest`**
- GPU: **RTX PRO 6000 Blackwell (96 GB)**
- HTTP port: **8188**
- Volume mount path: **`/workspace`** — not optional here. Without it the 31 GB
  payload re-downloads every pod and dies with the container.

GHCR packages are **private on first publish**, and RunPod cannot pull one
without credentials. Either make the package public (repo → Packages → the
package → Package settings → Change visibility), or add a GitHub PAT with
`read:packages` under RunPod Settings → Registry Credentials and attach it to
the template. Public is simpler and there is nothing secret in the image — the
models are all fetched at boot, not baked.

First boot downloads models in the background; ComfyUI is reachable immediately.
Watch progress in the pod log or `/workspace/model-download.log`.

## The five LoRAs you must upload yourself

Everything else resolves off Hugging Face automatically. These are community
LoRAs that aren't in `Kijai/WanVideo_comfy`, so `download_models.py` names them
at boot and moves on. **2.8 GB total.**

| File | Needed by |
|---|---|
| `detailz-wan.safetensors` | wan-flower extend |
| `sh4rpn3ss_v2_e56.safetensors` | wan-flower extend |
| `Wan2.1-Fun-14B-InP-MPS.safetensors` | celery-man v2v |
| `Wan14B_RealismBoost.safetensors` | celery-man v2v |
| `DetailEnhancerV1.safetensors` | celery-man v2v |

All five are on the local Windows box at
`ComfyUI_windows_portable_nvidia_cu118_or_cpu/…/ComfyUI/models/loras/`.
Easiest transfer, per file:

```sh
# on the local machine
runpodctl send DetailEnhancerV1.safetensors
# then on the pod, in /workspace/models/loras
runpodctl receive <code>
```

**The wan-flower INIT graph needs none of them** and works the moment the base
download finishes — that's the fastest path to seeing something render.

## What was changed from the local workflows

`port_workflows.py` did this, and prints every edit. It only ever rewrites
literal input values — **graph topology is never touched**, so the `_meta.title`
contract the React client patches against still holds.

| Change | Why |
|---|---|
| `StringConcatenate` delimiters `\` → `/` | the extend graph assembles the next segment's path by string concatenation; the separator was a literal Windows backslash |
| `easy string` node 515 → `/opt/ComfyUI/output/` | was the hardcoded Windows portable output dir |
| `VHS_LoadVideoPath` absolute paths → `/opt/ComfyUI/input/<basename>` | also stripped the stray `"` characters wrapping node 546's path, which ComfyUI passed through verbatim and then failed to open |
| LoRA fields `wan\Name.safetensors` → `Name.safetensors` | the v2v graph used a Windows subfolder; the downloader flattens into `models/loras/` |
| `blocks_to_swap` 30 → 0 (v2v: 10 → 0) | **the main speedup.** Block swapping existed only to fit 14B on a small card; on 96 GB every swap is a pointless round trip to system RAM on every step |
| 512 → 720 (nodes 167/168, 491/492, EmptyImage, `easy int`) | matches the resolution the `fast_flower.mp4` examples were actually rendered at |
| stale `ShowText` caches cleared | cosmetic, but they display dead Windows paths in the UI |

To retarget (different output root, different resolution), edit the constants at
the top of `port_workflows.py` and re-run — it's idempotent.

## Running wan-flower

1. Upload a mask video via the ComfyUI UI, or drop one in `/workspace/input/`.
2. Queue `wan_audio_reactive_fast_init` — 81 frames.
3. Queue `wan_audio_reactive_fast_extend` repeatedly — each pass adds 66 new
   frames and re-uses the previous render's last 15 as fixed context.

To drive it from the React client instead, point `comfy.js` at the pod:

```js
runComfyWorkflow(wf, { host: "https://<pod-id>-8188.proxy.runpod.net" })
```

The client also calls `POST /upload/image` and reads `/view`, both of which go
through the same proxy. Browser calls are cross-origin, so ComfyUI needs
`--enable-cors-header` (add it to the `exec` line in `entrypoint.sh`) or a small
proxy in front. Left off deliberately — it opens the instance to any page you
have loaded.

## Which Celery Man goes where

**ReActor stays on the cheap tier.** It's `inswapper_128` + retinaface — small
ONNX models, per-frame bound, no diffusion checkpoint at all. A 96 GB card buys
essentially nothing; the existing `reactor-celery` template already runs fully
self-contained on the cheapest GPU with no volume. Don't move it.

**The v2v version belongs here**, and it's already in this image. It's the same
14B Wan stack as the flower, so it inherits the same `blocks_to_swap` win. It
was previously bound by the local card the same way.

Its `VHS_LoadVideo` still points at `celeryman_dance_1.mp4` — copy the four
clips from `reactor-celery/input/` into `/workspace/input/`.

## Known gaps

- **Untested on real hardware.** Written and validated statically: JSON parses,
  no Windows path fragments remain, Python compiles, `bash -n` clean. Nothing
  here has been through a `docker build` or a live pod.
- **Custom-node repo URLs are unpinned `--depth 1` clones of `main`.** If any
  URL is wrong the build fails loudly, which is the intent — but pin SHAs before
  you rely on this for anything scheduled.
- **SageAttention is best-effort.** It's installed with `|| echo`, and the
  workflows stay on `sdpa` regardless. Once you confirm it imports on the pod,
  flipping `attention_mode` to `sageattn` is likely another solid speedup.
- The extend graph's `PROJECT_PATH` / frames-directory convention is subtle and
  drifted once already — the client writes `{project}/frames/1` but the renders
  that produced `fast_flower` used `{project}/1`. Verify the first extend pass
  actually finds the previous segment before queueing thirty of them.
