# wan-blackwell

ComfyUI + Wan 2.1 on an RTX PRO 6000 Blackwell, carrying two workflows that
share the same model stack:

- **wan-flower** — the audio-reactive VACE chain (`wan_audio_reactive_fast_*`)
- **celery-man v2v** — the Wan video-to-video re-render (`celery_man_wan_v2v_api`)

They ride in one image because they load the *same* base: `Wan2_1-T2V-14B_fp8`,
`Wan2_1_VAE_bf16`, `umt5-xxl-enc-bf16`, and `lightx2v`. Only the LoRA stack on
top differs, so a single volume serves both. Measured payload: **34.3 GiB**.

The ReActor Celery Man is **not** here — see "Which Celery Man goes where" below.

## Why this is a separate template

`audio-boy/runpod` pins `torch 2.4 / cu12.4`. Blackwell is **sm_120**, which
that build has no kernels for. This image is cu128, matching `reactor-celery`.

## Build and push

CI does it. `.github/workflows/wan-blackwell.yml` builds on a GitHub runner and
pushes to **`ghcr.io/aklevecz/comfyui-wan-blackwell`** on any push to `main`
touching `wan-blackwell/**`. Trigger a rebuild by hand with:

```sh
gh workflow run wan-blackwell.yml
gh run watch
```

There is deliberately no build cache — the GHA cache is capped at 10 GB per repo
and the torch layer alone exceeds that, so it would evict itself every run. A
cold build is the only kind there is.

### Tags

**There is no `latest`.** Every build publishes two immutable tags:

| Tag | Example | Use |
|---|---|---|
| `YYYYMMDD-<short-sha>` | `20260810-a72b5ea1c3d4` | the one to paste into RunPod; sorts chronologically |
| `sha-<short-sha>` | `sha-a72b5ea1c3d4` | same image, for looking up a known commit |

A RunPod template pins whatever tag you give it, so a moving tag means the pod
you booted last week is not the pod you boot today and nothing tells you which
you got. The tradeoff is real and worth knowing: you now edit the template on
every deploy. The CI run summary prints the exact ref to copy, so it is one
click rather than a log dig.

If you want a "known good on hardware" pointer once something is actually
verified on a pod, promote it by hand rather than letting CI move it:

```sh
docker buildx imagetools create -t ghcr.io/aklevecz/comfyui-wan-blackwell:stable \
                                   ghcr.io/aklevecz/comfyui-wan-blackwell:20260810-a72b5ea1c3d4
```

**Not RunPod's GitHub integration.** That path looks like the obvious fit and
isn't: it is Serverless-only, it requires a runpod handler function (this image
serves an interactive ComfyUI UI on 8188, so there is nothing to hand off to),
and images it builds are locked to RunPod's own infrastructure — they cannot be
pulled for a Pod.

Building locally instead is a fallback, not the default: it's ~25 GB of image
plus a ~10 GB push over a home connection.

```sh
cd wan-blackwell && docker build -t ghcr.io/aklevecz/comfyui-wan-blackwell:local .
```

## RunPod setup

- Container image: **`ghcr.io/aklevecz/comfyui-wan-blackwell:<date>-<sha>`** —
  copy the exact ref off the CI run summary; see "Tags". There is no `latest`.
- GPU: **RTX PRO 6000 Blackwell (96 GB)**
- HTTP port: **8188**
- Volume mount path: **`/workspace`** — not optional here. Without it the
  34.3 GiB payload re-downloads every pod and dies with the container.
- Container disk: **50 GB**. Do not take the default; it is 5–20 GB depending
  on how you deploy, and the image does not fit in either. 40 GB is the floor.
- Volume: **100 GB**. Fixed cost is 34.3 GiB of models; the rest is render
  space, and PNG frame sequences are what actually consume it — a 75-second
  flower piece is 2,268 frames at ~446 KB, so **1.7 GB per render**. The mp4s
  are rounding error next to that. 75 GB works if you housekeep.

`entrypoint.sh` symlinks `models`, `output`, `input` and `user` to `/workspace`,
so renders land on the volume and share the pool with the models. Nothing but
the OS and scratch lives on container disk.

The package published **public**, verified by pulling its manifest anonymously
(HTTP 200, no token), so RunPod needs no registry credentials. If it ever flips
private, add a GitHub PAT with `read:packages` under RunPod Settings → Registry
Credentials. Nothing secret is in the image — models are fetched at boot, not
baked.

First boot downloads models in the background; ComfyUI is reachable immediately.
Watch progress in the pod log or `/workspace/model-download.log`.

## Networking

Only **8188** needs exposing, as an HTTP port — RunPod serves it at
`https://<pod-id>-8188.proxy.runpod.net`. ComfyUI already binds `0.0.0.0`,
which the proxy requires. TCP 22 is worth adding only if you would rather scp
the two manual LoRAs than use `runpodctl` from the web terminal.

**That URL is public.** Pod-ID obscurity is the only thing in front of it, and
ComfyUI has no authentication of its own — anyone who has the URL can queue
work on a 96 GB card you are paying for. CORS ships open on top of that (see
"Environment variables"), so treat the URL as a secret and stop the pod when
you are done rather than leaving it parked.

The proxy runs through Cloudflare with a **100-second cap on any single
connection**, after which you get a 524. ComfyUI's design mostly sidesteps this
— `/prompt` returns a job id immediately and the client polls `/queue` and
`/history` — so long renders are not at risk. What *can* hit the cap is a
single slow transfer: uploading a large mask video to `/upload/image`, or
pulling a finished mp4 back through `/view`. If either times out, that is the
cause, and it is not a pod failure.

## Environment variables

**None are required.** `COMFY_ROOT` is baked into the image, and every model
resolves from Hugging Face anonymously, so there is no `HF_TOKEN` to set. An
empty env list is a valid deploy.

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_CORS` | `1` | CORS on. `0`/`false`/`no`/`off` disables it |
| `CORS_ORIGIN` | `*` | Restrict CORS to one origin instead of any |
| `COMFY_ARGS` | *(empty)* | Appended verbatim to the `main.py` command line |
| `COMFY_ROOT` | `/opt/ComfyUI` | Install root. No reason to change it on RunPod |

**CORS is on by default**, so the React client works against a fresh pod with
no configuration — its `/prompt`, `/upload/image` and `/view` calls are all
cross-origin through the proxy and would otherwise fail.

Know what that means: the proxy URL is public and ComfyUI has no
authentication, so with the default `*` any page loaded in any browser can
queue work on a card you are paying for. Two ways to narrow it without giving
up the client:

```
CORS_ORIGIN=https://your-client.example    # only that origin
ENABLE_CORS=0                              # off entirely
```

`COMFY_ARGS` covers everything else without a rebuild, which is 12 uncached
minutes. The one worth knowing about is `--max-upload-size`, which ComfyUI
defaults to **100 MB** — fine for the mask videos in `wan-flower/examples/`
(the largest is 12.8 MB), but a long mask or a big source clip for the v2v
graph can exceed it:

```
COMFY_ARGS=--max-upload-size 500
```

## The two LoRAs you must upload yourself

This was five. Three of them turned out to be on Hugging Face after all, just
not under names anything would have guessed, so they now download at boot like
everything else. **~920 MB left to hand-carry.**

| File | Needed by | Source |
|---|---|---|
| `sh4rpn3ss_v2_e56.safetensors` | wan-flower extend | manual — 613 MB |
| `DetailEnhancerV1.safetensors` | celery-man v2v | manual — 307 MB |
| ~~`Wan2.1-Fun-14B-InP-MPS`~~ | celery-man v2v | auto — `alibaba-pai/Wan2.1-Fun-Reward-LoRAs` |
| ~~`Wan14B_RealismBoost`~~ | celery-man v2v | auto — `anthonyluu/Wan14B_RealismBoost` |
| ~~`detailz-wan`~~ | wan-flower extend | auto — `Muapi/detailz-wan-…` |

The three automatic ones were matched by hashing the local originals, so they
are the same bytes rather than plausible substitutes, and `download_models.py`
pins each sha256 and verifies it on the way down. That check is not ceremony:
two of the three come from unofficial mirrors, `Muapi/wan14b_detailer-enhancer_t2v`
ships RealismBoost's bytes under a DetailEnhancer name, and the FusionX repo
that once hosted several of these now 401s. The digest is what makes depending
on a mirror safe — if one silently swaps contents, boot fails loudly instead of
rendering something subtly wrong.

The remaining two are Civitai-origin and not on Hugging Face under any
searchable name. Both are on the local Windows box at
`ComfyUI_windows_portable_nvidia_cu118_or_cpu/…/ComfyUI/models/loras/`:

```sh
# on the local machine
runpodctl send DetailEnhancerV1.safetensors
# then on the pod, in /workspace/models/loras
runpodctl receive <code>
```

Volume-resident, so this is once ever, not once per pod. To retire the step
entirely, upload both to a Hugging Face repo of your own and move them into
`WANTED` with their recorded hashes — `download_models.py` already carries the
sha256 of each known-good original.

**Don't substitute lookalikes for these two.** For celery-man v2v especially:
the positive and negative prompts are *empty*, so the LoRA stack is the entire
conditioning — a different "detail enhancer" is a different render, not a close
one. wan-flower extend degrades more gracefully; if you just want it moving,
dropping `sh4rpn3ss` to strength 0 costs sharpness but still runs.

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
through the same proxy. Those calls are cross-origin, and CORS is enabled by
default, so this works against a fresh pod with nothing to configure. Narrow it
with `CORS_ORIGIN` once you know the client's origin.

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

- **Never run on a live pod.** The image now builds green in CI and is published
  at `ghcr.io/aklevecz/comfyui-wan-blackwell` — 10.64 GiB compressed, 28 layers,
  ~12 min cold build. The build asserts torch is `2.11.0+cu128` carrying
  `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, so Blackwell support is confirmed in
  the wheel rather than assumed. Everything past that point is still unverified:
  no pod has booted it, no model has downloaded, no graph has been queued.
- **Custom-node repo URLs are unpinned `--depth 1` clones of `main`.** If any
  URL is wrong the build fails loudly, which is the intent — but pin SHAs before
  you rely on this for anything scheduled.
- **SageAttention installs, but is unproven.** `sageattention 1.0.6` lands from
  PyPI as a pure-Python wheel — no CUDA compile, so the `|| echo` guard never
  fires. That only means the package is present; it does not mean its Triton
  kernels work on sm_120. The workflows stay on `sdpa`. Confirm it imports and
  runs on the pod before flipping `attention_mode` to `sageattn`.
- The extend graph's `PROJECT_PATH` / frames-directory convention is subtle and
  drifted once already — the client writes `{project}/frames/1` but the renders
  that produced `fast_flower` used `{project}/1`. Verify the first extend pass
  actually finds the previous segment before queueing thirty of them.
