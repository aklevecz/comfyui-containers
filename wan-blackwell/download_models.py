#!/usr/bin/env python3
"""
Resolve and download the Wan model payload into ComfyUI's models tree.

Why this exists rather than a list of wget URLs: Kijai reorganises
WanVideo_comfy periodically (files move in and out of subfolders), and a stale
hardcoded URL fails at boot with a 404 that looks like a network problem. So
each file is resolved by BASENAME against the repo tree at runtime, and only
then downloaded.

Downloads are atomic (.part + rename). A partial file can never be left behind
looking complete -- ComfyUI would load it and fail deep inside safetensors with
an error that says nothing useful.

Already-present files are skipped, so this is safe to re-run and safe to run
against a warm network volume.
"""
import json
import os
import sys
import urllib.request
import urllib.error

COMFY = os.environ.get("COMFY_ROOT", "/opt/ComfyUI")
MODELS = os.path.join(COMFY, "models")

# Repos searched, in order, for each wanted file.
REPOS = [
    "Kijai/WanVideo_comfy",
]

# (destination subdir, filename)
WANTED = [
    ("diffusion_models", "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"),
    ("diffusion_models", "Wan2_1-VACE_module_14B_bf16.safetensors"),
    ("vae",              "Wan2_1_VAE_bf16.safetensors"),
    ("text_encoders",    "umt5-xxl-enc-bf16.safetensors"),
    ("loras",            "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"),
    ("loras",            "Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors"),
    ("loras",            "Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors"),
    ("loras",            "Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors"),
]

# Community LoRAs that are NOT in the repos above. They have to be uploaded to
# the volume by hand -- see the README. Listed here so the boot log names them
# explicitly instead of letting ComfyUI fail later with a bare file-not-found.
MANUAL = [
    ("loras", "detailz-wan.safetensors",            "wan-flower extend"),
    ("loras", "sh4rpn3ss_v2_e56.safetensors",       "wan-flower extend"),
    ("loras", "Wan2.1-Fun-14B-InP-MPS.safetensors", "celery-man v2v"),
    ("loras", "Wan14B_RealismBoost.safetensors",    "celery-man v2v"),
    ("loras", "DetailEnhancerV1.safetensors",       "celery-man v2v"),
]

_tree_cache = {}


def tree(repo):
    """Full recursive file listing for a HF repo, as {basename: path}."""
    if repo in _tree_cache:
        return _tree_cache[repo]
    url = "https://huggingface.co/api/models/%s/tree/main?recursive=true" % repo
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
    except Exception as e:
        print("  ! could not list %s: %s" % (repo, e), flush=True)
        data = []
    out = {}
    for entry in data:
        if entry.get("type") == "file":
            p = entry["path"]
            out.setdefault(os.path.basename(p), p)
    _tree_cache[repo] = out
    return out


def resolve(filename):
    for repo in REPOS:
        path = tree(repo).get(filename)
        if path:
            return "https://huggingface.co/%s/resolve/main/%s" % (repo, path)
    return None


def download(url, dest):
    part = dest + ".part"
    print("  downloading %s" % os.path.basename(dest), flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wan-blackwell/1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            mark = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done - mark > (total // 10 or 1):
                    mark = done
                    print("    %s %d%%" % (os.path.basename(dest), 100 * done // total), flush=True)
        os.rename(part, dest)
        print("  done %s" % os.path.basename(dest), flush=True)
        return True
    except Exception as e:
        print("  ! failed %s: %s" % (os.path.basename(dest), e), flush=True)
        if os.path.exists(part):
            os.remove(part)
        return False


def main():
    missing = []
    for subdir, name in WANTED:
        d = os.path.join(MODELS, subdir)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        if os.path.exists(dest):
            print("  have %s" % name, flush=True)
            continue
        url = resolve(name)
        if not url:
            print("  ! could not resolve %s in any of %s" % (name, ", ".join(REPOS)), flush=True)
            missing.append(name)
            continue
        if not download(url, dest):
            missing.append(name)

    absent_manual = []
    for subdir, name, used_by in MANUAL:
        if not os.path.exists(os.path.join(MODELS, subdir, name)):
            absent_manual.append((name, used_by))

    print("", flush=True)
    if absent_manual:
        print("=" * 68, flush=True)
        print("MANUAL UPLOAD REQUIRED -- these LoRAs are not on Hugging Face and", flush=True)
        print("must be copied to the volume yourself (see README):", flush=True)
        for name, used_by in absent_manual:
            print("   models/loras/%-42s  (%s)" % (name, used_by), flush=True)
        print("", flush=True)
        print("Graphs that need them will fail to queue until they are present.", flush=True)
        print("The wan-flower INIT graph does not need any of them and works now.", flush=True)
        print("=" * 68, flush=True)
    if missing:
        print("FAILED to fetch: %s" % ", ".join(missing), flush=True)
        return 1
    print("model payload ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
