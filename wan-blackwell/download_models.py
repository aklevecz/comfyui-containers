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
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error

COMFY = os.environ.get("COMFY_ROOT", "/opt/ComfyUI")
MODELS = os.path.join(COMFY, "models")

# Civitai requires auth for downloads -- the URL 401s without it. Optional: set
# it and the LoRA below arrives automatically, leave it unset and boot prints
# the manual instructions instead. Passed as a query parameter rather than an
# Authorization header on purpose: the download redirects to a CDN, and an
# unexpected auth header on the redirected request is a known way to get a 400.
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "").strip()

# Repos searched, in order, for any file that does not name its own.
REPOS = [
    "Kijai/WanVideo_comfy",
]

# subdir  -- destination under models/
# name    -- what ComfyUI must see on disk; the workflows reference this exactly
# repo    -- pin to one repo instead of searching REPOS
# remote  -- path inside that repo, when the mirror renamed the file
# sha256  -- verified after download; a mismatch deletes the file and fails
#
# The three pinned LoRAs below were community uploads that used to require a
# manual copy onto the volume. Each was matched to a public mirror by hashing
# the local original, so these are the same bytes, not lookalikes. The hashes
# are here because mirror names lie: Muapi/wan14b_detailer-enhancer_t2v ships
# RealismBoost's bytes under a DetailEnhancer name, and vrgamedevgirl84's
# FusionX repo -- the obvious home for several of these -- now 401s. Pinning
# the digest is what makes an unofficial mirror safe to depend on.
WANTED = [
    {"subdir": "diffusion_models", "name": "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"},
    {"subdir": "diffusion_models", "name": "Wan2_1-VACE_module_14B_bf16.safetensors"},
    {"subdir": "vae",              "name": "Wan2_1_VAE_bf16.safetensors"},
    {"subdir": "text_encoders",    "name": "umt5-xxl-enc-bf16.safetensors"},
    {"subdir": "loras",            "name": "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"},
    {"subdir": "loras",            "name": "Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors"},
    {"subdir": "loras",            "name": "Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors"},
    {"subdir": "loras",            "name": "Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors"},

    # celery-man v2v -- official Alibaba release.
    {"subdir": "loras", "name": "Wan2.1-Fun-14B-InP-MPS.safetensors",
     "repo": "alibaba-pai/Wan2.1-Fun-Reward-LoRAs",
     "sha256": "d5a8582d7a1a671e0485724fad9fe70caf7b061a01d2ec352e4998b86e5764c1"},

    # celery-man v2v -- community mirror, digest-pinned.
    {"subdir": "loras", "name": "Wan14B_RealismBoost.safetensors",
     "repo": "anthonyluu/Wan14B_RealismBoost",
     "sha256": "1cd2217d7df8f2e4de76f8a890174cced33d551cffb58d42f6d6f7f7a6d1c654"},

    # wan-flower extend -- mirror renamed the file, so remote differs from name.
    {"subdir": "loras", "name": "detailz-wan.safetensors",
     "repo": "Muapi/detailz-wan-detail-enhancer-for-wan-videos",
     "remote": "detailz-wan-detail-enhancer-for-wan-videos.safetensors",
     "sha256": "6e87dccd1ce65ceba4ab9590bf59bb5fe1a73edc8eba622a413862eaa8818f87"},

    # wan-flower extend -- not on Hugging Face in any form. Found on Civitai by
    # hashing the local original and querying their by-hash endpoint, which is
    # how the version id below was pinned rather than guessed. Needs
    # CIVITAI_TOKEN; without one this falls through to the manual instructions.
    {"subdir": "loras", "name": "sh4rpn3ss_v2_e56.safetensors",
     "civitai": 1928593,
     "sha256": "508163c59ac81e6f10250637b13b2624b714ebaf163aa8d48c9c599b3d0b02d4"},
]

# The last hand-carried file, 307 MB. Searched for properly and it is not
# published anywhere reachable: absent from Civitai's by-hash index, from their
# name search (eighteen "detail enhancer" LoRAs, none with this digest), from
# Hugging Face by name and by full-text, from the Civitai account of the
# FusionX author whose bundle it shipped in, and from anthonyluu, who mirrored
# its sibling Wan14B_RealismBoost but not this. Assume it is delisted.
#
# Listed here so the boot log names it instead of letting ComfyUI fail later
# with a bare file-not-found. The sha256 of the known-good local original is
# recorded so a copy that reaches the volume can be checked rather than trusted
# by filename -- and so it can be moved into WANTED unchanged if it is ever
# re-hosted, including to a repo of your own.
MANUAL = [
    ("loras", "DetailEnhancerV1.safetensors", "celery-man v2v",
     "9ab17e3520fd2b8f97ea25f987017766ec8e76939b3445caa994882966e6d47e"),
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


def resolve(item):
    """URL for one WANTED entry, or None if nothing carries it."""
    if item.get("civitai"):
        if not CIVITAI_TOKEN:
            return None
        return "https://civitai.com/api/download/models/%d?token=%s" % (
            item["civitai"], CIVITAI_TOKEN)
    repo, remote = item.get("repo"), item.get("remote")
    if repo and remote:
        return "https://huggingface.co/%s/resolve/main/%s" % (repo, remote)
    for r in ([repo] if repo else REPOS):
        path = tree(r).get(item["name"])
        if path:
            return "https://huggingface.co/%s/resolve/main/%s" % (r, path)
    return None


def download(url, dest, sha256=None):
    part = dest + ".part"
    print("  downloading %s" % os.path.basename(dest), flush=True)
    try:
        digest = hashlib.sha256()
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
                digest.update(chunk)
                done += len(chunk)
                if total and done - mark > (total // 10 or 1):
                    mark = done
                    print("    %s %d%%" % (os.path.basename(dest), 100 * done // total), flush=True)
        # Hashed on the way past, so verification costs nothing extra. A mirror
        # that swapped its contents is caught here rather than surfacing as a
        # render that looks subtly wrong.
        if sha256 and digest.hexdigest() != sha256:
            os.remove(part)
            print("  ! DIGEST MISMATCH %s\n      expected %s\n      got      %s"
                  % (os.path.basename(dest), sha256, digest.hexdigest()), flush=True)
            return False
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
    needs_token = []
    for item in WANTED:
        name = item["name"]
        d = os.path.join(MODELS, item["subdir"])
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        if os.path.exists(dest):
            print("  have %s" % name, flush=True)
            continue
        url = resolve(item)
        if not url:
            if item.get("civitai"):
                # Not a failure -- it is the documented unconfigured state.
                needs_token.append((name, item["civitai"]))
                continue
            where = item.get("repo") or ", ".join(REPOS)
            print("  ! could not resolve %s in %s" % (name, where), flush=True)
            missing.append(name)
            continue
        if not download(url, dest, item.get("sha256")):
            missing.append(name)

    absent_manual = []
    for subdir, name, used_by, sha in MANUAL:
        if not os.path.exists(os.path.join(MODELS, subdir, name)):
            absent_manual.append((name, used_by))

    print("", flush=True)
    if needs_token:
        print("=" * 68, flush=True)
        print("CIVITAI_TOKEN not set -- these are downloadable but were skipped:", flush=True)
        for name, ver in needs_token:
            print("   models/loras/%-42s  (civitai %s)" % (name, ver), flush=True)
        print("", flush=True)
        print("Set CIVITAI_TOKEN on the pod and restart to fetch them, or copy", flush=True)
        print("them onto the volume by hand. Get a token from Civitai under", flush=True)
        print("Account Settings -> API Keys.", flush=True)
        print("=" * 68, flush=True)
        print("", flush=True)
    if absent_manual:
        print("=" * 68, flush=True)
        print("MANUAL UPLOAD REQUIRED -- not published anywhere reachable, so", flush=True)
        print("this must be copied to the volume yourself (see README):", flush=True)
        for name, used_by in absent_manual:
            print("   models/loras/%-42s  (%s)" % (name, used_by), flush=True)
        print("", flush=True)
        print("Graphs that need it will fail to queue until it is present.", flush=True)
        print("The wan-flower INIT graph does not need it and works now.", flush=True)
        print("=" * 68, flush=True)
    if missing:
        print("FAILED to fetch: %s" % ", ".join(missing), flush=True)
        return 1
    print("model payload ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
