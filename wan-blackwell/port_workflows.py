#!/usr/bin/env python3
"""
Port the wan-flower workflows from the Windows portable install to the Linux
container layout, and retune them for a big-VRAM card.

Deliberately conservative: graph TOPOLOGY is never touched. Only literal input
values change, so the node-title contract App.jsx depends on stays intact.
Every edit is printed, so the diff is auditable.

Run from this directory:  python port_workflows.py
"""
import io
import json
import os
import sys

WORKFLOWS = "workflows"

# Where ComfyUI lives in the container. entrypoint.sh symlinks output/ onto the
# network volume, so this path is stable whether or not a volume is attached.
OUTPUT_DIR = "/opt/ComfyUI/output/"

# RTX PRO 6000 Blackwell has 96 GB. Block swapping exists only to fit a 14B
# model on a small card; every swapped block is a round trip to system RAM on
# every step. Setting this to 0 is the single biggest speedup of this port.
BLOCKS_TO_SWAP = 0

# The example renders (fast_flower.mp4) were made at 720x720; the JSONs had
# drifted to 512. Match the examples.
RESOLUTION = 720

edits = []


def note(f, node, field, old, new):
    if old == new:
        return False
    edits.append((f, node, field, old, new))
    return True


def port(fname):
    path = os.path.join(WORKFLOWS, fname)
    d = json.load(io.open(path, encoding="utf-8"))

    for nid, node in d.items():
        ct = node.get("class_type")
        ins = node.get("inputs", {})
        title = (node.get("_meta") or {}).get("title", "")

        # --- 1. Windows separators baked into the path-assembly chain ---------
        # StringConcatenate joins the output dir, project name and counter to
        # build the next segment's path. The delimiter was a literal backslash.
        if ct == "StringConcatenate":
            for field in ("delimiter", "string_a", "string_b"):
                v = ins.get(field)
                if isinstance(v, str) and "\\" in v:
                    new = v.replace("\\", "/")
                    if note(fname, nid, field, v, new):
                        ins[field] = new

        # --- 2. The hardcoded Windows output root ----------------------------
        if ct == "easy string":
            v = ins.get("value", "")
            if isinstance(v, str) and ("C:\\" in v or "c:\\" in v):
                if note(fname, nid, "value", v, OUTPUT_DIR):
                    ins["value"] = OUTPUT_DIR

        # --- 3. Absolute mask-video paths, and the stray quote characters -----
        # Node 546's path was wrapped in literal double quotes inside the string,
        # which ComfyUI passes through verbatim and then fails to open.
        if ct in ("VHS_LoadVideoPath", "VHS_LoadVideo"):
            v = ins.get("video")
            if isinstance(v, str):
                new = v.strip().strip('"').replace("\\", "/")
                if ":" in new[:3] or new.startswith("//"):
                    # An absolute Windows path -> point at the container input dir
                    new = "/opt/ComfyUI/input/" + os.path.basename(new)
                if note(fname, nid, "video", v, new):
                    ins["video"] = new

        # --- 4. ShowText caches: stale Windows strings shown in the UI --------
        # Purely cosmetic, but leaving them makes the graph look broken.
        if ct == "ShowText|pysssss":
            for field in list(ins):
                v = ins.get(field)
                if isinstance(v, str) and ("C:" in v or "\\" in v):
                    if note(fname, nid, field, v, ""):
                        ins[field] = ""

        # --- 5. ttN text PROJECT_PATH: normalise separators -------------------
        if ct == "ttN text":
            v = ins.get("text")
            if isinstance(v, str) and "\\" in v:
                new = v.replace("\\", "/")
                if note(fname, nid, "text", v, new):
                    ins["text"] = new

        # --- 6. Big-VRAM retune ----------------------------------------------
        if ct == "WanVideoBlockSwap":
            v = ins.get("blocks_to_swap")
            if note(fname, nid, "blocks_to_swap", v, BLOCKS_TO_SWAP):
                ins["blocks_to_swap"] = BLOCKS_TO_SWAP

        # --- 6b. LoRA paths -------------------------------------------------
        # The v2v graph refers to its LoRAs as "wan\Name.safetensors" -- a
        # Windows subfolder inside models/loras. Two problems on Linux: the
        # backslash is a literal filename character, and download_models.py
        # flattens everything into models/loras/. Reduce to the basename.
        if ct == "WanVideoLoraSelect":
            v = ins.get("lora")
            if isinstance(v, str) and ("\\" in v or "/" in v):
                new = v.replace("\\", "/").rsplit("/", 1)[-1]
                if note(fname, nid, "lora", v, new):
                    ins["lora"] = new

        # --- 7. Resolution: match the example renders -------------------------
        if ct == "Primitive integer [Crystools]" and title in ("Width", "Height"):
            v = ins.get("int")
            if note(fname, nid, "int(%s)" % title, v, RESOLUTION):
                ins["int"] = RESOLUTION

        if ct == "easy int" and ins.get("value") == 512:
            if note(fname, nid, "value(512->res)", 512, RESOLUTION):
                ins["value"] = RESOLUTION

        # EmptyImage nodes that hardcode 512 alongside a batch size
        if ct == "EmptyImage":
            for field in ("width", "height"):
                if ins.get(field) == 512:
                    if note(fname, nid, field, 512, RESOLUTION):
                        ins[field] = RESOLUTION

    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
    return d


def main():
    if not os.path.isdir(WORKFLOWS):
        sys.exit("run this from the template directory (no ./workflows found)")

    for f in sorted(os.listdir(WORKFLOWS)):
        if f.endswith(".json"):
            port(f)

    if not edits:
        print("no changes -- already ported")
        return

    w = max(len(e[0]) for e in edits)
    for f, nid, field, old, new in edits:
        so, sn = repr(old), repr(new)
        if len(so) > 62:
            so = so[:59] + "..."
        if len(sn) > 62:
            sn = sn[:59] + "..."
        print("%-*s  node %-4s %-18s %s -> %s" % (w, f, nid, field, so, sn))
    print("\n%d edits across %d files" % (edits.__len__(), len(set(e[0] for e in edits))))


if __name__ == "__main__":
    main()
