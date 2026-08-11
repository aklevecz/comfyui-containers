#!/usr/bin/env python3
"""Drive a full wan-flower audio-visual: upload mask, init, N extends, stitch.

  python scripts/wan_flower_chain.py --host https://<pod>-8188.proxy.runpod.net \
      --mask masks/track.mp4 --prompt "blooming flowers" --passes 30

Output is one continuous mp4 of 81 + passes*66 frames at 30 fps, downloaded
locally. 30 passes is ~69 seconds and ~45 min of GPU time at 93 s/pass.

Two things this handles that are easy to get wrong by hand:

curl, not urllib. The RunPod proxy is behind Cloudflare, which bans non-browser
user agents -- urllib gets 403 "error code 1010", which reads like an auth or
ComfyUI failure and is neither.

Saved graphs drift from their node packs. The packs are cloned unpinned from
main, so upstream adds required inputs that a July workflow has never heard of
and ComfyUI rejects the whole prompt with required_input_missing. Every node is
reconciled against the live /object_info before queueing.
"""
import argparse, json, os, subprocess, sys, time
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = os.path.join(HERE, "..", "wan-blackwell", "workflows")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"


def curl(host, args, timeout=900):
    r = subprocess.run(["curl", "-s", "-m", str(timeout), "-H", "User-Agent: " + UA] + args,
                       capture_output=True, text=True)
    return r.stdout


def api(host, path, timeout=300):
    return curl(host, [host.rstrip("/") + path], timeout)


def post_prompt(host, wf, scratch, label):
    body = os.path.join(scratch, "payload.json")
    json.dump({"prompt": wf, "client_id": "wan-flower-chain"},
              open(body, "w", encoding="utf-8"))
    out = curl(host, ["-X", "POST", "-H", "Content-Type: application/json",
                      "--data-binary", "@" + body, host.rstrip("/") + "/prompt"])
    try:
        res = json.loads(out)
    except Exception:
        print("  %s: unparseable response: %s" % (label, out[:300]), flush=True)
        return None
    if res.get("node_errors"):
        print("  %s REJECTED: %s" % (label, json.dumps(res["node_errors"])[:700]), flush=True)
        return None
    return res.get("prompt_id")


def wait_for_pod(host, timeout=600):
    """Block until ComfyUI answers again. After a restart the HTTP endpoint
    comes back well before the models are re-resident, but /object_info being
    servable is enough to queue against."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if api(host, "/system_stats", 30).strip().startswith("{"):
            return True
        time.sleep(10)
    return False


def run_with_retry(host, wf, scratch, label, attempts=3):
    """A pass can vanish: ComfyUI restarts, /history is wiped, and the queued
    prompt is silently dropped -- observed twice at ~45 minutes into a chain.
    The previous segment is already on disk, so re-queueing the same pass
    resumes the chain rather than ending it. Without this a restart costs every
    remaining pass; the first long run lost 2 of 30 to exactly this."""
    for k in range(attempts):
        pid = post_prompt(host, wf, scratch, label)
        if pid:
            good, mp4 = wait(host, pid, label)
            if good:
                return True, mp4
        if k < attempts - 1:
            print("  %s: lost -- waiting for pod, then retry %d/%d"
                  % (label, k + 2, attempts), flush=True)
            if not wait_for_pod(host):
                print("  %s: pod never came back" % label, flush=True)
                return False, None
    return False, None


def wait(host, pid, label, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(10)
        h = api(host, "/history/" + pid)
        if h.strip() and h.strip() != "{}":
            try:
                e = json.loads(h)[pid]
            except Exception:
                continue
            st = e.get("status", {}).get("status_str")
            mp4 = None
            for o in e.get("outputs", {}).values():
                for it in (o.get("gifs") or []):
                    mp4 = (it.get("filename"), it.get("subfolder"))
            print("  %s %s in %ds%s" % (label, st, int(time.time() - t0),
                                        " -> " + mp4[0] if mp4 else ""), flush=True)
            return (st == "success", mp4)
    print("  %s TIMED OUT after %ds" % (label, timeout), flush=True)
    return (False, None)


def load(name):
    with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as f:
        return json.load(f)


def reconcile(wf, oi):
    """Fill required inputs the saved graph predates, from the server's own
    defaults. Silent no-op when the graph is current."""
    filled = 0
    for node in wf.values():
        ct = node["class_type"]
        if ct not in oi:
            continue
        for k, v in oi[ct]["input"].get("required", {}).items():
            if k in node.get("inputs", {}):
                continue
            meta = v[1] if len(v) > 1 else {}
            if isinstance(meta, dict) and "default" in meta:
                node["inputs"][k] = meta["default"]
                filled += 1
    return filled


def drop_missing_loras(wf, avail):
    """A LoRA file that is absent does not degrade quality -- ComfyUI refuses
    the whole prompt. Cut the node and rewire prev_lora past it."""
    dropped = []
    for nid in [k for k, n in wf.items() if n["class_type"] == "WanVideoLoraSelect"]:
        node = wf.get(nid)
        if node is None or node["inputs"].get("lora") in avail:
            continue
        prev = node["inputs"].get("prev_lora")
        for other in wf.values():
            for k, v in list(other.get("inputs", {}).items()):
                if isinstance(v, list) and len(v) == 2 and v[0] == nid:
                    if prev is not None:
                        other["inputs"][k] = prev
                    else:
                        other["inputs"].pop(k, None)
        dropped.append(node["inputs"].get("lora"))
        del wf[nid]
    return dropped


def by_title(wf, title):
    return [n for n in wf.values() if (n.get("_meta") or {}).get("title") == title]


def lora_options(oi):
    spec = oi.get("WanVideoLoraSelect", {}).get("input", {})
    for grp in ("required", "optional"):
        if "lora" in spec.get(grp, {}):
            v = spec[grp]["lora"][0]
            return set(v) if isinstance(v, list) else set()
    return set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="https://<pod-id>-8188.proxy.runpod.net")
    p.add_argument("--mask", required=True, help="local mask video to upload")
    p.add_argument("--prompt", default="blooming flowers")
    p.add_argument("--passes", type=int, default=30, help="66 new frames each")
    p.add_argument("--project", default="flower")
    p.add_argument("--inverted", action="store_true", help="generate outside the mask")
    p.add_argument("--out", default=".", help="where to save the finished mp4")
    a = p.parse_args()

    host = a.host.rstrip("/")
    scratch = os.path.join(HERE, ".chain-tmp")
    os.makedirs(scratch, exist_ok=True)

    stats = api(host, "/system_stats", 60)
    if not stats.strip().startswith("{"):
        sys.exit("pod unreachable at %s (a 404 here means nothing is listening)" % host)
    dev = json.loads(stats).get("devices", [{}])[0].get("name", "?")
    print("pod: %s" % dev, flush=True)

    oi = json.loads(api(host, "/object_info", 300))
    avail = lora_options(oi)

    base = os.path.basename(a.mask)
    print("uploading %s (%.1f MB)" % (base, os.path.getsize(a.mask) / 1048576), flush=True)
    up = curl(host, ["-X", "POST", "-F", "image=@" + a.mask, "-F", "type=input",
                     "-F", "overwrite=true", host + "/upload/image"])
    if base not in up:
        sys.exit("mask upload failed: %s" % up[:300])
    mask_path = "/opt/ComfyUI/input/" + base

    suffix = "_inverted" if a.inverted else ""
    init_wf = load("wan_audio_reactive_fast_init%s.json" % suffix)
    ext_wf = load("wan_audio_reactive_fast_extend%s.json" % suffix)

    filled = reconcile(init_wf, oi)
    dropped = drop_missing_loras(init_wf, avail)
    print("reconciled %d missing input(s); dropped loras: %s" % (filled, dropped or "none"), flush=True)

    # init: node 546 is the mask loader in the init graph, which has no
    # MASK_PATH title the way extend does.
    for nid, node in init_wf.items():
        if node["class_type"] == "VHS_LoadVideoPath":
            node["inputs"]["video"] = mask_path
    for node in init_wf.values():
        if node["class_type"] == "WanVideoTextEncode":
            node["inputs"]["positive_prompt"] = a.prompt
        if node["class_type"] == "VHS_VideoCombine":
            node["inputs"]["filename_prefix"] = "%s/1" % a.project
        if node["class_type"] == "SaveImage":
            node["inputs"]["filename_prefix"] = "%s/frames/1" % a.project

    print("\ninit (81 frames), prompt: %r" % a.prompt, flush=True)
    if not run_with_retry(host, init_wf, scratch, "init")[0]:
        sys.exit("init failed -- nothing to extend from")

    ok = 0
    for n in range(1, a.passes + 1):
        wf = deepcopy(ext_wf)
        reconcile(wf, oi)
        drop_missing_loras(wf, avail)
        for node in by_title(wf, "MASK_PATH"):
            node["inputs"]["video"] = mask_path
        for node in by_title(wf, "PROJECT_PATH"):
            for k in ("text", "text_0"):
                if k in node["inputs"]:
                    node["inputs"][k] = "%s/1_%05d.mp4" % (a.project, n)
        for node in by_title(wf, "Incrementer"):
            node["inputs"]["seed"] = n
        for node in by_title(wf, "FINAL_OUTPUT"):
            node["inputs"]["filename_prefix"] = "%s/1" % a.project
        for node in by_title(wf, "CONCAT_FRAMES"):
            node["inputs"]["filename_prefix"] = "%s/frames/1" % a.project
        for node in wf.values():
            if node["class_type"] == "WanVideoTextEncode":
                node["inputs"]["positive_prompt"] = a.prompt
        if not run_with_retry(host, wf, scratch, "pass %d" % n)[0]:
            print("STOPPING at pass %d -- keeping the %d that landed" % (n, ok), flush=True)
            break
        ok += 1

    frames = 81 + ok * 66
    print("\n%d passes | %d frames | %.1fs at 30fps" % (ok, frames, frames / 30.0), flush=True)

    # Stitch on the pod: the frames are already there next to a working ffmpeg,
    # which beats pulling thousands of PNGs down to assemble locally.
    stitch = {
        "1": {"class_type": "VHS_LoadImagesPath",
              "inputs": {"directory": "/opt/ComfyUI/output/%s/frames" % a.project,
                         "image_load_cap": 0, "skip_first_images": 0, "select_every_nth": 1}},
        "2": {"class_type": "VHS_VideoCombine",
              "inputs": {"images": ["1", 0], "frame_rate": 30, "loop_count": 0,
                         "filename_prefix": "%s/FULL" % a.project, "format": "video/h264-mp4",
                         "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False,
                         "pingpong": False, "save_output": True}},
    }
    print("stitching", flush=True)
    pid = post_prompt(host, stitch, scratch, "stitch")
    if not pid:
        sys.exit("stitch rejected")
    good, mp4 = wait(host, pid, "stitch")
    if not good or not mp4:
        sys.exit("stitch failed")

    name, sub = mp4
    dest = os.path.join(a.out, "%s_%dframes.mp4" % (a.project, frames))
    curl(host, ["-G", "--data-urlencode", "filename=" + name,
                "--data-urlencode", "subfolder=" + (sub or ""),
                "--data-urlencode", "type=output", host + "/view", "-o", dest], 1800)
    print("\nsaved %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1048576), flush=True)
    print("NOTE: output/ lives on the pod's volume disk, which is deleted when the", flush=True)
    print("pod is terminated. This local copy is the one that survives.", flush=True)


if __name__ == "__main__":
    main()
