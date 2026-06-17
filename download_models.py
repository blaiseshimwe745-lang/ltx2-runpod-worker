import os
import sys
import traceback
from pathlib import Path
from huggingface_hub import snapshot_download

VOLUME = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
WAN_DIR = VOLUME / "wan2-animate-14b-diffusers"
REPO_ID = "Wan-AI/Wan2.2-Animate-14B-Diffusers"

def _hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None

def main():
    print(f"[download] Volume: {VOLUME}")
    token = _hf_token()
    print(f"[download] HF token present: {bool(token)}")
    
    # We download the entire Diffusers repository
    print(f"[download] Downloading {REPO_ID} to {WAN_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(WAN_DIR),
        token=token,
    )
    print("[download] DONE")

def diagnose():
    info = {
        "volume": str(VOLUME),
        "wan_dir_exists": WAN_DIR.exists(),
        "hf_token_present": bool(_hf_token())
    }
    try:
        files = []
        for p in sorted(WAN_DIR.rglob("*")):
            if p.is_file():
                files.append({"name": str(p.relative_to(WAN_DIR)), "mb": round(p.stat().st_size / 1e6, 1)})
        info["files"] = files[:50]
        info["total_mb"] = round(sum(f["mb"] for f in files), 1)
    except Exception as e:
        info["diagnose_error"] = str(e)
    return info

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[download] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
