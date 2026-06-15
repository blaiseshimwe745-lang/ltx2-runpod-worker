"""
First-boot model download.

Downloads LTX-2.3 weights + Gemma 3 text encoder to /runpod-volume/.
Wipes any stale Gemma variant before downloading the correct one.
Fails hard if anything fails so the worker logs show the real error.
"""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

VOLUME = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
LTX_DIR = VOLUME / "ltx-2.3"
GEMMA_DIR = VOLUME / "gemma-3"
LTX_DIR.mkdir(parents=True, exist_ok=True)
GEMMA_DIR.mkdir(parents=True, exist_ok=True)

LTX_REPO = "Lightricks/LTX-2.3"

LTX_FILES = [
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1_000_000


def _hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def _wipe_dir(d: Path):
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except Exception:
                pass


def main():
    print(f"[download] Volume: {VOLUME}")
    token = _hf_token()
    print(f"[download] HF token present: {bool(token)} (len={len(token) if token else 0})")

    # ---- LTX-2.3 weights ----
    for fname in LTX_FILES:
        out = LTX_DIR / fname
        if have(out):
            print(f"[download] OK  {fname}")
            continue
        print(f"[download] ... {fname}")
        path = hf_hub_download(
            repo_id=LTX_REPO,
            filename=fname,
            local_dir=str(LTX_DIR),
            token=token,
        )
        print(f"[download] DONE {path}")

    # ---- Gemma 3 12B text encoder (LTX-2 expects hidden_size=3840) ----
    gemma_id = os.environ.get("GEMMA_REPO", "google/gemma-3-12b-it")
    sentinel = GEMMA_DIR / f".downloaded_{gemma_id.replace('/', '_')}"

    if not sentinel.exists():
        # Wipe any previously-downloaded different Gemma variant — otherwise
        # the safetensors mix and LTX-2's loader (rglob *.safetensors) will
        # try to load weights from BOTH variants into one model.
        print(f"[download] wiping stale gemma at {GEMMA_DIR}")
        _wipe_dir(GEMMA_DIR)
        GEMMA_DIR.mkdir(parents=True, exist_ok=True)

        if not token:
            print(
                "[download] FATAL: HF token required for gated model "
                f"{gemma_id}. Set HF_TOKEN env var on the endpoint.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[download] ... {gemma_id} (this is ~24 GB, expect ~5 min)")
        snapshot_download(
            repo_id=gemma_id,
            local_dir=str(GEMMA_DIR),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.py"],
            token=token,
        )
        sentinel.write_text("ok")
        print(f"[download] DONE Gemma")
    else:
        print(f"[download] OK  Gemma ({gemma_id})")

    print("[download] all models ready")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"[download] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
