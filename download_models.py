"""
First-boot model download.

Downloads LTX-2.3 weights + Gemma 3 text encoder to /runpod-volume/ltx-2.3/.
Runs once per Network Volume (idempotent — skips files that already exist).
"""

import os
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


def main():
    print(f"[download] Volume: {VOLUME}")
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
        )
        print(f"[download] DONE {path}")

    # LTX-2.3 expects Gemma hidden_size=3840 → that's Gemma 3 12B (4B has 2560, 1B has 1152).
    # Verified via runtime size_mismatch error from a 4B download attempt.
    gemma_id = os.environ.get("GEMMA_REPO", "google/gemma-3-12b-it")
    sentinel = GEMMA_DIR / f".downloaded_{gemma_id.replace('/', '_')}"
    if not sentinel.exists():
        print(f"[download] ... {gemma_id}")
        snapshot_download(
            repo_id=gemma_id,
            local_dir=str(GEMMA_DIR),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.py"],
        )
        sentinel.write_text("ok")
        print(f"[download] DONE Gemma")
    else:
        print("[download] OK  Gemma")

    print("[download] all models ready")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[download] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
