"""
First-boot model download — CONFIG-VERIFYING & SELF-HEALING.

Instead of trusting a sentinel file (which can lie about what's actually on
disk), this reads the Gemma config.json on the volume and checks hidden_size.
LTX-2.3 needs Gemma 3 12B → hidden_size MUST be 3840. Anything else (e.g. the
4B variant with 2560) triggers a full wipe + correct re-download.
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

VOLUME = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
LTX_DIR = VOLUME / "ltx-2.3"
GEMMA_DIR = VOLUME / "gemma-3"
LTX_DIR.mkdir(parents=True, exist_ok=True)
GEMMA_DIR.mkdir(parents=True, exist_ok=True)

LTX_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = os.environ.get("GEMMA_REPO", "google/gemma-3-12b-it")
REQUIRED_HIDDEN_SIZE = 3840  # Gemma 3 12B

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


def _gemma_hidden_size_on_disk():
    """Read hidden_size from whatever Gemma config.json is on the volume.
    Returns int or None if no readable config found."""
    for cfg in GEMMA_DIR.rglob("config.json"):
        try:
            data = json.loads(cfg.read_text())
            # Gemma3 config nests text params under text_config
            tc = data.get("text_config", data)
            hs = tc.get("hidden_size") or data.get("hidden_size")
            if hs:
                return int(hs)
        except Exception:
            continue
    return None


def diagnose():
    """Return a dict describing what's actually on the volume."""
    info = {"volume": str(VOLUME), "gemma_dir": str(GEMMA_DIR)}
    try:
        info["gemma_hidden_size"] = _gemma_hidden_size_on_disk()
        files = []
        for p in sorted(GEMMA_DIR.rglob("*")):
            if p.is_file():
                files.append({"name": str(p.relative_to(GEMMA_DIR)), "mb": round(p.stat().st_size / 1e6, 1)})
        info["gemma_files"] = files[:50]
        info["gemma_total_mb"] = round(sum(f["mb"] for f in files), 1)
        ltx_files = []
        for p in sorted(LTX_DIR.glob("*")):
            if p.is_file():
                ltx_files.append({"name": p.name, "mb": round(p.stat().st_size / 1e6, 1)})
        info["ltx_files"] = ltx_files
        info["hf_token_present"] = bool(_hf_token())
    except Exception as e:
        info["diagnose_error"] = str(e)
    return info


def main(force_gemma=False):
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
        path = hf_hub_download(repo_id=LTX_REPO, filename=fname, local_dir=str(LTX_DIR), token=token)
        print(f"[download] DONE {path}")

    # ---- Gemma: verify by config, not by sentinel ----
    hs = _gemma_hidden_size_on_disk()
    print(f"[download] Gemma hidden_size on disk: {hs} (need {REQUIRED_HIDDEN_SIZE})")

    needs_download = force_gemma or hs != REQUIRED_HIDDEN_SIZE
    if needs_download:
        print(f"[download] wiping {GEMMA_DIR} (wrong/missing gemma)")
        _wipe_dir(GEMMA_DIR)
        GEMMA_DIR.mkdir(parents=True, exist_ok=True)
        if not token:
            print(f"[download] FATAL: HF token required for gated {GEMMA_REPO}", file=sys.stderr)
            sys.exit(2)
        print(f"[download] ... {GEMMA_REPO} (~24 GB, several minutes)")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=str(GEMMA_DIR),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.safetensors.index.json", "*.py"],
            token=token,
        )
        hs2 = _gemma_hidden_size_on_disk()
        print(f"[download] post-download Gemma hidden_size: {hs2}")
        if hs2 != REQUIRED_HIDDEN_SIZE:
            print(f"[download] FATAL: downloaded gemma hidden_size {hs2} != {REQUIRED_HIDDEN_SIZE}", file=sys.stderr)
            sys.exit(3)
        print("[download] DONE Gemma 12B verified")
    else:
        print("[download] OK  Gemma 12B already correct")

    print("[download] all models ready")


if __name__ == "__main__":
    try:
        main(force_gemma="--force-gemma" in sys.argv)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[download] FAIL: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
