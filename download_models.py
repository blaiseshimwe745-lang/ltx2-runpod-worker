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


def clean_mixed_gemma_shards():
    """Remove safetensors that don't belong to the 12B model + stale sentinels.

    Root cause of the recurring failure: the gemma dir accumulated shards from
    MULTIPLE variants (4B = *-of-00002, 12B = *-of-00005). LTX globs all
    *.safetensors and crashes on the dimension mismatch. We keep only the files
    referenced by the current model.safetensors.index.json (the 12B set).
    """
    import re
    index = GEMMA_DIR / "model.safetensors.index.json"
    valid = set()
    if index.exists():
        try:
            data = json.loads(index.read_text())
            valid = set(data.get("weight_map", {}).values())
            print(f"[clean] 12B index lists {len(valid)} shard files: {sorted(valid)}")
        except Exception as e:
            print(f"[clean] could not parse index: {e}")

    removed = []
    for f in GEMMA_DIR.glob("*.safetensors"):
        if valid and f.name not in valid:
            f.unlink(missing_ok=True)
            removed.append(f.name)
    # stale sentinels
    for s in GEMMA_DIR.glob(".downloaded*"):
        s.unlink(missing_ok=True)
        removed.append(s.name)
    print(f"[clean] removed {len(removed)}: {removed}")
    return removed


def consolidate_gemma():
    """LTX's loader opens a SINGLE gemma-3/model.safetensors. The HF 12B download
    is sharded (model-0000X-of-00005). Merge the shards into one file and drop
    the shards + index so the loader finds exactly one model.safetensors."""
    from safetensors.torch import load_file, save_file
    single = GEMMA_DIR / "model.safetensors"
    shards = sorted(GEMMA_DIR.glob("model-*-of-*.safetensors"))
    if single.exists() and not shards:
        print("[consolidate] already a single model.safetensors")
        return
    if not shards:
        print("[consolidate] no shards found, nothing to merge")
        return
    print(f"[consolidate] merging {len(shards)} shards into model.safetensors ...")
    merged = {}
    for sh in shards:
        d = load_file(str(sh))
        # Clone tensors to RAM to drop mmap references
        for k, v in d.items():
            merged[k] = v.clone()
        del d
        # Delete shard from disk immediately to stay under 100GB limit
        sh.unlink(missing_ok=True)
        print(f"[consolidate] loaded and deleted {sh.name}")
        
    (GEMMA_DIR / "model.safetensors.index.json").unlink(missing_ok=True)
    
    print(f"[consolidate] writing single model.safetensors to disk...")
    save_file(merged, str(single))
    del merged
    print(f"[consolidate] done: {single.stat().st_size/1e9:.1f} GB single file")


def main(force_gemma=False):
    print(f"[download] Volume: {VOLUME}")
    token = _hf_token()
    print(f"[download] HF token present: {bool(token)} (len={len(token) if token else 0})")
    if force_gemma:
        clean_mixed_gemma_shards()

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
        # If config says wrong variant AND we're not in surgical-clean mode, full wipe.
        # In force mode we already cleaned mixed shards surgically — just complete 12B.
        if not force_gemma and hs not in (None, REQUIRED_HIDDEN_SIZE):
            print(f"[download] full wipe {GEMMA_DIR} (wrong variant {hs})")
            _wipe_dir(GEMMA_DIR)
            GEMMA_DIR.mkdir(parents=True, exist_ok=True)
        if not token:
            print(f"[download] FATAL: HF token required for gated {GEMMA_REPO}", file=sys.stderr)
            sys.exit(2)
        print(f"[download] ... completing {GEMMA_REPO} (resumes existing shards)")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=str(GEMMA_DIR),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.safetensors.index.json", "*.py"],
            token=token,
        )
        # After download, clean any leftover non-12B shards again (belt & suspenders)
        clean_mixed_gemma_shards()
        hs2 = _gemma_hidden_size_on_disk()
        print(f"[download] post-download Gemma hidden_size: {hs2}")
        if hs2 != REQUIRED_HIDDEN_SIZE:
            print(f"[download] FATAL: gemma hidden_size {hs2} != {REQUIRED_HIDDEN_SIZE}", file=sys.stderr)
            sys.exit(3)
        print("[download] DONE Gemma 12B verified")
    else:
        print("[download] OK  Gemma 12B already correct")

    # LTX wants a single model.safetensors — merge shards if needed
    consolidate_gemma()

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
