"""
RunPod Serverless worker for LTX-2.3 (DistilledPipeline).

Notes:
  - The LTX-2 Python API is brand new (no diffusers integration yet). Imports
    below are best-effort against the public package layout. If the first run
    fails on an ImportError, paste the traceback back and we'll adjust the
    import path — the rest of the handler is generic.
  - Model weights live on the mounted Network Volume at /runpod-volume.
  - First cold-start downloads weights (~5-8 min); subsequent boots ~10-20s.

Input schema:
{
  "input": {
    "prompt": "...",
    "image": "<data:image/png;base64,... or https URL>",  # optional, for I2V
    "width":  768,
    "height": 512,
    "duration_seconds": 5.0,
    "fps": 24,
    "seed": 12345,
    "negative_prompt": "",
    "enhance_prompt": false
  }
}

Output:
{ "video": "data:video/mp4;base64,...", "width": 768, "height": 512, "fps": 24, "frames": 121, "seed_used": 12345 }
"""

import base64
import io
import os
import random
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import runpod

# ---- Boot: make sure models are on the volume ----
print("[boot] checking models on volume…")
try:
    subprocess.check_call([sys.executable, "/app/download_models.py"])
except Exception as e:
    print(f"[boot] download_models.py failed: {e}", file=sys.stderr)
    # Don't crash — handler returns the error on first invocation.

# ---- Imports (deferred so the handler can return a clean error message) ----
_pipe = None
_load_error = None


def _load_pipeline():
    global _pipe, _load_error
    if _pipe is not None or _load_error is not None:
        return
    try:
        import torch  # noqa: F401
        # LTX-2 package import — best-effort.
        # If this errors, paste the message; we'll fix the import path.
        try:
            from ltx_pipelines import DistilledPipeline  # type: ignore
        except Exception:
            from ltx_pipelines.distilled_pipeline import DistilledPipeline  # type: ignore

        import torch
        volume = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
        ltx_dir = volume / "ltx-2.3"
        gemma_dir = volume / "gemma-3"

        print("[boot] Loading DistilledPipeline…")
        _pipe = DistilledPipeline.from_pretrained(
            checkpoint_path=str(ltx_dir / "ltx-2.3-22b-distilled-1.1.safetensors"),
            spatial_upscaler_path=str(ltx_dir / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            distilled_lora_path=str(ltx_dir / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
            text_encoder_path=str(gemma_dir),
            dtype=torch.bfloat16,
        )
        _pipe = _pipe.to("cuda")
        print("[boot] Pipeline ready.")
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        print(f"[boot] Pipeline load failed: {_load_error}", file=sys.stderr)


def _round_dim(x: int) -> int:
    return max(256, (int(x) // 32) * 32)


def _round_frames(n: int) -> int:
    # Must be 8N + 1
    return max(9, ((int(n) - 1) // 8) * 8 + 1)


def _decode_input_image(s):
    if not s:
        return None
    from PIL import Image
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(s, timeout=30) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return None


def _frames_to_mp4(frames, fps: int, audio=None) -> bytes:
    """Encode a list of PIL frames (and optional audio waveform) to MP4 bytes."""
    import imageio.v3 as iio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp = f.name
    try:
        # Note: audio muxing left as TODO — LTX-2 returns audio too but we keep
        # this handler video-only on first iteration. Once video works we can
        # add ffmpeg-based audio mux.
        iio.imwrite(tmp, frames, fps=fps, codec="libx264", quality=8)
        return Path(tmp).read_bytes()
    finally:
        try: os.unlink(tmp)
        except Exception: pass


def handler(event):
    try:
        inp = (event or {}).get("input", {}) or {}
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "missing 'prompt'"}

        # Lazy-load pipeline
        if _pipe is None:
            _load_pipeline()
        if _load_error:
            return {"error": "pipeline_load_failed", "detail": _load_error}

        import torch

        width = _round_dim(inp.get("width", 768))
        height = _round_dim(inp.get("height", 512))
        fps = int(inp.get("fps", 24))
        duration = float(inp.get("duration_seconds", 5.0))
        num_frames = _round_frames(int(duration * fps))
        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        image = _decode_input_image(inp.get("image"))

        gen = torch.Generator(device="cuda").manual_seed(seed)

        call_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            generator=gen,
            negative_prompt=inp.get("negative_prompt") or None,
        )
        if image is not None:
            call_kwargs["image"] = image
        if inp.get("enhance_prompt"):
            call_kwargs["enhance_prompt"] = True

        print(f"[run] {width}x{height} f{num_frames} fps{fps} seed={seed}")
        out = _pipe(**call_kwargs)

        # Best-effort frame extraction (covers common return shapes)
        frames = None
        for attr in ("frames", "video", "videos", "images"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if v is not None:
                    frames = v
                    break
        if frames is None and isinstance(out, (list, tuple)):
            frames = out[0]

        if frames is None:
            return {"error": "no frames in pipeline output", "out_type": type(out).__name__}

        # If frames is a tensor (T,C,H,W) or (B,T,C,H,W), convert
        if hasattr(frames, "detach"):
            t = frames.detach().to("cpu")
            if t.dim() == 5:
                t = t[0]
            # (T,C,H,W) -> (T,H,W,C) uint8
            t = (t.clamp(-1, 1) + 1) / 2 if t.min() < 0 else t.clamp(0, 1)
            t = (t.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
            frames_np = t
        else:
            # Assume list of PIL Images
            import numpy as np
            frames_np = [
                (f if isinstance(f, (bytes, bytearray)) else
                 (f if hasattr(f, "shape") else
                  (lambda im: np.array(im.convert("RGB")))(f)))
                for f in frames
            ]

        mp4 = _frames_to_mp4(frames_np, fps=fps)
        b64 = base64.b64encode(mp4).decode("ascii")
        return {
            "video": f"data:video/mp4;base64,{b64}",
            "width": width,
            "height": height,
            "fps": fps,
            "frames": num_frames,
            "seed_used": seed,
            "size_bytes": len(mp4),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-3000:],
        }


runpod.serverless.start({"handler": handler})
