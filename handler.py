"""
RunPod Serverless worker for LTX-2.3 (DistilledPipeline).
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

sys.path.insert(0, "/app")  # so `import download_models` works in diagnostic mode

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
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        import torch  # noqa: F401
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.empty_cache()
        
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
        
        volume = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
        ltx_dir = volume / "ltx-2.3"
        gemma_dir = volume / "gemma-3"

        print("[boot] Loading DistilledPipeline…")
        
        loras = [
            LoraPathStrengthAndSDOps(
                path=str(ltx_dir / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP
            )
        ]
        
        _pipe = DistilledPipeline(
            distilled_checkpoint_path=str(ltx_dir / "ltx-2.3-22b-distilled-1.1.safetensors"),
            gemma_root=str(gemma_dir),
            spatial_upsampler_path=str(ltx_dir / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            loras=loras,
            device=torch.device("cuda"),
        )
        print("[boot] Pipeline ready.")
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        print(f"[boot] Pipeline load failed: {_load_error}", file=sys.stderr)


def _round_dim(x: int) -> int:
    return max(256, (int(x) // 32) * 32)


def _round_frames(n: int) -> int:
    # Must be 8N + 1
    return max(9, ((int(n) - 1) // 8) * 8 + 1)


def _decode_input_image(s, tmp_path):
    if not s:
        return None
    from PIL import Image
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    elif s.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(s, timeout=30) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGB")
    else:
        return None
    im.save(tmp_path)
    return tmp_path


def handler(event):
    image_tmp = None
    output_tmp = None
    try:
        inp = (event or {}).get("input", {}) or {}

        # --- Diagnostic / repair mode: inspect the network volume directly ---
        if inp.get("diagnostic"):
            import download_models as dm
            info = dm.diagnose()
            if inp.get("repair"):
                # Force a clean Gemma 12B re-download and report the result
                try:
                    dm.main(force_gemma=True)
                    info["repair"] = "ok"
                    info["gemma_hidden_size_after"] = dm.diagnose().get("gemma_hidden_size")
                except SystemExit as se:
                    info["repair"] = f"exit {se.code}"
                except Exception as e:
                    info["repair"] = f"error: {e}"
            return info

        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "missing 'prompt'"}

        # Lazy-load pipeline
        if _pipe is None:
            _load_pipeline()
        if _load_error:
            return {"error": "pipeline_load_failed", "detail": _load_error}

        width = _round_dim(inp.get("width", 768))
        height = _round_dim(inp.get("height", 512))
        fps = int(inp.get("fps", 24))
        duration = float(inp.get("duration_seconds", 5.0))
        num_frames = _round_frames(int(duration * fps))
        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        call_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=float(fps),
            seed=seed,
            images=[],
        )
        
        image_input_b64 = inp.get("image")
        if image_input_b64:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image_tmp = f.name
            
            img_path = _decode_input_image(image_input_b64, image_tmp)
            if img_path:
                from ltx_pipelines.utils.args import ImageConditioningInput
                call_kwargs["images"] = [
                    ImageConditioningInput(
                        path=img_path,
                        frame_idx=0,
                        strength=1.0,
                    )
                ]
            else:
                try: os.unlink(image_tmp)
                except Exception: pass
                image_tmp = None

        if inp.get("enhance_prompt"):
            call_kwargs["enhance_prompt"] = True

        print(f"[run] {width}x{height} f{num_frames} fps{fps} seed={seed}")
        
        import torch
        torch.cuda.empty_cache()
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video_iter, audio = _pipe(**call_kwargs)

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    output_tmp = f.name
                    
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
                
                tiling_config = getattr(_pipe, "tiling_config", None) or TilingConfig.default()
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
                
                encode_video(
                    video=video_iter,
                    fps=fps,
                    audio=audio,
                    output_path=output_tmp,
                    video_chunks_number=video_chunks_number,
                )
        
        mp4 = Path(output_tmp).read_bytes()
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
    finally:
        if image_tmp:
            try: os.unlink(image_tmp)
            except Exception: pass
        if output_tmp:
            try: os.unlink(output_tmp)
            except Exception: pass


runpod.serverless.start({"handler": handler})
