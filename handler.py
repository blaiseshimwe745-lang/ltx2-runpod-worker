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

sys.path.insert(0, "/app")

print("[boot] checking models on volume…")
try:
    subprocess.check_call([sys.executable, "/app/download_models.py"])
except Exception as e:
    print(f"[boot] download_models.py failed: {e}", file=sys.stderr)

_pipe = None
_load_error = None

def _load_pipeline():
    global _pipe, _load_error
    if _pipe is not None or _load_error is not None:
        return
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        from diffusers import WanAnimatePipeline
        
        volume = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
        wan_dir = volume / "wan2-animate-14b-diffusers"

        print("[boot] Loading WanAnimatePipeline…")
        
        _pipe = WanAnimatePipeline.from_pretrained(
            str(wan_dir),
            torch_dtype=torch.bfloat16
        )
        
        # Save memory on 96GB GPU
        _pipe.enable_model_cpu_offload()
        
        print("[boot] Pipeline ready.")
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        print(f"[boot] Pipeline load failed: {_load_error}", file=sys.stderr)

def _decode_input_image(s, tmp_path):
    if not s:
        return None
    from PIL import Image
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    else:
        return None
    im.save(tmp_path)
    return tmp_path

def _decode_input_video(s, tmp_path):
    if not s:
        return None
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return tmp_path
    return None

def handler(event):
    image_tmp = None
    pose_tmp = None
    output_tmp = None
    try:
        inp = (event or {}).get("input", {}) or {}

        if inp.get("diagnostic"):
            import download_models as dm
            return dm.diagnose()

        prompt = (inp.get("prompt") or "").strip()
        mode = inp.get("mode", "animate") # "animate" or "replace"
        
        if _pipe is None:
            _load_pipeline()
        if _load_error:
            return {"error": "pipeline_load_failed", "detail": _load_error}

        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        image_input_b64 = inp.get("image")
        pose_input_b64 = inp.get("pose_video")
        
        if not image_input_b64 or not pose_input_b64:
            return {"error": "Both 'image' and 'pose_video' are required for Wan-Animate"}

        from diffusers.utils import load_image, load_video, export_to_video
        import torch

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image_tmp = f.name
        _decode_input_image(image_input_b64, image_tmp)
        ref_image = load_image(image_tmp)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            pose_tmp = f.name
        _decode_input_video(pose_input_b64, pose_tmp)
        pose_video = load_video(pose_tmp)

        print(f"[run] mode={mode} seed={seed} prompt={prompt}")
        
        torch.cuda.empty_cache()
        
        # We don't use inference_mode() explicitly here because diffusers pipelines usually handle it correctly
        video_frames = _pipe(
            image=ref_image,
            pose_video=pose_video,
            face_video=pose_video, # Fallback face_video to pose_video
            prompt=prompt,
            mode=mode,
            segment_frame_length=77,
            prev_segment_conditioning_frames=1,
            guidance_scale=1.0,
            num_inference_steps=20,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_tmp = f.name
            
        export_to_video(video_frames, output_tmp, fps=24)
        
        mp4 = Path(output_tmp).read_bytes()
        b64 = base64.b64encode(mp4).decode("ascii")
        
        return {
            "video": f"data:video/mp4;base64,{b64}",
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
        if pose_tmp:
            try: os.unlink(pose_tmp)
            except Exception: pass
        if output_tmp:
            try: os.unlink(output_tmp)
            except Exception: pass

runpod.serverless.start({"handler": handler})
