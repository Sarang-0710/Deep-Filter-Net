"""
DeepFilterNet audio denoising backend — ONNX Runtime edition.

Startup: loads the ONNX model (deepfilternet3_mask_only.onnx) once.
POST /api/process: accepts an audio file, resamples to 48 kHz mono,
                   runs OnnxEnhancer.enhance(), streams back the cleaned wav.
GET  /api/health : simple health check.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from onnx_enhancer import OnnxEnhancer
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Allowed audio extensions
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# DeepFilterNet expects 48 kHz mono
TARGET_SR = 48_000


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model at startup; store in app.state."""
    print("Loading DeepFilterNet3 ONNX model …")
    app.state.onnx_model = OnnxEnhancer(model_dir="onnx_model")
    print("Model ready.")
    yield
    # Nothing to clean up — model lives in RAM until process exits


app = FastAPI(title="DeepFilterNet Audio Denoiser", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server and any localhost origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: resample + convert to mono at 48 kHz
# ---------------------------------------------------------------------------
def prepare_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono float32, resample to TARGET_SR.
    Returns (waveform [T], sample_rate).
    """
    data, sr = sf.read(path, always_2d=True)  # [T, C] float64
    data = data.mean(axis=1).astype(np.float32)  # mono, float32

    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        up, down = TARGET_SR // g, sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    return data, TARGET_SR


# ---------------------------------------------------------------------------
# Helper: delete a list of file paths (used as a BackgroundTask)
# ---------------------------------------------------------------------------
def cleanup_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass  # Best-effort cleanup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # --- Validate extension ---
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    input_tmp = None
    output_tmp = None

    try:
        # --- Save upload to temp file ---
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as f_in:
            shutil.copyfileobj(file.file, f_in)
            input_tmp = f_in.name

        # --- Prepare audio (resample + mono) ---
        waveform, sr = prepare_audio(input_tmp)   # numpy [T] float32, 48 kHz

        # --- Run ONNX inference ---
        onnx_model = app.state.onnx_model
        enhanced_np = onnx_model.enhance(waveform)  # numpy [T] float32

        # --- Write result to temp wav file ---
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as f_out:
            output_tmp = f_out.name

        sf.write(output_tmp, enhanced_np, sr, subtype="PCM_16")

        # Schedule cleanup AFTER the response is sent
        background_tasks.add_task(cleanup_files, input_tmp, output_tmp)

        return FileResponse(
            path=output_tmp,
            media_type="audio/wav",
            filename="denoised.wav",
        )

    except HTTPException:
        # Clean up immediately on known errors
        cleanup_files(input_tmp, output_tmp)
        raise
    except Exception as exc:
        cleanup_files(input_tmp, output_tmp)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
