"""
DeepFilterNet audio denoising backend.

Startup: loads the DeepFilterNet3 model once via init_df().
POST /api/process: accepts an audio file, resamples to 48 kHz mono,
                   runs enhance(), and streams back the cleaned wav.
GET  /api/health : simple health check.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

# Tune CPU threads for maximum speed (4 threads is the sweet spot from benchmarks)
torch.set_num_threads(4)
from df.enhance import enhance, init_df
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
    """Load DeepFilterNet3 model at startup; store in app.state."""
    print("Loading DeepFilterNet3 model …")
    model, df_state, _ = init_df()
    model.eval()
    app.state.model = model
    app.state.df_state = df_state
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
def prepare_audio(path: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file, convert to mono, resample to TARGET_SR.
    Uses soundfile to avoid torchaudio 2.11's TorchCodec dependency.
    Returns (waveform [1, T], sample_rate).
    """
    import numpy as np

    data, sr = sf.read(path, always_2d=True)  # shape: [T, C], float64
    data = data.T  # shape: [C, T]
    waveform = torch.from_numpy(data).float()

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed using torchaudio.functional (doesn't need TorchCodec)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=TARGET_SR)

    return waveform, TARGET_SR


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
        waveform, sr = prepare_audio(input_tmp)

        # --- Run DeepFilterNet inference ---
        model = app.state.model
        df_state = app.state.df_state

        with torch.no_grad():
            enhanced = enhance(model, df_state, waveform)

        # --- Write result to temp wav file ---
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as f_out:
            output_tmp = f_out.name

        # enhanced shape: [C, T] or [T] — flatten to [T] for soundfile
        audio_np = enhanced.cpu().squeeze().numpy()  # [T] or [C, T]
        if audio_np.ndim == 2:
            audio_np = audio_np.T  # soundfile wants [T, C]

        # Use soundfile to write WAV — avoids torchaudio 2.11 TorchCodec dependency
        sf.write(output_tmp, audio_np, sr, subtype="PCM_16")

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
