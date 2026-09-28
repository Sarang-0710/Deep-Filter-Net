"""
DeepFilterNet audio denoising backend — ONNX Runtime edition.

Endpoints:
  POST /api/process      — full-file upload → denoised WAV download
  WS   /api/stream       — streaming: send raw Float32 PCM frames, receive denoised frames
  GET  /api/health       — health check
"""

import asyncio
import io
import os
import shutil
import struct
import tempfile
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from onnx_enhancer import OnnxEnhancer, StreamingEnhancer, ParallelOnnxEnhancer
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TARGET_SR          = 48_000   # DeepFilterNet expects 48 kHz mono

# WebSocket protocol constants
# Client sends: 4-byte little-endian uint32 (num_samples) + num_samples * 4 bytes float32
# Server sends: same format with denoised samples, OR a 4-byte 0xFFFFFFFF sentinel (done)
WS_DONE_SENTINEL = struct.pack("<I", 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX models at startup; store in app.state."""
    print("Loading DeepFilterNet3 ONNX model …")
    app.state.onnx_model = OnnxEnhancer(model_dir="onnx_model")
    app.state.parallel_model = ParallelOnnxEnhancer(model_dir="onnx_model", num_workers=4)
    print("Model ready.")
    yield


app = FastAPI(title="DeepFilterNet Audio Denoiser", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file → mono float32 @ 48 kHz."""
    data, sr = sf.read(path, always_2d=True)
    data = data.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    return data, TARGET_SR


def cleanup_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def encode_frame(samples: np.ndarray) -> bytes:
    """Pack float32 array as: [uint32 count][float32 * count]"""
    arr = samples.astype(np.float32)
    header = struct.pack("<I", len(arr))
    return header + arr.tobytes()


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
    """Full-file upload → denoised WAV download (kept for backwards compat)."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    input_tmp = output_tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
            shutil.copyfileobj(file.file, f_in)
            input_tmp = f_in.name

        waveform, sr = prepare_audio(input_tmp)
        
        # Use parallel processing for fast execution on large files
        enhanced_np = await asyncio.to_thread(app.state.parallel_model.process, waveform, sr)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_out:
            output_tmp = f_out.name
        sf.write(output_tmp, enhanced_np, sr, subtype="PCM_16")

        background_tasks.add_task(cleanup_files, input_tmp, output_tmp)
        return FileResponse(path=output_tmp, media_type="audio/wav", filename="denoised.wav")

    except HTTPException:
        cleanup_files(input_tmp, output_tmp)
        raise
    except Exception as exc:
        cleanup_files(input_tmp, output_tmp)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


@app.websocket("/api/stream")
async def stream_audio(ws: WebSocket):
    """
    WebSocket streaming endpoint.

    Protocol (binary frames only):
      CLIENT → SERVER:
        • Each message = [uint32 n_samples LE][float32 * n_samples]  — raw PCM chunk @ 48 kHz mono
        • Special: n_samples == 0xFFFFFFFF  →  end-of-stream signal

      SERVER → CLIENT:
        • Each message = [uint32 n_samples LE][float32 * n_samples]  — denoised PCM
        • Final message = [uint32 0xFFFFFFFF]                        — done sentinel

    The client is responsible for:
      1. Decoding its audio file with AudioContext.decodeAudioData (→ 48 kHz mono if possible)
      2. Sending the first chunk as a prewarm frame (server discards it from output)
      3. Sending subsequent chunks and playing received denoised frames via Web Audio API
    """
    await ws.accept()

    enhancer = StreamingEnhancer(model_dir="onnx_model", context_sec=1.0)
    prewarm_done = False

    try:
        while True:
            raw = await ws.receive_bytes()

            if len(raw) < 4:
                break

            n_samples = struct.unpack_from("<I", raw, 0)[0]

            # End-of-stream sentinel
            if n_samples == 0xFFFFFFFF:
                # Flush remaining tail
                tail = await asyncio.to_thread(enhancer.flush)
                if len(tail) > 0:
                    await ws.send_bytes(encode_frame(tail))
                await ws.send_bytes(WS_DONE_SENTINEL)
                break

            # Decode PCM payload
            expected_bytes = 4 + n_samples * 4
            if len(raw) < expected_bytes:
                await ws.send_bytes(WS_DONE_SENTINEL)
                break

            chunk = np.frombuffer(raw[4:4 + n_samples * 4], dtype=np.float32).copy()

            if not prewarm_done:
                # First chunk is used as prewarm context only — no output sent
                await asyncio.to_thread(enhancer.prewarm, chunk)
                prewarm_done = True
                # Acknowledge prewarm received (send 0-sample frame)
                await ws.send_bytes(encode_frame(np.zeros(0, dtype=np.float32)))
                continue

            # Process chunk in a thread so the event loop stays responsive
            denoised = await asyncio.to_thread(enhancer.push, chunk)

            if len(denoised) > 0:
                await ws.send_bytes(encode_frame(denoised))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[stream] error: {exc}")
        try:
            await ws.send_bytes(WS_DONE_SENTINEL)
        except Exception:
            pass
