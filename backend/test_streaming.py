"""
Offline validation: StreamingEnhancer quality and latency.

Quality test: each streamed chunk is compared against a standalone full-file
enhancement of the SAME window (context + chunk), which is the true reference.
This tells us how much quality we lose purely from chunking artifacts.

Latency test: measures first-chunk and median per-chunk processing time.
"""
import time
import numpy as np
from onnx_enhancer import OnnxEnhancer, StreamingEnhancer


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    proj  = alpha * ref
    noise = est - proj
    return 10 * np.log10((np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8))


np.random.seed(42)
SR       = 48_000
DURATION = 30
CHUNK_S  = 0.5
CTX_S    = 1.0

audio    = np.random.randn(SR * DURATION).astype(np.float32)
chunk_sz = int(CHUNK_S * SR)    # 24000 samples
ctx_sz   = int(CTX_S  * SR)     # 48000 samples

print(f"Audio: {DURATION}s @ {SR} Hz  |  chunk: {CHUNK_S}s  context: {CTX_S}s")
print("=" * 60)

# ── Streaming ────────────────────────────────────────────────────────────────
print(f"\n[1] StreamingEnhancer (context={CTX_S}s, chunk={CHUNK_S}s)…")
se        = StreamingEnhancer("onnx_model", context_sec=CTX_S)
se.prewarm(audio[:ctx_sz])                   # seed context with first 1s

chunks    = [audio[ctx_sz + i : ctx_sz + i + chunk_sz]
             for i in range(0, DURATION * SR - ctx_sz, chunk_sz)]
stream_out_chunks = []
latencies = []

for c in chunks:
    if len(c) == 0:
        break
    t0  = time.perf_counter()
    out = se.push(c)
    latencies.append(time.perf_counter() - t0)
    if len(out):
        stream_out_chunks.append(out)

tail = se.flush()
if len(tail):
    stream_out_chunks.append(tail)

# ── Reference: full-file enhancement of each window ─────────────────────────
print(f"\n[2] Building per-chunk full-file reference…")
full_enh = OnnxEnhancer("onnx_model")
ref_chunks = []

for i, c in enumerate(chunks):
    if len(c) == 0:
        break
    # The true reference for chunk i is: enhance(context_for_i + c)
    # where context_for_i = audio[i*chunk_sz : i*chunk_sz + ctx_sz]
    ctx_start = i * chunk_sz          # shifts with each chunk
    ctx_end   = ctx_start + ctx_sz
    window    = np.concatenate([audio[ctx_start:ctx_end], c])
    enh_window = full_enh.enhance(window)
    # The chunk portion is the last chunk_sz samples of enh_window
    ref_chunk = enh_window[-len(c):]
    ref_chunks.append(ref_chunk)

# ── Metrics ──────────────────────────────────────────────────────────────────
print(f"\n[3] Per-chunk quality (streaming vs. reference)")
corrs, sdrs = [], []
for i, (s, r) in enumerate(zip(stream_out_chunks, ref_chunks)):
    m = min(len(s), len(r))
    c = np.corrcoef(s[:m], r[:m])[0, 1]
    sdr_val = si_sdr(r[:m], s[:m])
    corrs.append(c)
    sdrs.append(sdr_val)
    if i < 5 or i >= len(stream_out_chunks) - 2:
        print(f"    chunk {i:2d}: corr={c:.4f}  SI-SDR={sdr_val:.1f} dB")

print(f"\n    Median corr  : {np.median(corrs):.4f}  (target > 0.95)")
print(f"    Median SI-SDR: {np.median(sdrs):.1f} dB  (target > 20 dB)")
print(f"    Min corr     : {min(corrs):.4f}")

print(f"\n[4] Latency")
print(f"    First-chunk   : {latencies[0]*1000:.1f} ms")
print(f"    Median chunk  : {np.median(latencies)*1000:.1f} ms")
print(f"    Max chunk     : {max(latencies)*1000:.1f} ms")
print(f"    → User hears audio in ≈ {latencies[0]*1000 + CHUNK_S*1000:.0f} ms "
      f"after clicking 'process'")
