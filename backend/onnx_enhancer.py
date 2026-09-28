"""
ONNX Runtime inference engine for DeepFilterNet3.
Replaces PyTorch entirely at inference time.

This version uses the 3-file export (enc.onnx, erb_dec.onnx, df_dec.onnx)
and manually implements the feature padding (pad_feat) and deep filter
application (unfold + einsum) using NumPy to achieve identical quality
to the full PyTorch model.

Pipeline:
  audio (float32, 48kHz, mono)
      │
      ├─ libdf.analysis()  ──────────────────────────────────── STFT
      │     └─ spec: [C, T, F] complex64
      │
      ├─ feature extraction (libdf Rust builtins)
      │     ├─ feat_erb:  [1, 1, T, nb_erb]  float32
      │     └─ feat_spec: [1, 2, T, nb_df]   float32  (real+imag channels)
      │
      ├─ pad_feat() ─── time-shifts features by 2 frames (lookahead)
      │
      ├─ enc.onnx    → e0, e1, e2, e3, emb, c0, lsnr
      ├─ erb_dec.onnx → m  (ERB mask)
      ├─ df_dec.onnx  → coefs  (DF filter coefficients)
      │
      ├─ apply ERB mask  (NumPy matmul)
      ├─ apply deep filter (NumPy unfold + einsum)
      │
      └─ libdf.synthesis() ──────────────────────────────────── iSTFT
            └─ enhanced audio: [C, T]
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from libdf import DF, erb, erb_norm, unit_norm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


def _spec_unfold(spec: np.ndarray, frame_size: int, lookahead: int) -> np.ndarray:
    """
    Pad and unfold spectrogram for multi-frame filtering.
    """
    pad_before = frame_size - 1 - lookahead
    pad_after  = lookahead
    # Pad along the time axis (axis=1)
    spec_pad = np.pad(spec, ((0, 0), (pad_before, pad_after), (0, 0)), mode="constant")

    C, T_pad, F = spec_pad.shape
    T = T_pad - frame_size + 1

    # Strided unfold along axis 1 → [C, T, F, frame_size]
    strides = spec_pad.strides
    shape   = (C, T, F, frame_size)
    new_strides = (strides[0], strides[1], strides[2], strides[1])
    return np.lib.stride_tricks.as_strided(spec_pad, shape=shape, strides=new_strides)


def _apply_df(spec: np.ndarray, coefs: np.ndarray,
              nb_df: int, df_order: int, df_lookahead: int) -> np.ndarray:
    """
    Apply the deep filter to the first nb_df frequency bins.
    """
    B = coefs.shape[0]
    C, T, F = spec.shape

    # 1. Reshape coefs: [B, T, nb_df, df_order*2] → [B, T, nb_df, df_order, 2]
    #    then make complex: [B, T, nb_df, df_order]
    coefs_ri = coefs.reshape(B, T, nb_df, df_order, 2)
    coefs_c  = coefs_ri[..., 0] + 1j * coefs_ri[..., 1]          # [B, T, nb_df, df_order]

    # 2. Permute to match einsum shape [B, df_order, T, nb_df]  (== [B, C_df, N, T, F] collapsed)
    coefs_c = coefs_c.transpose(0, 3, 1, 2)                       # [B, df_order, T, nb_df]

    # 3. Unfold spec over time: [C, T, nb_df, df_order]
    spec_slice = spec[:, :, :nb_df]                                # [C, T, nb_df]
    spec_u     = _spec_unfold(spec_slice, df_order, df_lookahead)  # [C, T, nb_df, df_order]

    # 4. einsum: "...tfn,...ntf->...tf" i.e. sum over filter taps
    spec_f = np.einsum("ctfn,bntf->ctf", spec_u, coefs_c)         # [C, T, nb_df] complex

    # 5. Write filtered bins back
    spec_e = spec.copy()
    spec_e[:, :, :nb_df] = spec_f
    return spec_e


def pad_feat(x: np.ndarray, pad_frames: int = 2) -> np.ndarray:
    """
    Matches PyTorch ConstantPad2d((0,0,-2,2)) used in DeepFilterNet.
    Shifts the time axis (axis 2) forward by `pad_frames`.
    """
    T = x.shape[2]
    out = np.zeros_like(x)
    if T > pad_frames:
        out[:, :, :T-pad_frames, :] = x[:, :, pad_frames:, :]
    return out


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class OnnxEnhancer:
    """
    Drop-in replacement for df.enhance using ONNX Runtime.
    Restores the full Deep Filter (DF) logic for maximum quality.
    """

    def __init__(self, model_dir: str | os.PathLike = "onnx_model"):
        model_dir = Path(model_dir)

        # ── ONNX sessions ────────────────────────────────────────────────────
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _sess(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(model_dir / name),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self._enc     = _sess("enc.onnx")
        self._erb_dec = _sess("erb_dec.onnx")
        self._df_dec  = _sess("df_dec.onnx")

        # ── Pre-computed filterbank & params ─────────────────────────────────
        self._erb_inv_fb: np.ndarray = np.load(model_dir / "erb_inv_fb.npy")   # [nb_erb, F]
        self._alpha: float           = float(np.load(model_dir / "norm_alpha.npy")[0])
        params                       = np.load(model_dir / "params.npy", allow_pickle=True).item()

        self.sr          = int(params["sr"])
        self.fft_size    = int(params["fft_size"])
        self.hop_size    = int(params["hop_size"])
        self.nb_erb      = int(params["nb_erb"])
        self.nb_df       = int(params["nb_df"])
        self.df_order    = int(params.get("df_order", 5))
        self.df_lookahead= int(params.get("df_lookahead", 2))

        # ── libdf DF state (handles STFT) ────────────────────────────────────
        self._df_state = DF(
            sr        = self.sr,
            fft_size  = self.fft_size,
            hop_size  = self.hop_size,
            nb_bands  = self.nb_erb,
            min_nb_erb_freqs = 2,
        )
        self._erb_fb: np.ndarray = self._df_state.erb_widths()   # [nb_erb] uint64

        print(
            f"OnnxEnhancer (Full DF Model) ready | sr={self.sr} fft={self.fft_size} "
            f"nb_df={self.nb_df} df_order={self.df_order}"
        )

    # ── Public API ────────────────────────────────────────────────────────── #

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise a mono audio array.
        """
        audio = _to_float32(audio)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]   # [1, T]

        orig_len = audio.shape[-1]

        # Pad with fft_size zeros (same as PyTorch enhance(pad=True))
        # This ensures the STFT captures the last few frames correctly.
        pad = np.zeros((audio.shape[0], self.fft_size), dtype=np.float32)
        audio_padded = np.concatenate([audio, pad], axis=1)

        # 1. STFT via libdf
        spec = self._df_state.analysis(audio_padded)         # [C, T_frames, F] complex64

        # 2. Feature extraction
        feat_erb_np  = erb_norm(erb(spec, self._erb_fb), self._alpha)     # [C, T_frames, nb_erb] f32
        feat_spec_c  = unit_norm(spec[..., :self.nb_df], self._alpha)     # [C, T_frames, nb_df] c64

        # Shape: ONNX enc expects [B, 1, T, nb_erb] and [B, 2, T, nb_df]
        feat_erb_in  = feat_erb_np[np.newaxis, :]                         # [1, C, T, nb_erb]
        feat_spec_ri = np.stack(
            [feat_spec_c.real, feat_spec_c.imag], axis=1
        ).astype(np.float32)                                              # [C, 2, T, nb_df]
        feat_spec_in = feat_spec_ri                                       # [C, 2, T, nb_df] (C=1)

        # Apply DeepFilterNet pad_feat (time-shift)
        feat_erb_in  = pad_feat(feat_erb_in)
        feat_spec_in = pad_feat(feat_spec_in)

        # 3. Run Encoder
        enc_out = self._enc.run(
            None,
            {"feat_erb": feat_erb_in, "feat_spec": feat_spec_in},
        )
        e0, e1, e2, e3, emb, c0, lsnr = enc_out

        # 4. Run ERB Decoder → mask
        m = self._erb_dec.run(
            None,
            {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0},
        )[0]

        # 5. Run DF Decoder → coefs
        coefs = self._df_dec.run(
            None,
            {"emb": emb, "c0": c0},
        )[0]

        # 6. Apply ERB mask
        mask_sq  = m.squeeze(0).squeeze(0)                             # [T, nb_erb]
        mask_lin = mask_sq @ self._erb_inv_fb                          # [T, F] float32
        spec_m = spec * mask_lin[np.newaxis, :, :]                     # [C, T, F] complex64

        # 7. Apply deep filter to first nb_df bins
        spec_e = _apply_df(
            spec_m, coefs,
            self.nb_df, self.df_order, self.df_lookahead,
        )

        # 8. iSTFT via libdf
        enhanced_audio = self._df_state.synthesis(spec_e)              # [C, T_audio]

        # Trim: same delay compensation as PyTorch's enhance(pad=True):
        d = self.fft_size - self.hop_size
        return enhanced_audio[0, d : orig_len + d].astype(np.float32)


# --------------------------------------------------------------------------- #
# Streaming enhancer  (overlapping context-window strategy)
# --------------------------------------------------------------------------- #

class StreamingEnhancer:
    """
    Stateful streaming wrapper around the ONNX inference pipeline.

    Maintains a rolling context buffer so each ONNX inference call receives
    `context_sec` seconds of warm-up audio before the new chunk.  This
    compensates for the GRU hidden states being reset on every ONNX call,
    which would otherwise cause audible glitches / quality degradation at
    chunk boundaries.

    Usage
    -----
        se = StreamingEnhancer("onnx_model")
        for raw_chunk in incoming_chunks:
            denoised_chunk = se.push(raw_chunk)
        last_block = se.flush()           # drain leftover samples at stream end
    """

    def __init__(
        self,
        model_dir="onnx_model",
        context_sec: float = 1.0,
    ):
        model_dir = Path(model_dir)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _sess(name):
            return ort.InferenceSession(
                str(model_dir / name),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

        self._enc     = _sess("enc.onnx")
        self._erb_dec = _sess("erb_dec.onnx")
        self._df_dec  = _sess("df_dec.onnx")

        self._erb_inv_fb = np.load(model_dir / "erb_inv_fb.npy")
        self._alpha      = float(np.load(model_dir / "norm_alpha.npy")[0])
        params           = np.load(model_dir / "params.npy", allow_pickle=True).item()

        self.sr           = int(params["sr"])
        self.fft_size     = int(params["fft_size"])
        self.hop_size     = int(params["hop_size"])
        self.nb_erb       = int(params["nb_erb"])
        self.nb_df        = int(params["nb_df"])
        self.df_order     = int(params.get("df_order", 5))
        self.df_lookahead = int(params.get("df_lookahead", 2))

        self._ctx_samples = int(context_sec * self.sr)
        self._ctx_buf     = np.zeros(self._ctx_samples, dtype=np.float32)
        self._tail        = np.zeros(0, dtype=np.float32)
        # How many raw audio samples have been fed so far (for warmup tracking)
        self._samples_fed: int = 0

        _tmp = DF(sr=self.sr, fft_size=self.fft_size, hop_size=self.hop_size,
                  nb_bands=self.nb_erb, min_nb_erb_freqs=2)
        self._erb_fb = _tmp.erb_widths()

        print(
            f"StreamingEnhancer ready | sr={self.sr}  context={context_sec:.1f}s  "
            f"df_order={self.df_order}"
        )

    # ── Internal ────────────────────────────────────────────────────────────

    def _run_window(self, window: np.ndarray) -> np.ndarray:
        """Run the full pipeline on a 1-D window. Returns raw synthesis output."""
        audio_2d = window[np.newaxis, :]
        df_state = DF(
            sr=self.sr, fft_size=self.fft_size, hop_size=self.hop_size,
            nb_bands=self.nb_erb, min_nb_erb_freqs=2,
        )
        spec        = df_state.analysis(audio_2d)
        feat_erb_np = erb_norm(erb(spec, self._erb_fb), self._alpha)
        feat_spec_c = unit_norm(spec[..., :self.nb_df], self._alpha)

        feat_erb_in  = pad_feat(feat_erb_np[np.newaxis, :])
        feat_spec_ri = np.stack(
            [feat_spec_c.real, feat_spec_c.imag], axis=1
        ).astype(np.float32)
        feat_spec_in = pad_feat(feat_spec_ri)

        e0, e1, e2, e3, emb, c0, lsnr = self._enc.run(
            None, {"feat_erb": feat_erb_in, "feat_spec": feat_spec_in}
        )
        m     = self._erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0]
        coefs = self._df_dec.run(None, {"emb": emb, "c0": c0})[0]

        mask_lin = m.squeeze(0).squeeze(0) @ self._erb_inv_fb
        spec_m   = spec * mask_lin[np.newaxis, :, :]
        spec_e   = _apply_df(spec_m, coefs, self.nb_df, self.df_order, self.df_lookahead)

        return df_state.synthesis(spec_e)[0]   # [T_audio]

    # ── Public API ──────────────────────────────────────────────────────────

    def prewarm(self, audio: np.ndarray) -> None:
        """
        Feed up to `context_sec` seconds of audio as warm-up context.
        This audio will NOT appear in the output stream — it is used only
        to give the model meaningful history before the first real chunk.
        Call this before the first push() for best quality on early chunks.

        Example (when you have the whole file ahead of time):
            se.prewarm(audio[:sr])   # first 1 second as context
            for chunk in chunks:
                out = se.push(chunk)
        """
        audio = _to_float32(audio)
        if audio.ndim != 1:
            raise ValueError(f"prewarm() expects 1-D audio, got shape {audio.shape}")
        # Take only the last ctx_samples worth of audio
        self._ctx_buf      = np.concatenate([self._ctx_buf, audio])[-self._ctx_samples:]
        self._samples_fed += len(audio)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """
        Feed the next raw audio chunk (float32, 48 kHz, mono).
        Returns the denoised audio for the samples that were processed.
        Chunks smaller than one hop_size are buffered and return empty array.
        """
        chunk = _to_float32(chunk)
        if chunk.ndim != 1:
            raise ValueError(f"push() expects 1-D audio, got shape {chunk.shape}")

        chunk  = np.concatenate([self._tail, chunk])
        n_hops = len(chunk) // self.hop_size
        self._tail = chunk[n_hops * self.hop_size:]
        chunk      = chunk[:n_hops * self.hop_size]

        if n_hops == 0:
            return np.zeros(0, dtype=np.float32)

        chunk_len = len(chunk)
        window    = np.concatenate([
            self._ctx_buf,
            chunk,
            np.zeros(self.fft_size, dtype=np.float32),
        ])

        enh = self._run_window(window)

        # d = algorithmic delay (fft_size - hop_size = 480 samples)
        d         = self.fft_size - self.hop_size

        # If we don't yet have a full context worth of real audio, the GRU
        # warm-up will be partial.  We still extract at the correct offset,
        # which avoids hard alignment bugs, but quality will be slightly lower
        # for the first ~(ctx_samples / hop_size) chunks.
        ctx_actual = min(self._ctx_samples, self._samples_fed)
        out_start  = ctx_actual + d
        out_end    = out_start + chunk_len

        if out_end > len(enh):
            enh = np.pad(enh, (0, out_end - len(enh)))

        result = enh[out_start:out_end].astype(np.float32)

        # Roll context buffer and update sample counter
        self._ctx_buf      = np.concatenate([self._ctx_buf, chunk])[-self._ctx_samples:]
        self._samples_fed += chunk_len
        return result

    def flush(self) -> np.ndarray:
        """
        Process any remaining tail samples and reset state.
        Call once after the last push() at end-of-stream.
        """
        result = np.zeros(0, dtype=np.float32)
        if len(self._tail) > 0:
            result = self.push(self._tail)
        self._tail    = np.zeros(0, dtype=np.float32)
        self._ctx_buf = np.zeros(self._ctx_samples, dtype=np.float32)
        return result

# ---------------------------------------------------------------------------
# Parallel Enhancer (for fast full-file processing)
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor

class ParallelOnnxEnhancer:
    """
    Splits long audio files into chunks and processes them in parallel across multiple 
    CPU threads using independent ONNX instances. This is ideal for getting the full 
    denoised audio in < 1 second for downstream tasks like STT.
    """
    def __init__(self, model_dir: str, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = []
        
        # Initialize multiple ONNX instances for parallel processing
        # Important: Set threads to 1 per instance so they don't fight for CPU cores
        for _ in range(num_workers):
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            enhancer = OnnxEnhancer(model_dir=model_dir)
            
            # Re-initialize the ONNX sessions on this specific enhancer with the single-thread options
            def _sess(name: str):
                return ort.InferenceSession(
                    str(Path(model_dir) / name), 
                    sess_options=opts, 
                    providers=["CPUExecutionProvider"]
                )
            enhancer._enc     = _sess("enc.onnx")
            enhancer._erb_dec = _sess("erb_dec.onnx")
            enhancer._df_dec  = _sess("df_dec.onnx")
            
            self.workers.append(enhancer)
            
    def process(self, audio_data: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        total_len = len(audio_data)
        if total_len == 0:
            return audio_data
            
        chunk_size = total_len // self.num_workers
        if chunk_size == 0:
            return self.workers[0].enhance(audio_data)
            
        # DeepFilterNet needs 1 second of context
        ctx_samples = sample_rate * 1  
        
        def process_chunk(worker_idx):
            start_idx = worker_idx * chunk_size
            end_idx = start_idx + chunk_size if worker_idx < self.num_workers - 1 else total_len
            
            # Prepend 1 second of context to avoid glitches at boundaries
            process_start = max(0, start_idx - ctx_samples)
            actual_context_added = start_idx - process_start
            
            chunk_audio = audio_data[process_start:end_idx]
            
            # Denoise using this specific thread's ONNX instance
            enhanced = self.workers[worker_idx].enhance(chunk_audio)
            
            # Remove context from the output
            if actual_context_added > 0:
                return enhanced[actual_context_added:]
            return enhanced

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            results = list(pool.map(process_chunk, range(self.num_workers)))
            
        return np.concatenate(results)
