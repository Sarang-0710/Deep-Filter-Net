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
