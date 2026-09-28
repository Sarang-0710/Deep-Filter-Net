"""
One-time script to export DeepFilterNet3 PyTorch model → 3 ONNX files.
Run this once: python3 export_onnx.py
Outputs: onnx_model/enc.onnx, onnx_model/erb_dec.onnx, onnx_model/df_dec.onnx
"""
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort
import torch
from df.enhance import ModelParams, df_features, init_df

OUT_DIR = Path("onnx_model")
OUT_DIR.mkdir(exist_ok=True)


def export_submodel(path, model, inputs, input_names, output_names, dynamic_axes):
    print(f"  Exporting {path.name} ...")

    with torch.no_grad():
        torch.onnx.export(
            model=deepcopy(model),
            f=str(path),
            args=tuple(inputs),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            dynamo=False,   # use legacy TorchScript exporter; dynamo breaks on GRU flat_weights
        )

    # Validate with ONNX checker
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    # Spot-check outputs match PyTorch
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_out = model(*inputs)
    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    ort_out = sess.run(output_names, {k: v.detach().numpy() for k, v in zip(input_names, inputs)})
    for name, ref, out in zip(output_names, ref_out, ort_out):
        try:
            np.testing.assert_allclose(ref.detach().numpy(), out, rtol=1e-5, atol=1e-4)
            print(f"    ✓ {name}: outputs match")
        except AssertionError as e:
            print(f"    ⚠ {name}: {e}")

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"    Saved → {path} ({size_mb:.1f} MB)")
    return ref_out



def main():
    print("Loading DeepFilterNet3 model (PyTorch)...")
    model, df_state, _ = init_df(log_level="WARNING", log_file=None)
    model = deepcopy(model).cpu().eval()

    p = ModelParams()

    # Build a 1-second sample to trace shapes
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # enc takes feat_spec in [B, T, F, 2] transposed form
    feat_spec_t = feat_spec.transpose(1, 4).squeeze(4)

    # ──────────────────────────────────────────────────────────────────
    # 1. Encoder: (feat_erb, feat_spec_t) → (e0, e1, e2, e3, emb, c0, lsnr)
    # ──────────────────────────────────────────────────────────────────
    enc_inputs = (feat_erb, feat_spec_t)
    enc_input_names = ["feat_erb", "feat_spec"]
    enc_output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    enc_dynamic = {
        "feat_erb":  {2: "S"},
        "feat_spec": {2: "S"},
        "e0":  {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"}, "e3": {2: "S"},
        "emb": {1: "S"}, "c0": {2: "S"}, "lsnr": {1: "S"},
    }
    enc_out = export_submodel(
        OUT_DIR / "enc.onnx", model.enc, enc_inputs,
        enc_input_names, enc_output_names, enc_dynamic
    )
    e0, e1, e2, e3, emb, c0, lsnr = enc_out

    # ──────────────────────────────────────────────────────────────────
    # 2. ERB Decoder: (emb, e3, e2, e1, e0) → (m,)
    # ──────────────────────────────────────────────────────────────────
    erb_inputs = (emb.clone(), e3, e2, e1, e0)
    erb_input_names = ["emb", "e3", "e2", "e1", "e0"]
    erb_output_names = ["m"]
    erb_dynamic = {
        "emb": {1: "S"}, "e3": {2: "S"}, "e2": {2: "S"},
        "e1": {2: "S"}, "e0": {2: "S"}, "m": {2: "S"},
    }
    export_submodel(
        OUT_DIR / "erb_dec.onnx", model.erb_dec, erb_inputs,
        erb_input_names, erb_output_names, erb_dynamic
    )

    # ──────────────────────────────────────────────────────────────────
    # 3. DF Decoder: (emb, c0) → (coefs,)
    # ──────────────────────────────────────────────────────────────────
    df_inputs = (emb.clone(), c0)
    df_input_names = ["emb", "c0"]
    df_output_names = ["coefs"]
    df_dynamic = {
        "emb": {1: "S"}, "c0": {2: "S"}, "coefs": {1: "S"},
    }
    export_submodel(
        OUT_DIR / "df_dec.onnx", model.df_dec, df_inputs,
        df_input_names, df_output_names, df_dynamic
    )

    # Copy config.ini alongside the ONNX files
    src_cfg = Path.home() / ".cache/DeepFilterNet/DeepFilterNet3/config.ini"
    shutil.copy(src_cfg, OUT_DIR / "config.ini")
    print(f"\n✅ config.ini copied")

    # Print final sizes
    print("\n=== ONNX model sizes ===")
    total = 0
    for f in sorted(OUT_DIR.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name}: {sz/1024:.0f} KB")
    print(f"  TOTAL: {total/1024/1024:.1f} MB")
    print("\nDone! Run: python3 onnx_enhancer.py to test inference.")


if __name__ == "__main__":
    main()
