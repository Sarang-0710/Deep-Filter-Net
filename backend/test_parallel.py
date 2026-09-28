import time
import numpy as np
from onnx_enhancer import OnnxEnhancer, ParallelOnnxEnhancer

def run_benchmark():
    np.random.seed(42)
    SR = 48000
    DURATION = 120  # 2 minutes
    print(f"Generating {DURATION} seconds of test audio...")
    audio = np.random.randn(SR * DURATION).astype(np.float32)

    print("\n--- 1. Testing Standard OnnxEnhancer (Sequential) ---")
    seq_enh = OnnxEnhancer("onnx_model")
    t0 = time.time()
    out_seq = seq_enh.enhance(audio)
    t1 = time.time()
    print(f"  Time taken: {t1-t0:.3f} seconds")

    print("\n--- 2. Testing ParallelOnnxEnhancer (4 Workers) ---")
    par_enh = ParallelOnnxEnhancer("onnx_model", num_workers=4)
    t0 = time.time()
    out_par = par_enh.process(audio)
    t1 = time.time()
    print(f"  Time taken: {t1-t0:.3f} seconds")

    print("\n--- 3. Verifying Audio Quality ---")
    # Verify that the parallel processing didn't introduce glitches
    corr = np.corrcoef(out_seq, out_par)[0, 1]
    print(f"  Correlation vs Sequential: {corr:.5f} (target 1.00000)")

if __name__ == "__main__":
    run_benchmark()
