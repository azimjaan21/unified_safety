"""
FPS_benchmark_fixed.py
-----------------------
Benchmark average FPS for:
1️⃣ PyTorch (FP32)
2️⃣ TensorRT FP16
3️⃣ TensorRT INT8

Author: Azimjon Akhtamov
"""

import torch, time
from ultralytics import YOLO

# === CONFIG ===
MODEL_PT = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\fine_tune_unify_safety\finetune_unified_safety\weights\best.pt"
MODEL_FP16 = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\engine_exports\FP16\best_fp16.engine"
MODEL_INT8 = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\engine_exports\INT8\best_int8.engine"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
N_ITERS = 100  # number of runs for averaging


def benchmark(model_path, label, warmup=10):
    """Run FPS benchmark for YOLO model (.pt or .engine)."""
    print(f"\n🚀 Benchmarking: {label}")
    model = YOLO(model_path)

    # Random test image (simulates webcam frame)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to("cuda" if torch.cuda.is_available() else "cpu")

    # Warmup
    for _ in range(warmup):
        model.predict(source=dummy, device=DEVICE, verbose=False)

    # Timed inference
    start = time.time()
    for _ in range(N_ITERS):
        model.predict(source=dummy, device=DEVICE, verbose=False)
    end = time.time()

    fps = N_ITERS / (end - start)
    print(f"✅ {label}: {fps:.2f} FPS")
    return fps


if __name__ == "__main__":
    print("📊 Starting performance benchmark...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    fps_pt = benchmark(MODEL_PT, "PyTorch FP32")
    fps_fp16 = benchmark(MODEL_FP16, "TensorRT FP16")
    fps_int8 = benchmark(MODEL_INT8, "TensorRT INT8")

    print("\n==============================")
    print("📈 FPS Summary (TITAN RTX 24GB)")
    print("==============================")
    print(f"🔹 PyTorch FP32 : {fps_pt:.2f} FPS")
    print(f"🔸 TensorRT FP16: {fps_fp16:.2f} FPS")
    print(f"🔶 TensorRT INT8: {fps_int8:.2f} FPS")
    print("==============================")
    print("✅ Benchmark complete.")
