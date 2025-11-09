# QUANTIZATION SCRIPT FOR EXPORTING TRT FP16 AND INT8 MODELS
from ultralytics import YOLO
import torch

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\fine_tune_unify_safety\finetune_unified_safety\weights\best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH).to(DEVICE)

# === FP16 EXPORT ===
print("\n🔶 Exporting TensorRT FP16 engine...")
model.export(
    format="engine",     # TensorRT engine
    device=DEVICE,
    half=True,           # FP16 precision
    dynamic=True,        # allow dynamic batch sizes
    simplify=True        # optimize graph
)
print("✅ FP16 TensorRT export complete!\n")

# === INT8 EXPORT ===
# NOTE: Requires TensorRT calibration data (some sample images)
# This uses internal random calibration if dataset is not provided.
print("🔷 Exporting TensorRT INT8 engine...")
model.export(
    format="engine",
    device=DEVICE,
    int8=True,           # enable INT8 quantization
    dynamic=True,
    simplify=True
)
print("✅ INT8 TensorRT export complete!\n")

print("🎉 All exports finished successfully!")
