"""
export_trt_int8.py
-------------------
Export YOLO model to TensorRT INT8 (quantized) engine.
Uses calibration from your dataset for accurate INT8 scaling.

Author: Azimjon Akhtamov
"""

from ultralytics import YOLO
import torch, os

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\fine_tune_unify_safety\finetune_unified_safety\weights\best.pt"
DATA_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety\new_unify_safety.yaml"
EXPORT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\engine_exports\INT8"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"🚀 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# === INT8 EXPORT ===
print("\n🔷 Exporting TensorRT INT8 engine with calibration...")
model.export(
    format="engine",         # TensorRT format
    int8=True,               # enable INT8 quantization
    data=DATA_YAML,          # use dataset for calibration
    imgsz=640,               # calibration image size
    device=DEVICE,
    dynamic=True,            # allow dynamic batch
    simplify=True,           # graph simplification
    project=EXPORT_DIR,
    name="best_int8"
)

print(f"\n✅ INT8 TensorRT export complete!\n📁 Saved to: {EXPORT_DIR}")
print("🎉 INT8 quantized model ready for deployment.")
