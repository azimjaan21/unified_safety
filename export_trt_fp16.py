"""
----------------------
Export YOLO model to TensorRT FP16 format

Author: Azimjon Akhtamov
"""

from ultralytics import YOLO
import torch, os

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\fine_tune_unify_safety\finetune_unified_safety\weights\best.pt"
BASE_EXPORT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\engine_exports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH).to(DEVICE)

# === Ensure export folders exist ===
fp16_dir = os.path.join(BASE_EXPORT_DIR, "FP16")
os.makedirs(fp16_dir, exist_ok=True)

# === FP16 EXPORT ===
print("\n🔶 Exporting TensorRT FP16 engine...")
fp16_out = model.export(
    format="engine",
    device=DEVICE,
    half=True,           # FP16 precision
    dynamic=True,
    simplify=True,
    project=fp16_dir,
    name="best_fp16",
)
print(f"✅ FP16 TensorRT export complete!\n📁 Saved to: {fp16_dir}")

