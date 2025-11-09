import cv2
import torch
import time
from ultralytics import YOLO
import numpy as np

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\fine_tune_unify_safety\finetune_unified_safety\weights\best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
CAM_INDEX = 0  # 0 = default webcam
WINDOW_NAME = "YOLO Unified Safety Detection"

# === LOAD MODEL ===
print(f"🚀 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"✅ Model loaded on {DEVICE.upper()}")

# === SETUP CAMERA ===
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

# Optional: set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === FPS TRACKING ===
fps_smooth = 0
alpha = 0.9  # smoothing factor
prev_time = time.time()

print("🎥 Press 'Q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received, skipping...")
        continue

    # === Inference ===
    start_time = time.time()
    results = model(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)
    end_time = time.time()

    # === FPS Calculation ===
    fps = 1 / (end_time - start_time)
    fps_smooth = alpha * fps_smooth + (1 - alpha) * fps  # exponential moving average

    # === Visualization ===
    annotated = results[0].plot()
    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"AVG: {fps_smooth:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    # === Display ===
    cv2.imshow(WINDOW_NAME, annotated)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam test ended.")
