"""
webcam_test.py
-----------------------------
Runs live detection using either:
- PyTorch (.pt)
- TensorRT (.engine)

Shows smoothed FPS and GPU name overlay.
Author: Azimjon Akhtamov
"""

import cv2, torch, time, os
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\runs\engine_exports\FP16\best_fp16.engine"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF = 0.5
CAM = 0
EPS = 1e-5

# === CLASS COLORS (BGR) ===
CLASS_COLORS = {
    0: (0, 255, 255),   # helmet → yellow
    1: (0, 255, 0),     # vest → green
    2: (0, 0, 255),     # head → red
    3: (0, 165, 255),   # fire → orange
}

# === LOAD MODEL ===
print(f"🚀 Loading model from: {MODEL_PATH}")
ext = os.path.splitext(MODEL_PATH)[-1].lower()

if ext == ".pt":
    model = YOLO(MODEL_PATH).to(DEVICE)
else:
    # TensorRT engine doesn’t use .to()
    model = YOLO(MODEL_PATH, task="detect")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"✅ Model loaded on {gpu_name}")

# === CAMERA SETUP ===
cap = cv2.VideoCapture(CAM)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === FPS TRACKING ===
smooth_fps = 0
alpha = 0.9

print("🎥 Press 'Q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    t0 = time.time()
    results = model.predict(source=frame, conf=CONF, device=DEVICE, verbose=False)
    dt = max(time.time() - t0, EPS)
    fps = 1.0 / dt
    smooth_fps = alpha * smooth_fps + (1 - alpha) * fps

    annotated = frame.copy()
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # === FPS + GPU OVERLAY ===
    fps_text = f"FPS: {smooth_fps:.1f}"
    gpu_text = f"GPU: {gpu_name}"

    # Black background for overlay
    box_width = 360 if len(gpu_name) > 15 else 320
    cv2.rectangle(annotated, (15, 15), (15 + box_width, 90), (0, 0, 0), -1)

    # White FPS text
    cv2.putText(annotated, fps_text, (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Green GPU text
    cv2.putText(annotated, gpu_text, (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("🟢 Unified Safety Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam test ended.")
