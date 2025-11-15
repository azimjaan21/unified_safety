from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Colors

# Ultralytics brand blue (BGR) via palette index 0
ULTRA_BLUE = Colors()(0, bgr=True)

# -----------------------------
# CONFIGURATION
# -----------------------------
pose_model_path = "yolo11m-pose.pt"   # YOLOv11m-pose model
ppe_model_path = "weights/ppe.pt"             # PPE (helmet detection) model
image_path = "image.png"              # Input image
output_path = "output_helmet_pose.jpg"

# Define keypoint indices for head (COCO format)
head_kpt_indices = [0, 1, 2, 3, 4]  # nose, left eye, right eye, left ear, right ear

# Define head keypoint connection pairs
head_connections = [
    (1, 0), (2, 0),  # eyes → nose
    (1, 3), (2, 4),  # eyes → ears
    (1, 2)           # left ↔ right eye
]

# -----------------------------
# LOAD MODELS
# -----------------------------
pose_model = YOLO(pose_model_path)
ppe_model = YOLO(ppe_model_path)

# -----------------------------
# RUN INFERENCE
# -----------------------------
pose_results = pose_model(image_path, save=False, verbose=False)
ppe_results = ppe_model(image_path, save=False, verbose=False)

# Read image for drawing
img = cv2.imread(image_path)

# -----------------------------
# DRAW HELMET DETECTIONS
# -----------------------------
for r in ppe_results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), ULTRA_BLUE, 5)  # blue box

        # Add 'helmet' label in white with blue background
        label = "helmet"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 6
        tx, ty = x1, max(0, y1 - th - baseline - pad)
        cv2.rectangle(img, (tx - pad, ty), (tx + tw + pad, y1), ULTRA_BLUE, cv2.FILLED)
        cv2.putText(img, label, (tx, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# -----------------------------
# DRAW POSE KEYPOINTS & CONNECTIONS
# -----------------------------
for r in pose_results:
    if r.keypoints is None:
        continue

    for kps in r.keypoints.xy:
        pts = [(int(x), int(y)) for i, (x, y) in enumerate(kps) if i in head_kpt_indices]


       # Draw GREEN connections
        for i1, i2 in head_connections:
            if i1 < len(kps) and i2 < len(kps):
                x1, y1 = map(int, kps[i1])
                x2, y2 = map(int, kps[i2])
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                
        # Draw GREEN keypoints
        for (x, y) in pts:
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)

# -----------------------------
# SAVE & SHOW OUTPUT
# -----------------------------
cv2.imwrite(output_path, img)
cv2.imshow("Helmet + Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Combined visualization saved to: {output_path}")
