# visualize_ppe_label.py
import cv2, random, yaml
from pathlib import Path

# === CONFIG ===
PPE_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe.yaml"   
NUM_IMAGES = 5                            # how many random images to visualize
WINDOW_SIZE = 800                         # resize preview window width

# === LOAD DATA ===
def read_yaml(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

data = read_yaml(PPE_YAML)
names = data.get("names", {0:"helmet",1:"vest",2:"person_head"})
base = Path(data.get("path", "."))
img_dir = base / data.get("val", data.get("train"))  
lbl_dir = img_dir.parent / "labels" / Path(img_dir).name

# === RANDOM SAMPLES ===
imgs = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
random.shuffle(imgs)
imgs = imgs[:NUM_IMAGES]

# === VISUALIZE ===
for img_path in imgs:
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ Cannot read:", img_path)
        continue
    h, w = img.shape[:2]

    if lbl_path.exists():
        for line in open(lbl_path, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
            cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            color = [(255,0,0),(0,255,0),(0,255,255)][cls % 3]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, names.get(cls, str(cls)), (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print("⚠ No label file for:", img_path.name)

    # resize for display
    scale = WINDOW_SIZE / max(w, h)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(f"{img_path.name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print("✅ Visualization complete.")