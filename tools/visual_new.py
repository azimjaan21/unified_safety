import random, cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# === CONFIG ===
DATASET_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\fire_test\t2")  # change to t1 or t2
NUM_IMAGES = 9
GRID_COLS = 3
CLASSES = {0: "fire", 3: "others", 4: "smoke"}
COLORS = {0: '#00bfff', 3: '#ff4500', 4: '#a020f0'}  # blue=others, orange=fire, purple=smoke

img_dir = DATASET_PATH / "images"
lbl_dir = DATASET_PATH / "labels"

# === COLLECT IMAGES ===
all_imgs = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
imgs = []
for p in all_imgs:
    lbl = lbl_dir / p.with_suffix(".txt").name
    if not lbl.exists():
        continue
    with open(lbl, "r", encoding="utf-8") as f:
        if any(int(line.split()[0]) in CLASSES for line in f if line.strip()):
            imgs.append(p)

if not imgs:
    print("❌ No fire/smoke/others labeled images found!")
    exit()

random.shuffle(imgs)
imgs = imgs[:NUM_IMAGES]

# === VISUALIZE ===
rows = (NUM_IMAGES + GRID_COLS - 1) // GRID_COLS
fig, axes = plt.subplots(rows, GRID_COLS, figsize=(16, rows * 5))
axes = axes.flatten() if NUM_IMAGES > 1 else [axes]

for idx, img_path in enumerate(imgs):
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
    img = cv2.imread(str(img_path))
    if img is None:
        axes[idx].set_title("❌ Cannot read image")
        axes[idx].axis('off')
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(img_path.name, fontsize=10)

    for line in open(lbl_path, "r", encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            if cls not in CLASSES:
                continue
            x, y, bw, bh = map(float, parts[1:5])  # ✅ safely take first 4 numbers only
        except ValueError:
            continue

        cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
        x1, y1 = cx - bw / 2, cy - bh / 2
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                 edgecolor=COLORS.get(cls, 'white'),
                                 facecolor='none')
        axes[idx].add_patch(rect)
        axes[idx].text(x1, y1 - 5, CLASSES[cls],
                       color='white', fontsize=9, weight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=2))

# Hide unused axes
for j in range(len(imgs), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
print(f"✅ Visualization complete — showing {len(imgs)} fire/smoke/others images from {lbl_dir}")
