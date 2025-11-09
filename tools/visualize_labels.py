import random, yaml
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === CONFIG ===
PPE_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_data.yaml"
NUM_IMAGES = 10
GRID_COLS = 3

# === LOAD YAML ===
def read_yaml(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

data = read_yaml(PPE_YAML)
names = data.get("names", ["fire", "others", "smoke"])
base = Path(data.get("path", "."))
img_dir = base / data.get("val", data.get("train"))  # use val if exists

# 🧠 FIXED: derive label dir by replacing 'images' with 'labels'
if "images" in str(img_dir):
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
else:
    lbl_dir = img_dir.parent / "labels"

# === FILTER LABELED IMAGES ONLY ===
all_imgs = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
imgs = [p for p in all_imgs if (lbl_dir / p.with_suffix(".txt").name).exists()]

if not imgs:
    print(f"❌ No labeled images found!\n🧭 Checked labels in: {lbl_dir}")
    exit()

random.shuffle(imgs)
imgs = imgs[:NUM_IMAGES]

# === COLOR PALETTE ===
colors = ['#ff3030', '#30ff30', '#30b0ff', '#ffb030', '#a030ff']

# === VISUALIZE IN GRID ===
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

    # Draw boxes
    for line in open(lbl_path, 'r', encoding='utf-8'):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
        cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
        x1, y1 = cx - bw / 2, cy - bh / 2
        rect = patches.Rectangle((x1, y1), bw, bh,
                                 linewidth=2,
                                 edgecolor=colors[cls % len(colors)],
                                 facecolor='none')
        axes[idx].add_patch(rect)
        axes[idx].text(x1, y1 - 5, names[cls],
                       color='white', fontsize=9, weight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=2))

# Hide any unused axes
for j in range(len(imgs), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
print(f"✅ Visualization complete — showing {len(imgs)} labeled samples from {lbl_dir}")
