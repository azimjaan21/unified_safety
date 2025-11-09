import random, yaml, cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# === CONFIG ===
YAML_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety.yaml"
NUM_IMAGES = 12
GRID_COLS = 4

def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

data = read_yaml(YAML_PATH)
names = data.get("names", ["helmet", "vest", "head", "fire"])
base = Path(data["path"])
img_dir = base / "val" / "images"
lbl_dir = base / "val" / "labels"

colors = {0:'#00bfff', 1:'#32cd32', 2:'#ffd700', 3:'#ff4500'}  # helmet, vest, head, fire

# Collect labeled images
all_imgs = [p for p in img_dir.rglob("*.jpg")] + [p for p in img_dir.rglob("*.png")]
imgs = [p for p in all_imgs if (lbl_dir / p.with_suffix(".txt").name).exists()]
random.shuffle(imgs)
imgs = imgs[:NUM_IMAGES]

rows = (NUM_IMAGES + GRID_COLS - 1) // GRID_COLS
fig, axes = plt.subplots(rows, GRID_COLS, figsize=(18, rows * 5))
axes = axes.flatten() if NUM_IMAGES > 1 else [axes]

for idx, img_path in enumerate(imgs):
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
    img = cv2.imread(str(img_path))
    if img is None:
        axes[idx].set_title("❌ Can't read")
        axes[idx].axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    axes[idx].imshow(img)
    axes[idx].axis("off")
    axes[idx].set_title(img_path.name, fontsize=9)

    for line in open(lbl_path, "r", encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x, y, bw, bh = map(float, parts[1:5])
        cx, cy, bw, bh = x*w, y*h, bw*w, bh*h
        x1, y1 = cx - bw/2, cy - bh/2
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                 edgecolor=colors.get(cls, 'white'), facecolor='none')
        axes[idx].add_patch(rect)
        axes[idx].text(x1, y1-5, names[cls],
                       color='white', fontsize=8, weight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=1))

for j in range(len(imgs), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
print(f"✅ Visualization complete — showing {len(imgs)} labeled samples from {lbl_dir}")
