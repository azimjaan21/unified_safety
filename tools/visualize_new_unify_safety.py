import random, cv2, yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# === CONFIG ===
YAML_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety\new_unify_safety.yaml")
NUM_IMAGES = 20
GRID_COLS = 4

# === LOAD DATA ===
def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

data = read_yaml(YAML_PATH)
base = Path(data["path"])
names = data["names"]

splits = ["train"]
colors = ['#ff3030', '#30ff30', '#30b0ff', '#ffb030']  # helmet, vest, head, fire

# === COLLECT LABELED IMAGES ===
imgs = []
for split in splits:
    img_dir = base / split / "images"
    lbl_dir = base / split / "labels"
    for img_path in img_dir.glob("*.*"):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            imgs.append((img_path, lbl_path))
random.shuffle(imgs)
imgs = imgs[:NUM_IMAGES]

if not imgs:
    print("❌ No labeled images found.")
    exit()

# === VISUALIZE ===
rows = (NUM_IMAGES + GRID_COLS - 1) // GRID_COLS
fig, axes = plt.subplots(rows, GRID_COLS, figsize=(20, rows * 5))
axes = axes.flatten()

for idx, (img_path, lbl_path) in enumerate(imgs):
    img = cv2.imread(str(img_path))
    if img is None:
        axes[idx].set_title("Missing image")
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
        cls, x, y, bw, bh = int(parts[0]), *map(float, parts[1:5])
        cx, cy, bw, bh = x*w, y*h, bw*w, bh*h
        x1, y1 = cx - bw/2, cy - bh/2
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                 edgecolor=colors[cls % len(colors)], facecolor='none')
        axes[idx].add_patch(rect)
        axes[idx].text(x1, y1 - 5, names[cls],
                       color='white', fontsize=8, weight='bold',
                       bbox=dict(facecolor='black', alpha=0.5, pad=1))

for j in range(len(imgs), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
print(f"✅ Visualization complete — showing {len(imgs)} samples.")
