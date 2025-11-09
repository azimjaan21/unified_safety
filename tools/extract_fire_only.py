import shutil, random
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
PPE_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\ppe_test")
FIRE_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\fire_only")
DST_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety")

VAL_RATIO = 0.1  # 10% validation
SEED = 42
random.seed(SEED)

# === HELPERS ===
def collect_pairs(base_path, subfolders=("test1", "test2")):
    pairs = []
    for sub in subfolders:
        img_dir = base_path / sub / "images"
        lbl_dir = base_path / sub / "labels"
        for lbl in lbl_dir.rglob("*.txt"):
            img = img_dir / lbl.with_suffix(".jpg").name
            if not img.exists():
                img = img_dir / lbl.with_suffix(".png").name
            if img.exists():
                pairs.append((img, lbl))
    return pairs

def copy_pairs(pairs, split):
    img_dst = DST_PATH / split / "images"
    lbl_dst = DST_PATH / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img, lbl in tqdm(pairs, desc=f"Copying {split}", ncols=90):
        shutil.copy2(img, img_dst / img.name)
        shutil.copy2(lbl, lbl_dst / lbl.name)

# === 1️⃣ Collect all pairs ===
ppe_pairs = collect_pairs(PPE_PATH)
fire_pairs = collect_pairs(FIRE_PATH, subfolders=("t3", "t4"))
all_pairs = ppe_pairs + fire_pairs

print(f"\n📊 Found {len(ppe_pairs)} PPE samples + {len(fire_pairs)} Fire samples = {len(all_pairs)} total")

# === 2️⃣ Shuffle and split ===
random.shuffle(all_pairs)
n_val = int(len(all_pairs) * VAL_RATIO)
val_pairs = all_pairs[:n_val]
train_pairs = all_pairs[n_val:]

print(f"🧩 Splitting: {len(train_pairs)} train / {len(val_pairs)} val")

# === 3️⃣ Copy files ===
copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("\n✅ Unified dataset created successfully!")
print(f"   Total train: {len(train_pairs)}")
print(f"   Total val: {len(val_pairs)}")
print(f"📁 Path: {DST_PATH}")

# === 4️⃣ Write YAML ===
yaml_content = f"""# Unified PPE + Fire dataset
path: {DST_PATH}
train: train/images
val: val/images

nc: 4
names: ['helmet', 'vest', 'head', 'fire']
"""
yaml_file = DST_PATH / "new_unify_safety.yaml"
yaml_file.write_text(yaml_content, encoding="utf-8")
print(f"🧾 YAML saved → {yaml_file}")
