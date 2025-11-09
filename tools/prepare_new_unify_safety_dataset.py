import shutil, random
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
SOURCE_PPE_TEST = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\ppe_test")
DEST_BASE = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\new_unify_safety")

TRAIN_RATIO = 0.8     # 80 % train, 20 % val
IMG_EXTS = (".jpg", ".jpeg", ".png")

# === STEP 1: gather all images + labels ===
all_images = []
for split in ["test1", "test2"]:
    img_dir = SOURCE_PPE_TEST / split / "images"
    all_images += [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

print(f"🔍 Found {len(all_images)} total images across test1/test2")

random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_RATIO)
train_imgs = all_images[:split_idx]
val_imgs   = all_images[split_idx:]

splits = {"train": train_imgs, "val": val_imgs}

# === STEP 2: create folders ===
for subset in ["train", "val"]:
    (DEST_BASE / subset / "images").mkdir(parents=True, exist_ok=True)
    (DEST_BASE / subset / "labels").mkdir(parents=True, exist_ok=True)

# === STEP 3: copy files ===
for subset, img_list in splits.items():
    print(f"\n🚀 Copying {subset.upper()} split ({len(img_list)} images)...")
    for img_path in tqdm(img_list, ncols=100):
        # locate matching label
        label_path = (img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name)
        dest_img = DEST_BASE / subset / "images" / img_path.name
        dest_lbl = DEST_BASE / subset / "labels" / img_path.with_suffix(".txt").name

        shutil.copy2(img_path, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_lbl)

print("\n🎉 Dataset creation complete.")
print(f"   Train images: {len(train_imgs)}")
print(f"   Val images:   {len(val_imgs)}")

# === STEP 4: write YAML file ===
yaml_path = DEST_BASE / "new_unify_safety.yaml"
yaml_content = f"""# Unified PPE + Fire dataset
path: {DEST_BASE.as_posix()}
train: train/images
val: val/images

nc: 4
names: ['helmet', 'vest', 'head', 'fire']
"""
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✅ YAML file saved → {yaml_path}")
