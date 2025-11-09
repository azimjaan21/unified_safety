import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
FIRE_BASE = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\fire_test")
PPE_BASE  = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\ppe_test")

SPLITS = [("t1", "test1"), ("t2", "test2")]  # fire→ppe mapping
IMG_EXTS = (".jpg", ".png", ".jpeg")

moved_imgs, moved_labels = 0, 0

for fire_split, ppe_split in SPLITS:
    fire_img_dir = FIRE_BASE / fire_split / "images"
    fire_lbl_dir = FIRE_BASE / fire_split / "labels"

    ppe_img_dir = PPE_BASE / ppe_split / "images"
    ppe_lbl_dir = PPE_BASE / ppe_split / "labels"

    # create ppe dirs if not exist
    ppe_img_dir.mkdir(parents=True, exist_ok=True)
    ppe_lbl_dir.mkdir(parents=True, exist_ok=True)

    # === move images ===
    img_files = [p for p in fire_img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    for img in tqdm(img_files, desc=f"📦 Moving {fire_split} images → {ppe_split}"):
        dest = ppe_img_dir / img.name
        if dest.exists():
            dest = ppe_img_dir / f"fire_{img.name}"  # avoid overwrite
        shutil.move(str(img), str(dest))
        moved_imgs += 1

    # === move labels ===
    lbl_files = list(fire_lbl_dir.rglob("*.txt"))
    for lbl in tqdm(lbl_files, desc=f"📝 Moving {fire_split} labels → {ppe_split}"):
        dest = ppe_lbl_dir / lbl.name
        if dest.exists():
            dest = ppe_lbl_dir / f"fire_{lbl.name}"  # avoid overwrite
        shutil.move(str(lbl), str(dest))
        moved_labels += 1

print("\n🎉 Merge complete!")
print(f"   Total images moved: {moved_imgs}")
print(f"   Total labels moved: {moved_labels}")
print("✅ Fire test data successfully merged into PPE test folders.")
