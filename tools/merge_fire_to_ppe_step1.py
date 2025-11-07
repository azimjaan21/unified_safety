import shutil
from pathlib import Path

# === CONFIG ===
PPE_ROOT  = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe")
FIRE_ROOT = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire_smoke")

SPLITS = ["train", "val"]  # step 1 only train/val
IMG_EXTS = [".jpg", ".png", ".jpeg"]

def copy_folder(src_img, src_lbl, dst_img, dst_lbl):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0

    for img_file in src_img.rglob("*"):
        if img_file.suffix.lower() not in IMG_EXTS:
            continue
        lbl_file = src_lbl / img_file.with_suffix(".txt").name
        if not lbl_file.exists():
            continue  # skip unlabeled

        # define destination
        dst_img_file = dst_img / img_file.name
        dst_lbl_file = dst_lbl / lbl_file.name

        # copy files
        shutil.copy2(img_file, dst_img_file)
        shutil.copy2(lbl_file, dst_lbl_file)
        copied += 1

    return copied


if __name__ == "__main__":
    total_copied = 0
    for split in SPLITS:
        src_img = FIRE_ROOT / split / "images"
        src_lbl = FIRE_ROOT / split / "labels"
        dst_img = PPE_ROOT / split / "images"
        dst_lbl = PPE_ROOT / split / "labels"

        print(f"\n🚀 Merging {split.upper()} split...")
        n = copy_folder(src_img, src_lbl, dst_img, dst_lbl)
        total_copied += n
        print(f"✅ {n} Fire/Smoke {split} images merged into PPE.")

    print(f"\n🎉 Step 1 complete — Total {total_copied} Fire/Smoke images merged into PPE.")
    print("🟢 Next: move TEST split after verifying this merge works fine.")
