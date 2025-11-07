import shutil
from pathlib import Path

# === CONFIG ===
PPE_ROOT  = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe")
FIRE_ROOT = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire_smoke")

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

        # Define destination
        dst_img_file = dst_img / img_file.name
        dst_lbl_file = dst_lbl / lbl_file.name

        # Copy image and label
        shutil.copy2(img_file, dst_img_file)
        shutil.copy2(lbl_file, dst_lbl_file)
        copied += 1

    return copied


if __name__ == "__main__":
    src_img = FIRE_ROOT / "test" / "images"
    src_lbl = FIRE_ROOT / "test" / "labels"
    dst_img = PPE_ROOT / "train" / "images"   # ✅ Add Fire test into PPE train
    dst_lbl = PPE_ROOT / "train" / "labels"

    print("\n🚀 Merging Fire TEST → PPE TRAIN ...")
    n = copy_folder(src_img, src_lbl, dst_img, dst_lbl)
    print(f"✅ {n} Fire/Smoke test images merged into PPE/train/")
    print("🎉 Step 2 complete — Fire test data fully integrated into unified training set.")
