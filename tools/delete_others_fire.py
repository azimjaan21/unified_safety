import os
from pathlib import Path

# === CONFIG ===
FIRE_DATA_ROOT = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire_smoke"
SPLITS = ["train", "val", "test"]
TARGET_CLASS = 0   # "others"

# === MAIN ===
deleted = {"labels": 0, "images": 0}

for split in SPLITS:
    lbl_dir = Path(FIRE_DATA_ROOT) / split / "labels"
    img_dir = Path(FIRE_DATA_ROOT) / split / "images"

    if not lbl_dir.exists():
        print(f"⚠ No label folder for {split}, skipping.")
        continue

    for lbl_file in lbl_dir.rglob("*.txt"):
        remove_flag = False
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cls_id = int(line.split()[0])
                if cls_id == TARGET_CLASS:
                    remove_flag = True
                    break

        if remove_flag:
            # delete label
            lbl_file.unlink(missing_ok=True)
            deleted["labels"] += 1

            # delete corresponding image
            img_file = img_dir / lbl_file.with_suffix(".jpg").name
            if not img_file.exists():
                img_file = img_dir / lbl_file.with_suffix(".png").name
            if img_file.exists():
                img_file.unlink(missing_ok=True)
                deleted["images"] += 1

print("\n✅ Deletion complete.")
print(f"   Removed label files: {deleted['labels']}")
print(f"   Removed image files: {deleted['images']}")
