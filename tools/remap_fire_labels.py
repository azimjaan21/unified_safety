from pathlib import Path

# === CONFIG ===
FIRE_DATA_ROOT = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire_smoke"
SPLITS = ["train", "val", "test"]

# Mapping: old_class → new_class
CLASS_MAP = {1: 3, 2: 4}

# === MAIN ===
updated_files = 0
skipped_files = 0

for split in SPLITS:
    lbl_dir = Path(FIRE_DATA_ROOT) / split / "labels"
    if not lbl_dir.exists():
        print(f"⚠ No label folder found for {split}, skipping.")
        continue

    for lbl_file in lbl_dir.rglob("*.txt"):
        lines = []
        changed = False
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                if cls_id in CLASS_MAP:
                    parts[0] = str(CLASS_MAP[cls_id])
                    changed = True
                lines.append(" ".join(parts))

        if changed:
            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            updated_files += 1
        else:
            skipped_files += 1

print("\n✅ Re-indexing complete.")
print(f"   Updated label files: {updated_files}")
print(f"   Unchanged label files: {skipped_files}")
