from pathlib import Path

# === CONFIG ===
FIRE_TEST_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\fire_test")
SPLITS = ["t1", "t2"]   # test1 = t1, test2 = t2
REMAP = {0: 3}          # fire:0 → 3

updated = 0
skipped = 0

for split in SPLITS:
    lbl_dir = FIRE_TEST_PATH / split / "labels"
    if not lbl_dir.exists():
        print(f"⚠ No labels folder found for {split}")
        continue

    label_files = list(lbl_dir.rglob("*.txt"))
    for f in label_files:
        new_lines = []
        changed = False
        for line in open(f, "r", encoding="utf-8"):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except:
                continue
            if cls in REMAP:
                cls = REMAP[cls]
                changed = True
            parts[0] = str(cls)
            new_lines.append(" ".join(parts))
        if changed:
            updated += 1
        else:
            skipped += 1
        with open(f, "w", encoding="utf-8") as out:
            out.write("\n".join(new_lines) + "\n")

print(f"\n✅ Re-indexing complete.")
print(f"   Updated label files: {updated}")
print(f"   Unchanged label files: {skipped}")
