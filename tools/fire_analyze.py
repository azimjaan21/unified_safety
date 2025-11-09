from pathlib import Path
from collections import Counter
from tqdm import tqdm

# === CONFIG ===
FIRE_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\fire_test")
SPLITS = ["t3", "t4"]

print("🔍 Scanning Fire dataset...")
counter = Counter()

for split in SPLITS:
    lbl_dir = FIRE_PATH / split / "labels"
    if not lbl_dir.exists():
        print(f"⚠ No label folder found in {split}")
        continue

    label_files = list(lbl_dir.rglob("*.txt"))
    print(f"\n📂 {split.upper()} → {len(label_files)} label files")

    for f in tqdm(label_files, desc=f"Counting {split}", ncols=100):
        for line in open(f, "r", encoding="utf-8"):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                counter[cls] += 1
            except:
                continue

print("\n📊 Class Counts Across All Splits:")
for k, v in sorted(counter.items()):
    print(f"  Class {k} → {v} boxes")
print("Total boxes:", sum(counter.values()))
print("✅ Analysis complete.")
