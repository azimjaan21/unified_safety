from pathlib import Path
from collections import Counter
from tqdm import tqdm

# === CONFIG ===
PPE_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\Datasets\ppe_test")
SPLITS = ["test1", "test2"]
NAMES = {0: "helmet", 1: "vest", 2: "head", 3: "fire"}  # 4-class unified setup

def count_labels(lbl_dir):
    counter = Counter()
    label_files = list(lbl_dir.rglob("*.txt"))
    for f in tqdm(label_files, desc=f"Counting {lbl_dir.name}", ncols=100):
        for line in open(f, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls in NAMES:
                counter[cls] += 1
    return counter, len(label_files)

# === MAIN ===
total_counts = Counter()
for split in SPLITS:
    lbl_dir = PPE_PATH / split / "labels"
    if not lbl_dir.exists():
        print(f"⚠ No labels for {split}")
        continue

    counts, nfiles = count_labels(lbl_dir)
    print(f"\n✅ {split.upper()} → {nfiles} label files")
    for k in sorted(counts.keys()):
        print(f"  {k}: {NAMES[k]} → {counts[k]} boxes")
    print(f"  Total boxes in {split}: {sum(counts.values())}\n")

    total_counts.update(counts)

print("📈 Overall Totals:")
for k in sorted(total_counts.keys()):
    print(f"  {k}: {NAMES[k]} → {total_counts[k]} boxes")
print(f"Total boxes (all splits): {sum(total_counts.values())}")
print("✅ Counting complete.")
