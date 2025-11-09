from pathlib import Path
from collections import Counter
from tqdm import tqdm
import yaml

# === CONFIG ===
YAML_PATH = Path(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety\new_unify_safety.yaml")

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

data = read_yaml(YAML_PATH)
BASE_PATH = Path(data["path"])
NAMES = data["names"]
SPLITS = ["train", "val"]

overall = Counter()

for split in SPLITS:
    lbl_dir = BASE_PATH / split / "labels"
    counter = Counter()
    label_files = list(lbl_dir.rglob("*.txt"))

    for f in tqdm(label_files, desc=f"Counting {split}", ncols=90):
        for line in open(f, "r", encoding="utf-8"):
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1
                overall[cls] += 1

    print(f"\n📂 {split.upper()} → {len(label_files)} label files")
    for k in sorted(counter.keys()):
        print(f"  {k}: {NAMES[k]} → {counter[k]} boxes")
    print(f"  Total boxes in {split}: {sum(counter.values())}\n")

print("📈 Overall Totals:")
for k in sorted(overall.keys()):
    print(f"  {k}: {NAMES[k]} → {overall[k]} boxes")
print(f"Total boxes (all splits): {sum(overall.values())}")
print("✅ Counting complete.")
