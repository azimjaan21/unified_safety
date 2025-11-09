from pathlib import Path
from collections import Counter
from tqdm import tqdm
import yaml

# === CONFIG ===
YAML_PATH = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\new_unify_safety.yaml"

def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

data = read_yaml(YAML_PATH)
names = data.get("names", {})
base = Path(data["path"])

splits = {"train": base / "train" / "labels", "val": base / "val" / "labels"}

overall = Counter()
for split, lbl_dir in splits.items():
    counts = Counter()
    label_files = list(lbl_dir.rglob("*.txt"))
    for f in tqdm(label_files, desc=f"Counting {split}", ncols=100):
        for line in open(f, "r", encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            counts[cls] += 1
            overall[cls] += 1
    print(f"\n✅ {split.upper()} → {len(label_files)} label files")
    for k in sorted(counts.keys()):
        name = names[k] if isinstance(names, list) and k < len(names) else f"class_{k}"
        print(f"  {k}: {name} → {counts[k]} boxes")
    print(f"  Total boxes in {split}: {sum(counts.values())}\n")

print("📈 Overall Totals:")
for k in sorted(overall.keys()):
    name = names[k] if isinstance(names, list) and k < len(names) else f"class_{k}"
    print(f"  {k}: {name} → {overall[k]} boxes")
print(f"Total boxes (all splits): {sum(overall.values())}")
print("✅ Counting complete.")
