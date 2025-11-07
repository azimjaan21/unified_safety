import os, yaml
from collections import Counter
from pathlib import Path
from tqdm import tqdm  

# === CONFIG ===
PPE_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe.yaml"

# === FUNCTIONS ===
def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_label_dirs(data):
    base = Path(data.get('path', '.'))
    subsets = ['train', 'val', 'test']
    label_dirs = []
    for s in subsets:
        sub = data.get(s)
        if sub:
            subpath = base / sub.replace('images', 'labels')
            if subpath.exists():
                label_dirs.append(subpath)
    return label_dirs

def count_labels(label_dirs):
    counter = Counter()
    all_label_files = []
    for d in label_dirs:
        all_label_files += list(Path(d).rglob("*.txt"))

    for f in tqdm(all_label_files, desc="🔍 Counting labels", ncols=100):
        for line in open(f, "r", encoding="utf-8"):
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

    return counter, len(all_label_files)

# === MAIN ===
if __name__ == "__main__":
    data = read_yaml(PPE_YAML)
    label_dirs = get_label_dirs(data)
    counts, nfiles = count_labels(label_dirs)

    print(f"\n✅ Scanned {nfiles} label files from:")
    for d in label_dirs:
        print("   ", d)

    print("\n📊 Class counts:")
    names = data.get("names", [])
    for k in sorted(counts.keys()):
        if isinstance(names, list) and k < len(names):
            name = names[k]
        else:
            name = f"class_{k}"
        print(f"  {k}: {name} → {counts[k]} boxes")

    print("Total boxes:", sum(counts.values()))
    print("✅ Counting complete.")