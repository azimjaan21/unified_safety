import os, yaml
from collections import Counter
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# === CONFIG ===
PPE_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire.yaml"

# === FUNCTIONS ===
def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_label_dirs(data):
    base = Path(data.get('path', '.'))
    subsets = ['train', 'val', 'test']
    dirs = {}
    for s in subsets:
        sub = data.get(s)
        if sub:
            subpath = base / sub.replace('images', 'labels')
            if subpath.exists():
                dirs[s] = subpath
    return dirs

def count_labels_in_dir(label_dir):
    """Count classes in one label directory"""
    counter = Counter()
    label_files = list(Path(label_dir).rglob("*.txt"))
    for f in tqdm(label_files, desc=f"   📂 {label_dir.name}", ncols=100):
        for line in open(f, "r", encoding="utf-8"):
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1
    return counter, len(label_files)

def merge_counters(counters):
    total = Counter()
    for c in counters.values():
        total.update(c)
    return total

# === MAIN ===
if __name__ == "__main__":
    data = read_yaml(PPE_YAML)
    dirs = get_label_dirs(data)
    names = data.get("names", [])
    split_results = {}

    print("\n🔍 Counting labels per split...\n")

    for split, d in dirs.items():
        c, nfiles = count_labels_in_dir(d)
        split_results[split] = c
        print(f"✅ {split.upper()} → {nfiles} label files")

    # --- Print results per split
    print("\n📊 Split-wise Class Counts:")
    for split, c in split_results.items():
        print(f"\n🧩 {split.upper()}:")
        for k in sorted(c.keys()):
            name = names[k] if isinstance(names, list) and k < len(names) else f"class_{k}"
            print(f"  {k}: {name} → {c[k]} boxes")
        print(f"  Total boxes in {split}: {sum(c.values())}")

    # --- Print total summary
    total_counts = merge_counters(split_results)
    print("\n📈 Overall Totals:")
    for k in sorted(total_counts.keys()):
        name = names[k] if isinstance(names, list) and k < len(names) else f"class_{k}"
        print(f"  {k}: {name} → {total_counts[k]} boxes")
    print(f"Total boxes (all splits): {sum(total_counts.values())}")
    print("\n✅ Counting complete.")
