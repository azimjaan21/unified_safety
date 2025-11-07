# verify_ppe_labels.py
import os, yaml, cv2
from pathlib import Path

PPE_YAML = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\fire.yaml"
VALID_CLASSES = {0, 1, 2}

def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_labels(label_dir, img_dir):
    bad_classes = []
    missing_images = []
    invalid_boxes = []
    for lbl_path in Path(label_dir).rglob('*.txt'):
        img_path = Path(img_dir) / lbl_path.name.replace('.txt', '.jpg')
        if not img_path.exists():
            img_path = Path(img_dir) / lbl_path.name.replace('.txt', '.png')
        if not img_path.exists():
            missing_images.append(lbl_path)
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for ln in f:
                if not ln.strip():
                    continue
                parts = ln.split()
                cls = int(parts[0])
                if cls not in VALID_CLASSES:
                    bad_classes.append((lbl_path, cls))
                vals = list(map(float, parts[1:]))
                if not all(0 <= v <= 1 for v in vals):
                    invalid_boxes.append((lbl_path, vals))
    return bad_classes, missing_images, invalid_boxes

if __name__ == "__main__":
    data = read_yaml(PPE_YAML)
    base = Path(data.get('path', '.'))
    for subset in ['train', 'val']:
        img_dir = base / data[subset]
        lbl_dir = img_dir.parent / 'labels' / Path(data[subset]).name
        print(f"\n🔍 Checking {subset}: {lbl_dir}")
        bad_cls, miss_img, bad_box = check_labels(lbl_dir, img_dir)
        print(f"  ⚠ Wrong class indices: {len(bad_cls)}")
        print(f"  ⚠ Missing images: {len(miss_img)}")
        print(f"  ⚠ Invalid boxes: {len(bad_box)}")

        if bad_cls:
            print("  → Examples:", bad_cls[:5])
        if miss_img:
            print("  → Missing image examples:", miss_img[:5])
        if bad_box:
            print("  → Invalid box examples:", bad_box[:3])
