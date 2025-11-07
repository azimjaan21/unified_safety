from pathlib import Path

IMG_DIR = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe\val\images"
LBL_DIR = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\data\ppe\val\labels"

missing = []
for img in Path(IMG_DIR).rglob("*.jpg"):
    lbl = Path(LBL_DIR) / img.with_suffix(".txt").name
    if not lbl.exists():
        missing.append(img.name)

print(f"Total unlabeled images: {len(missing)}")
for name in missing[:20]:  # show first 20
    print(" -", name)
