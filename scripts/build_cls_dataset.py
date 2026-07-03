"""Build vehicle classification dataset — self-contained, no project imports."""

import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ====== Paths ======
DATASET_ROOT = Path("/home/via/mai/datasets")
IMAGES_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "sam3_label"
OUTPUT_DIR = Path("/home/via/mai/datasets/cls_dataset")

# ====== Params ======
TARGET_CLASSES = ['construction vehicle', 'truck', 'car', 'bus']
CONF_MIN = 0.9
MAX_PER_CLASS = 5000
FOREGROUND_RATIO = 0.85


# ====== Image utils (inlined from shared) ======

def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """Crop RGBA image to foreground, pad to square, then pad to target ratio."""
    arr = np.array(image)
    assert arr.shape[-1] == 4
    alpha = np.where(arr[..., 3] > 0)
    y1, y2, x1, x2 = alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max()

    fg = arr[y1:y2, x1:x2]

    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    fg = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")

    new_size = int(size / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    fg = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")

    return Image.fromarray(fg)


def prepare_object_image(image: Image.Image, mask: np.ndarray,
                        foreground_ratio: float = FOREGROUND_RATIO) -> Image.Image | None:
    """Crop mask bbox, composite on white background, return RGB."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()

    cropped_img = np.array(image.crop((x1, y1, x2 + 1, y2 + 1)))
    cropped_mask = mask[y1:y2 + 1, x1:x2 + 1]

    rgba = np.zeros((y2 - y1 + 1, x2 - x1 + 1, 4), dtype=np.uint8)
    rgba[:, :, :3] = cropped_img
    rgba[:, :, 3] = cropped_mask.astype(np.uint8) * 255

    fg = resize_foreground(Image.fromarray(rgba, mode="RGBA"), foreground_ratio)
    bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, fg).convert("RGB")


# ====== Core logic ======

def extract_objects(npz_path: Path) -> list | None:
    """Extract qualifying object images from one NPZ file."""
    rel = npz_path.relative_to(LABELS_ROOT)
    img_path = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = IMAGES_ROOT / rel.parent / f"{npz_path.stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        return None
    base_name = f"{rel.parent.name}_{rel.stem}"

    data = np.load(npz_path, allow_pickle=True)
    classes = data["classes"]
    masks = data["masks"]
    confs = data["confidences"]

    if not (len(classes) > 0 and img_path.exists()):
        return None

    img = Image.open(img_path).convert("RGB")
    objs = []
    for i in range(len(classes)):
        cls = str(classes[i])
        if cls not in TARGET_CLASSES:
            continue
        if confs[i] < CONF_MIN:
            continue
        if masks[i].sum() < 256:
            continue
        obj = prepare_object_image(img, masks[i])
        if obj is not None:
            objs.append((obj, cls, i, base_name))

    return objs if objs else None


def main():
    all_npz = sorted(LABELS_ROOT.rglob("*.npz"))
    print(f"Found {len(all_npz)} npz files")

    # Clear output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = {c: 0 for c in TARGET_CLASSES}
    t0 = time.time()

    for npz_path in tqdm(all_npz, desc="Processing", unit="npz"):
        try:
            extracted = extract_objects(npz_path)
            if extracted is not None:
                for obj, cls, idx, base_name in extracted:
                    if class_counts[cls] >= MAX_PER_CLASS:
                        continue
                    fname = f"{base_name}_{idx}.png"
                    cls_dir = OUTPUT_DIR / cls
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    obj.save(cls_dir / fname)
                    class_counts[cls] += 1
        except Exception as e:
            print(f"Error processing {npz_path.name}: {e}", flush=True)

    elapsed = time.time() - t0
    total = len(all_npz)
    print(f"Done. {total} files in {elapsed:.1f}s ({total/elapsed:.1f} files/s)")

    print("\nClass distribution:")
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {class_counts[cls]}")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
