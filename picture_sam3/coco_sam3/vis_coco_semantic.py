from pathlib import Path
import numpy as np
from PIL import Image

NPZ_ROOT = Path("/home/via/mai/datasets/sam3_coco_label")
SAVE_ROOT = Path("/home/via/mai/datasets/sam3_coco_label_vis")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "road": (0, 80, 255),
    "construction vehicle": (255, 160, 0),
    "truck": (255, 0, 0),
    "car": (0, 255, 0),
    "bus": (255, 255, 0),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 255, 255),
    "rider": (180, 0, 255),
    "pedestrian": (255, 128, 128),
}

ALPHA = 0.45


def overlay_one(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    image_path = str(data["image_path"])
    semantic_label = data["semantic_label"]
    semantic_names = data["semantic_names"].astype(str)

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32)

    overlay = img_np.copy()

    for cid, cname in enumerate(semantic_names):
        mask = semantic_label == cid
        if not mask.any():
            continue

        color = np.array(COLORS.get(cname, (255, 255, 255)), dtype=np.float32)
        overlay[mask] = overlay[mask] * (1 - ALPHA) + color * ALPHA

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    rel = npz_path.relative_to(NPZ_ROOT)
    save_dir = SAVE_ROOT / rel.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{npz_path.stem}_coco_vis.jpg"
    Image.fromarray(overlay).save(save_path)

    return save_path


def main():
    files = sorted(NPZ_ROOT.rglob("*.npz"))

    print("npz 数量:", len(files))

    for p in files:
        save_path = overlay_one(p)
        print("saved:", save_path)


if __name__ == "__main__":
    main()
