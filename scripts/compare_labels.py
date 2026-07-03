"""Side-by-side comparison: SAM3 raw vs classifier-refined labels."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

IMAGES_ROOT = Path("/home/via/mai/datasets/images")
ORIG_LABELS = Path("/home/via/mai/datasets/sam3_label")
REFINED_LABELS = Path("/home/via/mai/datasets/sam3_label_refined")
OUTPUT_DIR = Path("/home/via/mai/datasets/compare_labels")

CLASS_COLORS = {
    "car": (0, 255, 0),
    "construction vehicle": (255, 140, 0),
    "truck": (255, 220, 0),
    "bus": (255, 80, 0),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 255, 255),
    "rider": (255, 0, 0),
    "pedestrian": (255, 0, 0),
    "person": (255, 0, 0),
    "road": (0, 80, 255),
    "drivable ground": (0, 120, 255),
    "fence": (255, 255, 0),
    "barrier": (255, 255, 0),
    "railing": (255, 255, 0),
}
FALLBACK = [(255, 0, 0), (0, 255, 0), (0, 80, 255), (255, 255, 0)]


def color_for(cls, i):
    return CLASS_COLORS.get(str(cls), FALLBACK[i % len(FALLBACK)])


def find_image(npz_path, label_root):
    rel = npz_path.relative_to(label_root)
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = IMAGES_ROOT / rel.parent / f"{npz_path.stem}{ext}"
        if p.exists():
            return p
    return None


def make_vis(npz_path, label_root, title, max_side=480):
    """Generate a single visualization image from one npz."""
    img_path = find_image(npz_path, label_root)
    if img_path is None:
        return None

    data = np.load(npz_path, allow_pickle=True)
    classes = data["classes"]
    masks = data["masks"]

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        if mask.shape != (h, w):
            m = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            mask = np.array(m) > 0
        c = np.array(color_for(str(classes[i]), i), dtype=np.float32)
        arr[mask] = arr[mask] * 0.55 + c * 0.45

    vis = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = font_sm = ImageFont.load_default()

    # Title bar
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 16, th + 10], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)

    # Scale
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        vis = vis.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    return vis


def get_vehicle_labels(npz_path, label_root):
    """Return dict of {instance_id: cls} for vehicle classes in this npz."""
    data = np.load(npz_path, allow_pickle=True)
    classes = data["classes"]
    vehicle_classes = {'car', 'construction vehicle', 'truck', 'bus'}
    return {i: str(classes[i]) for i in range(len(classes)) if str(classes[i]) in vehicle_classes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1, help="只对比前 N 个有改动的文件")
    parser.add_argument("--all", action="store_true", help="对比所有文件，不只是有改动的")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    refined_npz = sorted(REFINED_LABELS.rglob("*.npz"))
    print(f"Refined: {len(refined_npz)} files")

    # Find files with actual changes
    changed = []
    if args.all:
        changed = refined_npz
    else:
        for rp in tqdm(refined_npz, desc="Finding changes"):
            orig = ORIG_LABELS / rp.relative_to(REFINED_LABELS)
            if not orig.exists():
                continue
            orig_vehicles = get_vehicle_labels(orig, ORIG_LABELS)
            refined_vehicles = get_vehicle_labels(rp, REFINED_LABELS)
            if orig_vehicles != refined_vehicles:
                changed.append(rp)

    print(f"Files with changes: {len(changed)}")

    if args.limit > 0:
        changed = changed[:args.limit]

    for rp in tqdm(changed, desc="Generating comparisons"):
        rel = rp.relative_to(REFINED_LABELS)
        orig = ORIG_LABELS / rel

        left = make_vis(orig, ORIG_LABELS, "SAM3 原始", max_side=440)
        right = make_vis(rp, REFINED_LABELS, "分类器修正后", max_side=440)
        if left is None or right is None:
            continue

        # Side by side
        lw, lh = left.size
        rw, rh = right.size
        w = lw + rw + 10
        h = max(lh, rh)
        combined = Image.new("RGB", (w, h), (30, 30, 30))
        combined.paste(left, (0, 0))
        combined.paste(right, (lw + 10, 0))

        # Divider
        draw = ImageDraw.Draw(combined)
        draw.line([(lw + 5, 0), (lw + 5, h)], fill=(100, 100, 100), width=2)

        # Diff summary in center
        orig_v = get_vehicle_labels(orig, ORIG_LABELS)
        ref_v = get_vehicle_labels(rp, REFINED_LABELS)
        diff_pairs = []
        for i, o_cls in orig_v.items():
            r_cls = ref_v.get(i, o_cls)
            if o_cls != r_cls:
                diff_pairs.append(f"#{i}: {o_cls} → {r_cls}")

        if diff_pairs:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            except Exception:
                font = ImageFont.load_default()
            y = 12
            for line in diff_pairs:
                bbox = draw.textbbox((0, 0), line, font=font)
                draw.rectangle([8, y, 8 + bbox[2] - bbox[0] + 12, y + bbox[3] - bbox[1] + 8],
                               fill=(0, 0, 0))
                draw.text((14, y + 4), line, fill=(255, 255, 0), font=font)
                y += 24

        out_path = OUTPUT_DIR / f"{rp.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(out_path, quality=92)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
