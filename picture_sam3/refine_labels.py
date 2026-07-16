"""
用训练好的分类器修正 SAM3 标签
==============================
功能：加载训练好的车辆分类器，对 SAM3 标签中的车辆实例逐个重新分类，修正可能的误标。
- 输入：sam3_label/ .npz 标签 + images/ 原图 + checkpoints/best_model_cls.pth
- 输出：sam3_label_refined/ .npz 标签（结构与输入一致，仅修改类别名）
- 策略：只处理 car/construction vehicle/truck 三类；仅当分类器置信度 ≥ 0.9 且预测类别与原始不同时才覆盖
- 每个目标裁剪 → 贴白底 → 方形 → 送入分类器推理
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from train_cls import ResNetClassifier

# ====== Paths ======
IMAGES_ROOT = Path("/home/via/mai/datasets/images")
LABELS_ROOT = Path("/home/via/mai/datasets/sam3_label")
OUTPUT_ROOT = Path("/home/via/mai/datasets/sam3_label_refined")

# ====== Params ======
TARGET_CLASSES = {'car', 'construction vehicle', 'truck'}
CLASSIFIER_CONF_THRESHOLD = 0.9  # 只有分类器非常确定时才覆盖 SAM3 标签
FOREGROUND_RATIO = 0.85

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(crop_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(crop_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def crop_object(image: Image.Image, mask: np.ndarray) -> Image.Image | None:
    """Crop object by mask bbox, paste on white background, return RGB square."""
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

    fg_img = Image.fromarray(rgba, mode="RGBA")

    # Pad to square, then pad to foreground_ratio
    arr = np.array(fg_img)
    alpha = np.where(arr[..., 3] > 0)
    ay1, ay2 = alpha[0].min(), alpha[0].max()
    ax1, ax2 = alpha[1].min(), alpha[1].max()
    fg = arr[ay1:ay2, ax1:ax2]

    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    fg = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")

    new_size = int(size / FOREGROUND_RATIO)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    fg = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")

    bg = Image.new("RGBA", (new_size, new_size), (255, 255, 255, 255))
    fg_rgba = Image.fromarray(fg, mode="RGBA")
    return Image.alpha_composite(bg, fg_rgba).convert("RGB")


def refine_one(npz_path: Path, model, transform, device) -> int:
    """Refine labels in one npz file. Returns number of changes made."""
    data = np.load(npz_path, allow_pickle=True)
    classes = data["classes"]
    masks = data["masks"]

    cls_list = [str(c) for c in classes]
    n = len(cls_list)

    # Find original image
    rel = npz_path.relative_to(LABELS_ROOT)
    img_path = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = IMAGES_ROOT / rel.parent / f"{npz_path.stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        return 0

    img = Image.open(img_path).convert("RGB")
    changes = 0

    for i in range(n):
        if cls_list[i] not in TARGET_CLASSES:
            continue
        mask = masks[i]
        if mask.sum() < 256:
            continue

        crop = crop_object(img, mask)
        if crop is None:
            continue

        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = probs.max(0)
            pred_cls = model.classes[pred_idx.item()]

        if pred_cls != cls_list[i] and conf.item() >= CLASSIFIER_CONF_THRESHOLD:
            cls_list[i] = pred_cls
            changes += 1

    out_path = OUTPUT_ROOT / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        classes=np.array(cls_list),
        confidences=data["confidences"],
        masks=data["masks"],
    )
    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                        "checkpoints", "best_model_cls.pth"))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch", action="store_true",
                        help="Process every object in a batch for speed "
                             "(needs classifier wrapper support)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.model, map_location=device)
    model = ResNetClassifier(len(ckpt["classes"]), pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    model.classes = ckpt["classes"]  # attach for convenience
    print(f"Loaded model epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    print(f"Classifier classes: {model.classes}")

    transform = make_transform()

    all_npz = sorted(LABELS_ROOT.rglob("*.npz"))
    if args.limit > 0:
        all_npz = all_npz[:args.limit]

    total = len(all_npz)
    print(f"Found {total} npz files")
    print(f"Output: {OUTPUT_ROOT}")

    total_changes = 0
    changed_files = 0

    for npz_path in tqdm(all_npz, desc="Refining", unit="npz"):
        try:
            n = refine_one(npz_path, model, transform, device)
            if n > 0:
                total_changes += n
                changed_files += 1
        except Exception as e:
            tqdm.write(f"Error {npz_path.name}: {e}")

    print(f"\nDone. {changed_files}/{total} files modified, {total_changes} labels changed")
    print(f"Saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
