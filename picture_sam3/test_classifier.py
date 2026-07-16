"""
分类器测试与可视化
==================
功能：加载训练好的车辆分类器，对测试图片目录进行推理，生成带概率标注的图片。
- 输入：checkpoints/best_model_cls.pth + test_dir（图片目录）
- 输出：test_output/ 目录下的标注图片（文件名含预测类别和置信度）
- 每张图上叠加各类别的 softmax 概率，预测类别用绿色高亮
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ResNetClassifier, ImageDirDataset



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_eval_transform(crop_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(crop_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def run_classifier_test(model_path, test_dir, output_dir, visualize_fn,
                        img_size=224, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    model = ResNetClassifier(len(ckpt["classes"]), pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    classes = ckpt["classes"]

    print(f"Model: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    print(f"Classes: {classes}")

    transform = make_eval_transform(crop_size=img_size)
    ds = ImageDirDataset(test_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    os.makedirs(output_dir, exist_ok=True)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        ImageFont.truetype(font_path, 14)
    except Exception:
        font_path = None

    t0 = time.time()
    n_done = 0
    pbar = tqdm(loader, desc="Test", ncols=100)
    for imgs, paths in pbar:
        imgs_gpu = imgs.to(device)
        logits = model(imgs_gpu)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(1)
        preds = preds.cpu()
        confs = confs.cpu()

        for i in range(len(imgs)):
            img_pil = Image.open(paths[i]).convert("RGB")
            visualize_fn(img_pil, classes, probs[i].cpu(), preds[i].item(),
                         confs[i].item(), font_path, n_done, output_dir, paths[i])
            n_done += 1

    elapsed = time.time() - t0
    print(f"\nDone. {n_done} images in {elapsed:.1f}s ({n_done/elapsed:.1f} img/s)")
    print(f"Saved to {output_dir}")


def visualize_cls(img_pil, classes, probs, pred_idx, conf, font_path, idx,
                  output_dir, _path):
    w, h = img_pil.size
    scale = max(1, 448 // max(w, h))
    img_pil = img_pil.resize((w * scale, h * scale), Image.NEAREST)

    draw = ImageDraw.Draw(img_pil)
    lines = [f"{c}: {p:.2f}" for c, p in zip(classes, probs.tolist())]
    font_size = max(10, min(img_pil.width, img_pil.height) // 45)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    line_h = font_size + 3
    x, y = 3, 3
    for li, line in enumerate(lines):
        color = (0, 220, 0) if li == pred_idx else (150, 150, 150)
        draw.text((x, y + li * line_h), line, fill=color, font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))

    pred_cls = classes[pred_idx]
    fname = f"{idx:05d}_{pred_cls}_{conf:.2f}.png"
    img_pil.save(os.path.join(output_dir, fname))


if __name__ == "__main__":
    HERE = os.path.dirname(__file__)
    run_classifier_test(
        model_path=os.path.join(HERE, "checkpoints", "best_model_cls.pth"),
        test_dir="/home/via/mai/datasets/cls_dataset",
        output_dir=os.path.join(HERE, "test_output"),
        visualize_fn=visualize_cls,
    )
