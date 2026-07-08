"""
SAM3 标签可视化与实例索引生成
==============================
功能：将 SAM3 标签（.npz）可视化叠加到原图上，生成带标注面板的检查图。
- 输入：images/ 原图 + sam3_label_with_pickup/ .npz 标签
- 输出：
  1. sam3_check_with_pickup/all_single/ — 单张可视化图（mask 叠加 + 右侧面板）
  2. sam3_check_with_pickup/all_grid/ — 多图拼页（每页 12 帧）
  3. sam3_check_with_pickup/all_instances_index.csv — 所有实例的索引表
- 功能亮点：
  - 前景/背景分离绘制（road 不会盖住车）
  - 重复目标去重（重叠 > 60% 且同属前景/背景的实例只保留一个）
  - 去重优先级：前景 > 背景 > 置信度 > 类别优先级 > 面积
  - 右侧面板列出实例详情、被抑制的重复项
"""

from pathlib import Path
import argparse
from collections import Counter
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_ROOT = Path("/home/via/mai/datasets/images")
LABEL_ROOT = Path("/home/via/mai/datasets/sam3_label_with_pickup")
SAVE_ROOT = Path("/home/via/mai/datasets/sam3_check_with_pickup")


CLASS_COLORS = {
    "road": (0, 80, 255),
    "drivable ground": (0, 120, 255),
    "asphalt ground": (0, 120, 255),
    "construction vehicle": (255, 140, 0),
    "truck": (255, 220, 0),
    "pickup": (255, 180, 0),
    "bus": (255, 80, 0),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 255, 255),
    "rider": (255, 0, 0),
    "pedestrian": (255, 0, 0),
    "person": (255, 0, 0),
    "fence": (255, 255, 0),
    "railing": (255, 255, 0),
    "barrier": (255, 255, 0),
}

FALLBACK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 80, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 180, 120),
    (255, 80, 160),
]


BACKGROUND_CLASSES = {
    "road",
    "drivable ground",
    "asphalt ground",
}


CLASS_PRIORITY = {
    "construction vehicle": 100,
    "truck": 90,
    "bus": 85,
    "pickup": 80,
    "car": 70,
    "motorcycle": 70,
    "bicycle": 70,
    "rider": 60,
    "pedestrian": 60,
    "person": 60,
    "fence": 50,
    "railing": 50,
    "barrier": 50,
    "road": 10,
    "drivable ground": 10,
    "asphalt ground": 10,
}


def cls_to_str(x):
    """
    兼容 numpy 字符串、bytes、普通字符串。
    """
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def get_font(size=16, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)

    return ImageFont.load_default()


def find_image_for_npz(npz_path: Path) -> Path | None:
    """
    sam3_label 和 images 目录保持相同相对路径。
    比如：
        sam3_label/video1/000001.npz
    对应：
        images/video1/000001.jpg / png / jpeg
    """
    rel = npz_path.relative_to(LABEL_ROOT).with_suffix("")

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img_path = IMAGE_ROOT / rel.with_suffix(ext)
        if img_path.exists():
            return img_path

    return None


def color_for_class(cls: str, idx: int):
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def is_background(cls: str) -> bool:
    return cls in BACKGROUND_CLASSES


def resize_mask_to_image(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask.astype(bool)

    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
    return np.array(mask_img) > 0


def compute_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    使用 intersection / min(area_a, area_b) 判断两个 mask 是否基本是同一个物体。
    这个比 IoU 更适合处理一个 mask 被另一个 mask 包住的情况。
    """
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())

    if area_a == 0 or area_b == 0:
        return 0.0

    inter = int(np.logical_and(mask_a, mask_b).sum())
    return inter / max(1, min(area_a, area_b))


def deduplicate_instances(classes, confs, masks, overlap_thr=0.60):
    """
    去掉同一物体的重复显示。

    当前规则：
        1. 先区分前景/背景，背景和前景不互相压制；
        2. 再看置信度，谁置信度高保留谁；
        3. 置信度接近或相同时，再看类别优先级；
        4. 最后看面积。

    注意：
        这里只影响可视化，不修改原始 npz 文件。
    """
    candidates = []

    for i, mask in enumerate(masks):
        if not mask.any():
            continue

        cls = cls_to_str(classes[i])
        conf = float(confs[i])
        area = int(mask.sum())
        priority = CLASS_PRIORITY.get(cls, 50)
        bg = is_background(cls)

        candidates.append(
            {
                "idx": i,
                "cls": cls,
                "conf": conf,
                "area": area,
                "priority": priority,
                "mask": mask,
                "is_bg": bg,
            }
        )

    # 前景优先，然后置信度，再类别优先级，再面积
    candidates.sort(
        key=lambda x: (
            0 if x["is_bg"] else 1,
            x["conf"],
            x["priority"],
            x["area"],
        ),
        reverse=True,
    )

    kept = []
    suppressed = []

    for cand in candidates:
        duplicated = False

        for keep in kept:
            # 背景和前景不互相压制，避免 road 把车压掉，或者车把 road 压掉
            if cand["is_bg"] != keep["is_bg"]:
                continue

            overlap = mask_overlap_ratio(cand["mask"], keep["mask"])

            if overlap >= overlap_thr:
                duplicated = True
                suppressed.append(
                    {
                        "drop_idx": cand["idx"],
                        "drop_cls": cand["cls"],
                        "drop_conf": cand["conf"],
                        "keep_idx": keep["idx"],
                        "keep_cls": keep["cls"],
                        "keep_conf": keep["conf"],
                        "overlap": overlap,
                    }
                )
                break

        if not duplicated:
            kept.append(cand)

    # 按原始 instance id 排序，方便和 npz 对应
    kept = sorted(kept, key=lambda x: x["idx"])
    kept_indices = [x["idx"] for x in kept]

    return kept_indices, suppressed


def shorten_text(text, max_len=60):
    text = str(text)
    if len(text) <= max_len:
        return text
    return "..." + text[-max_len:]


def draw_small_badge(draw, x, y, text, color, font, img_w, img_h):
    """
    图上只画小编号，不画类别和置信度，避免遮挡。
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 6
    pad_y = 3

    x = max(4, min(x, img_w - tw - 2 * pad_x - 4))
    y = max(4, min(y, img_h - th - 2 * pad_y - 4))

    box = [
        x,
        y,
        x + tw + 2 * pad_x,
        y + th + 2 * pad_y,
    ]

    draw.rounded_rectangle(box, radius=4, fill=(0, 0, 0), outline=color, width=2)
    draw.text((x + pad_x, y + pad_y - 1), text, fill=color, font=font)


def draw_panel_text(draw, xy, text, font, fill=(230, 230, 230)):
    draw.text(xy, text, fill=fill, font=font)


def overlay_one(
    npz_path: Path,
    frame_idx: int,
    total_frames: int,
    alpha: float = 0.38,
    max_side: int = 960,
    panel_width: int = 430,
    dedup_overlap: float = 0.60,
    no_dedup: bool = False,
    show_label_on_image: bool = False,
):
    """
    返回：
        vis: 可视化图片
        rows: 当前 npz 中保留下来的实例信息，用于写 csv
    """
    img_path = find_image_for_npz(npz_path)

    if img_path is None:
        print(f"[Skip] 找不到原图: {npz_path}")
        return None, []

    data = np.load(npz_path, allow_pickle=True)
    classes = data["classes"]
    confs = data["confidences"]
    masks = data["masks"]

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    h, w = img_np.shape[:2]

    # 1. resize 所有 mask
    resized_masks = []

    for mask in masks:
        mask = resize_mask_to_image(mask, w, h)
        resized_masks.append(mask)

    # 2. 同一物体重复识别时，只保留一个用于显示
    if no_dedup:
        keep_indices = [i for i, m in enumerate(resized_masks) if m.any()]
        suppressed = []
    else:
        keep_indices, suppressed = deduplicate_instances(
            classes=classes,
            confs=confs,
            masks=resized_masks,
            overlap_thr=dedup_overlap,
        )

    rel_npz = npz_path.relative_to(LABEL_ROOT)

    try:
        rel_img = img_path.relative_to(IMAGE_ROOT)
    except Exception:
        rel_img = img_path

    # 3. 叠加 mask
    # 背景先画；前景后画，避免 road 盖住车
    def draw_order_key(i):
        cls = cls_to_str(classes[i])
        area = int(resized_masks[i].sum())

        bg_order = 0 if is_background(cls) else 1
        return bg_order, area

    draw_indices = sorted(keep_indices, key=draw_order_key)

    overlay = img_np.copy()

    for i in draw_indices:
        mask = resized_masks[i]

        if not mask.any():
            continue

        cls = cls_to_str(classes[i])
        color = np.array(color_for_class(cls, i), dtype=np.float32)

        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    vis_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(vis_img)

    font_badge = get_font(14, bold=True)
    font_small = get_font(13)

    instance_infos = []

    # 4. 图上画 bbox + 小编号
    # 默认只写 #id，不写 class/conf，避免遮挡目标
    for i in keep_indices:
        mask = resized_masks[i]

        if not mask.any():
            continue

        cls = cls_to_str(classes[i])
        conf = float(confs[i])
        color = color_for_class(cls, i)
        bbox = compute_bbox(mask)
        area = int(mask.sum())

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if show_label_on_image:
            badge_text = f"#{i} {cls} {conf:.2f}"
            badge_font = font_small
        else:
            badge_text = f"#{i}"
            badge_font = font_badge

        draw_small_badge(
            draw=draw,
            x=x1 + 4,
            y=y1 + 4,
            text=badge_text,
            color=color,
            font=badge_font,
            img_w=w,
            img_h=h,
        )

        instance_infos.append(
            {
                "frame_idx": frame_idx,
                "npz_path": str(rel_npz),
                "image_path": str(rel_img),
                "instance_id": i,
                "class": cls,
                "confidence": conf,
                "area": area,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
            }
        )

    # 5. 缩放图像部分
    scale = min(max_side / max(w, h), 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        vis_img = vis_img.resize((new_w, new_h), resample=Image.BILINEAR)

    # 6. 右侧信息面板
    img_w, img_h = vis_img.size
    canvas_w = img_w + panel_width

    # 改动 1：右侧面板高度从 420 提高到 650，避免 Objects shown 列表显示不全
    canvas_h = max(img_h, 650)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (25, 25, 25))
    canvas.paste(vis_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    font_title = get_font(18, bold=True)
    font = get_font(15)
    font_small = get_font(13)

    panel_x = img_w + 16
    y = 16

    draw_panel_text(
        draw,
        (panel_x, y),
        "SAM3 Check Result",
        font_title,
        fill=(255, 255, 255),
    )
    y += 34

    draw_panel_text(
        draw,
        (panel_x, y),
        f"Frame: {frame_idx}/{total_frames}",
        font,
        fill=(230, 230, 230),
    )
    y += 24

    draw_panel_text(
        draw,
        (panel_x, y),
        f"Shown objects: {len(instance_infos)}",
        font,
        fill=(230, 230, 230),
    )
    y += 24

    draw_panel_text(
        draw,
        (panel_x, y),
        f"Suppressed duplicates: {len(suppressed)}",
        font,
        fill=(230, 230, 230),
    )
    y += 32

    draw_panel_text(draw, (panel_x, y), "NPZ:", font, fill=(255, 255, 255))
    y += 22

    draw_panel_text(
        draw,
        (panel_x, y),
        shorten_text(rel_npz, 48),
        font_small,
        fill=(190, 190, 190),
    )
    y += 34

    draw_panel_text(draw, (panel_x, y), "Image:", font, fill=(255, 255, 255))
    y += 22

    draw_panel_text(
        draw,
        (panel_x, y),
        shorten_text(rel_img, 48),
        font_small,
        fill=(190, 190, 190),
    )
    y += 36

    draw.line([(panel_x, y), (canvas_w - 18, y)], fill=(90, 90, 90), width=1)
    y += 14

    draw_panel_text(draw, (panel_x, y), "Objects shown:", font, fill=(255, 255, 255))
    y += 26

    # 右侧列出编号、类别、置信度、面积
    for row in instance_infos:
        # 改动 2：原来是 canvas_h - 120，现在改成 canvas_h - 40，尽量多显示目标
        if y > canvas_h - 40:
            break

        i = row["instance_id"]
        cls = row["class"]
        conf = row["confidence"]
        area = row["area"]
        color = color_for_class(cls, i)

        draw.rectangle(
            [panel_x, y + 3, panel_x + 14, y + 17],
            fill=color,
            outline=(230, 230, 230),
        )

        text = f"#{i:<2} {cls:<22} conf={conf:.3f}"
        draw_panel_text(
            draw,
            (panel_x + 22, y),
            text,
            font_small,
            fill=(230, 230, 230),
        )
        y += 20

        text2 = f"    area={area}"
        draw_panel_text(
            draw,
            (panel_x + 22, y),
            text2,
            font_small,
            fill=(150, 150, 150),
        )
        y += 19

    # 右侧列出被压掉的重复项
    if suppressed:
        y += 6
        draw.line([(panel_x, y), (canvas_w - 18, y)], fill=(90, 90, 90), width=1)
        y += 12

        draw_panel_text(
            draw,
            (panel_x, y),
            "Suppressed:",
            font,
            fill=(255, 180, 180),
        )
        y += 24

        for item in suppressed[:8]:
            if y > canvas_h - 24:
                break

            # 改动 3：显示 drop 和 keep 的置信度
            text = (
                f"drop #{item['drop_idx']} {item['drop_cls']}({item['drop_conf']:.3f}) "
                f"-> keep #{item['keep_idx']} {item['keep_cls']}({item['keep_conf']:.3f}) "
                f"ov={item['overlap']:.2f}"
            )

            draw_panel_text(
                draw,
                (panel_x, y),
                text,
                font_small,
                fill=(200, 200, 200),
            )
            y += 19

    return canvas, instance_infos


def make_grid(images, cols: int = 3, pad: int = 12, bg=(20, 20, 20)):
    if not images:
        return None

    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    rows = (len(images) + cols - 1) // cols

    grid_w = cols * max_w + (cols + 1) * pad
    grid_h = rows * max_h + (rows + 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), bg)

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols

        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + pad)

        grid.paste(im, (x, y))

    return grid


def save_grid_page(page_images, page_idx, cols, out_dir):
    grid = make_grid(page_images, cols=cols)

    if grid is None:
        return

    out_path = out_dir / f"grid_page_{page_idx:04d}.jpg"
    grid.save(out_path, quality=95)

    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cols", type=int, default=3, help="拼图列数")
    parser.add_argument("--alpha", type=float, default=0.38, help="mask 透明度")
    parser.add_argument("--max-side", type=int, default=760, help="单张原图最长边")
    parser.add_argument("--panel-width", type=int, default=430, help="右侧信息面板宽度")
    parser.add_argument("--page-size", type=int, default=12, help="每张拼图包含多少帧")
    parser.add_argument("--limit", type=int, default=-1, help="只处理前 N 张；-1 表示全部")
    parser.add_argument("--no-grid", action="store_true", help="不生成拼图，只保存单张")

    parser.add_argument(
        "--dedup-overlap",
        type=float,
        default=0.60,
        help="同一物体重复 mask 的重叠阈值，越低越容易去重",
    )

    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="不做重复目标去重，完整显示 npz 里的所有 mask",
    )

    parser.add_argument(
        "--show-label-on-image",
        action="store_true",
        help="在图上显示完整类别和置信度；默认只显示 #编号，避免遮挡",
    )

    args = parser.parse_args()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    single_dir = SAVE_ROOT / "all_single"
    grid_dir = SAVE_ROOT / "all_grid"

    single_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    all_npz = sorted(LABEL_ROOT.rglob("*.npz"))

    if args.limit > 0:
        all_npz = all_npz[:args.limit]

    total = len(all_npz)

    print(f"[Info] 找到 npz 数量: {total}")
    print(f"[Info] 单张结果保存目录: {single_dir}")
    print(f"[Info] 拼图结果保存目录: {grid_dir}")

    if not all_npz:
        raise RuntimeError(f"没有找到 npz: {LABEL_ROOT}")

    csv_path = SAVE_ROOT / "all_instances_index.csv"

    class_counter = Counter()
    page_images = []
    page_idx = 1

    fieldnames = [
        "frame_idx",
        "npz_path",
        "image_path",
        "instance_id",
        "class",
        "confidence",
        "area",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    skipped = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(enumerate(all_npz, 1), total=total, desc="Processing", unit="frame")

        for frame_idx, npz_path in pbar:
            rel = npz_path.relative_to(LABEL_ROOT)
            vis, rows = overlay_one(
                npz_path=npz_path,
                frame_idx=frame_idx,
                total_frames=total,
                alpha=args.alpha,
                max_side=args.max_side,
                panel_width=args.panel_width,
                dedup_overlap=args.dedup_overlap,
                no_dedup=args.no_dedup,
                show_label_on_image=args.show_label_on_image,
            )

            if vis is None:
                skipped += 1
                pbar.set_postfix({"skip": skipped})
                continue

            # 保存单张可视化图，保持原始相对目录结构和文件名
            out_single = single_dir / rel.with_suffix(".jpg")
            out_single.parent.mkdir(parents=True, exist_ok=True)
            vis.save(out_single, quality=95)

            # 写 csv，只写当前可视化保留的实例
            for row in rows:
                writer.writerow(row)
                class_counter[row["class"]] += 1

            # 分页拼图
            if not args.no_grid:
                page_images.append(vis)

                if len(page_images) >= args.page_size:
                    save_grid_page(page_images, page_idx, args.cols, grid_dir)
                    page_images = []
                    page_idx += 1

        if not args.no_grid and page_images:
            save_grid_page(page_images, page_idx, args.cols, grid_dir)

    print("\n[Done] 所有单张可视化图已保存:")
    print(single_dir)

    if not args.no_grid:
        print("\n[Done] 分页拼图已保存:")
        print(grid_dir)

    print("\n[Done] 标注索引 CSV 已保存:")
    print(csv_path)

    print("\n[Info] 当前可视化保留类别统计:")
    for k, v in class_counter.most_common():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()