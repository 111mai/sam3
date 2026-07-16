"""
将 CVAT 导出的 COCO 标注转换成统一 GT
=====================================

输入：
    CVAT 导出的 instances_default.json

输出：
1. 每张图片一个实例级 .npz
2. 每张图片一个单通道语义标签 PNG

类别编号：
0   background
1   road
2   construction vehicle
3   truck
4   car
5   bus
6   motorcycle
7   bicycle
8   rider
9   pedestrian
255 ignore

说明：
- road 与前景物体重叠时，前景物体覆盖 road。
- 两个不同前景类别发生重叠时，冲突像素设为 255。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


# ============================================================
# 路径
# ============================================================

COCO_JSON = Path(
    "/home/via/Downloads/"
    "task_2_annotations_2026_07_13_05_41_23_coco 1.0/"
    "annotations/instances_default.json"
)

IMAGE_ROOT = Path(
    "/home/via/mai/datasets/images"
)

OUTPUT_ROOT = Path(
    "/home/via/mai/datasets/"
    "sam3_benchmark_v1/cvat_roundtrip_test10"
)

INSTANCE_ROOT = OUTPUT_ROOT / "gt_instances"
SEMANTIC_ROOT = OUTPUT_ROOT / "gt_semantic"


# ============================================================
# 统一类别
# ============================================================

CLASS_NAMES = [
    "background",
    "road",
    "construction vehicle",
    "truck",
    "car",
    "bus",
    "motorcycle",
    "bicycle",
    "rider",
    "pedestrian",
]

CLASS_TO_ID = {
    name: class_id
    for class_id, name in enumerate(CLASS_NAMES)
}

IGNORE_INDEX = 255


def parse_cvat_filename(
    filename: str,
) -> tuple[str, str]:
    """
    例如：

    fo-20240208_100047008_B__00000001.png

    转换为：

    sequence = fo-20240208_100047008_B
    image_name = 00000001.png
    """
    if "__" not in filename:
        raise ValueError(
            f"无法恢复原始路径，文件名缺少 '__': {filename}"
        )

    sequence, image_name = filename.split("__", maxsplit=1)

    return sequence, image_name


def decode_coco_mask(
    segmentation: dict,
    height: int,
    width: int,
) -> np.ndarray:
    """
    同时支持：

    压缩 RLE：
        counts 为字符串

    非压缩 RLE：
        counts 为整数列表
    """
    counts = segmentation.get("counts")

    if isinstance(counts, list):
        # CVAT 导出的非压缩 RLE
        rle = mask_utils.frPyObjects(
            segmentation,
            height,
            width,
        )
    else:
        # 压缩 RLE
        rle = dict(segmentation)

        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(bool)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    """由 mask 重新计算 xyxy box。"""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.array(
            [0, 0, 0, 0],
            dtype=np.float32,
        )

    return np.array(
        [
            xs.min(),
            ys.min(),
            xs.max() + 1,
            ys.max() + 1,
        ],
        dtype=np.float32,
    )


def build_semantic_map(
    masks: np.ndarray,
    class_ids: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, int]:
    """
    将实例 mask 转成语义标签。

    规则：
    1. background 初始为 0。
    2. road 先写入。
    3. 其他物体覆盖 road。
    4. 不同前景类别冲突时写成 255。
    """
    semantic = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # -------------------------
    # road 先写入
    # -------------------------

    for mask, class_id in zip(masks, class_ids):
        if int(class_id) == 1:
            semantic[mask] = 1

    # -------------------------
    # 其他前景类别
    # -------------------------

    conflict_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    for mask, class_id in zip(masks, class_ids):
        class_id = int(class_id)

        if class_id == 1:
            continue

        current = semantic[mask]

        # 背景、road 或同类别可以直接覆盖
        writable = (
            (current == 0)
            | (current == 1)
            | (current == class_id)
        )

        ys, xs = np.where(mask)

        write_y = ys[writable]
        write_x = xs[writable]

        semantic[write_y, write_x] = class_id

        # 两种不同的前景类别发生重叠
        conflicting = (
            (current != 0)
            & (current != 1)
            & (current != class_id)
            & (current != IGNORE_INDEX)
        )

        conflict_y = ys[conflicting]
        conflict_x = xs[conflicting]

        semantic[conflict_y, conflict_x] = IGNORE_INDEX
        conflict_mask[conflict_y, conflict_x] = True

    return semantic, int(conflict_mask.sum())


def main() -> None:
    if not COCO_JSON.exists():
        raise FileNotFoundError(
            f"COCO文件不存在: {COCO_JSON}"
        )

    INSTANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEMANTIC_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with COCO_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    images = {
        int(item["id"]): item
        for item in coco["images"]
    }

    categories = {
        int(item["id"]): str(item["name"])
        for item in coco["categories"]
    }

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:
        annotations_by_image[
            int(annotation["image_id"])
        ].append(annotation)

    total_instances = 0
    total_conflicts = 0

    for image_id, image_info in sorted(images.items()):
        cvat_filename = str(
            image_info["file_name"]
        )

        height = int(image_info["height"])
        width = int(image_info["width"])

        sequence, original_name = (
            parse_cvat_filename(cvat_filename)
        )

        original_image_path = (
            IMAGE_ROOT
            / sequence
            / original_name
        )

        relative_npz_path = (
            Path(sequence)
            / f"{Path(original_name).stem}.npz"
        )

        relative_png_path = (
            Path(sequence)
            / f"{Path(original_name).stem}.png"
        )

        output_npz = (
            INSTANCE_ROOT / relative_npz_path
        )

        output_png = (
            SEMANTIC_ROOT / relative_png_path
        )

        output_npz.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_png.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        classes = []
        class_ids = []
        masks = []
        boxes = []

        image_annotations = (
            annotations_by_image.get(image_id, [])
        )

        for annotation in image_annotations:
            category_id = int(
                annotation["category_id"]
            )

            if category_id not in categories:
                raise ValueError(
                    f"未知COCO category_id: {category_id}"
                )

            class_name = categories[category_id]

            if class_name not in CLASS_TO_ID:
                raise ValueError(
                    f"未知类别名: {class_name}"
                )

            class_id = CLASS_TO_ID[class_name]

            mask = decode_coco_mask(
                annotation["segmentation"],
                height,
                width,
            )

            if mask.shape != (height, width):
                raise RuntimeError(
                    f"mask尺寸错误: {mask.shape}, "
                    f"expected={(height, width)}"
                )

            if not np.any(mask):
                print(
                    f"[Warning] 跳过空mask: "
                    f"{cvat_filename}, "
                    f"annotation={annotation['id']}"
                )
                continue

            classes.append(class_name)
            class_ids.append(class_id)
            masks.append(mask)
            boxes.append(mask_to_box(mask))

        if masks:
            masks_array = np.stack(
                masks,
                axis=0,
            ).astype(bool)

            boxes_array = np.stack(
                boxes,
                axis=0,
            ).astype(np.float32)
        else:
            masks_array = np.empty(
                (0, height, width),
                dtype=bool,
            )

            boxes_array = np.empty(
                (0, 4),
                dtype=np.float32,
            )

        classes_array = np.asarray(
            classes,
            dtype="<U32",
        )

        class_ids_array = np.asarray(
            class_ids,
            dtype=np.int16,
        )

        semantic, conflict_pixels = (
            build_semantic_map(
                masks=masks_array,
                class_ids=class_ids_array,
                height=height,
                width=width,
            )
        )

        np.savez_compressed(
            output_npz,
            classes=classes_array,
            class_ids=class_ids_array,
            boxes=boxes_array,
            masks=masks_array,
            image_path=str(original_image_path),
            cvat_filename=cvat_filename,
            orig_shape=np.array(
                [height, width],
                dtype=np.int32,
            ),
            source=np.array(
                "cvat_corrected_coco",
                dtype="<U32",
            ),
        )

        success = cv2.imwrite(
            str(output_png),
            semantic,
        )

        if not success:
            raise RuntimeError(
                f"无法保存语义标签: {output_png}"
            )

        total_instances += len(classes_array)
        total_conflicts += conflict_pixels

        print(
            f"[OK] {cvat_filename}: "
            f"instances={len(classes_array)}, "
            f"conflict_pixels={conflict_pixels}"
        )

    print()
    print("=" * 64)
    print("[Conversion Done]")
    print(f"图片数: {len(images)}")
    print(f"实例数: {total_instances}")
    print(f"冲突像素总数: {total_conflicts}")
    print(f"实例GT: {INSTANCE_ROOT}")
    print(f"语义GT: {SEMANTIC_ROOT}")
    print("=" * 64)


if __name__ == "__main__":
    main()