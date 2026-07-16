"""
将10张测试图片对应的 SAM3 .npz 预测转换为 COCO RLE mask
=========================================================

输入：
- CVAT测试图片：
  /home/via/mai/datasets/cvat_test10

- 原始SAM3标签：
  /home/via/mai/datasets/sam3_single/sam3_label_single

输出：
- /home/via/mai/datasets/cvat_test10_annotations.json

说明：
- 不修改原始SAM3标签。
- 不运行SAM3模型。
- 只进行格式转换。
- 生成的COCO JSON可以通过 CVAT 的 Upload annotations 导入。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ============================================================
# 路径
# ============================================================

CVAT_IMAGE_DIR = Path(
    "/home/via/mai/datasets/cvat_test10"
)

SAM3_LABEL_ROOT = Path(
    "/home/via/mai/datasets/sam3_single/sam3_label_single"
)

OUTPUT_JSON = Path(
    "/home/via/mai/datasets/cvat_test10_annotations.json"
)


# ============================================================
# 统一类别
# ============================================================

CLASS_NAMES = [
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
    class_name: class_id
    for class_id, class_name in enumerate(
        CLASS_NAMES,
        start=1,
    )
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def to_string(value) -> str:
    """兼容 NumPy 字符串和 bytes。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def collect_images(folder: Path) -> list[Path]:
    """收集测试图片。"""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_cvat_filename(
    filename: str,
) -> tuple[str, str]:
    """
    将CVAT测试图片名恢复为原始相对路径。

    例如：

    fo-20240208_100047008_B__00000001.png

    恢复为：

    sequence_folder = fo-20240208_100047008_B
    original_name   = 00000001.png
    """
    if "__" not in filename:
        raise ValueError(
            "测试图片名中缺少 '__'，无法恢复原目录："
            f"{filename}"
        )

    sequence_folder, original_name = filename.split(
        "__",
        maxsplit=1,
    )

    return sequence_folder, original_name


def encode_binary_mask(mask: np.ndarray) -> dict:
    """
    将二维bool mask转换成COCO压缩RLE。

    pycocotools要求Fortran顺序。
    """
    mask_uint8 = np.asfortranarray(
        mask.astype(np.uint8)
    )

    rle = mask_utils.encode(mask_uint8)

    # pycocotools返回的counts是bytes，
    # JSON保存前必须转换成字符串。
    rle["counts"] = rle["counts"].decode("ascii")

    rle["size"] = [
        int(rle["size"][0]),
        int(rle["size"][1]),
    ]

    return rle


def main() -> None:
    if not CVAT_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"测试图片目录不存在: {CVAT_IMAGE_DIR}"
        )

    if not SAM3_LABEL_ROOT.exists():
        raise FileNotFoundError(
            f"SAM3标签目录不存在: {SAM3_LABEL_ROOT}"
        )

    image_paths = collect_images(CVAT_IMAGE_DIR)

    if len(image_paths) != 10:
        raise RuntimeError(
            f"期望10张测试图片，实际找到{len(image_paths)}张"
        )

    coco_images = []
    coco_annotations = []

    annotation_id = 1
    total_instances = 0

    for image_id, image_path in enumerate(
        image_paths,
        start=1,
    ):
        sequence_folder, original_name = (
            parse_cvat_filename(image_path.name)
        )

        original_stem = Path(original_name).stem

        npz_path = (
            SAM3_LABEL_ROOT
            / sequence_folder
            / f"{original_stem}.npz"
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"找不到对应标签: {npz_path}"
            )

        with Image.open(image_path) as image:
            width, height = image.size

        coco_images.append(
            {
                "id": image_id,

                # 必须与CVAT中的图片文件名完全一致
                "file_name": image_path.name,

                "width": width,
                "height": height,
            }
        )

        with np.load(
            npz_path,
            allow_pickle=True,
        ) as data:
            classes = [
                to_string(value)
                for value in np.asarray(
                    data["classes"]
                ).reshape(-1)
            ]

            masks = np.asarray(
                data["masks"],
                dtype=bool,
            )

            confidences = np.asarray(
                data["confidences"],
                dtype=np.float32,
            ).reshape(-1)

        if len(classes) != len(masks):
            raise RuntimeError(
                f"实例数量不一致: {npz_path}, "
                f"classes={len(classes)}, masks={len(masks)}"
            )

        if len(confidences) != len(classes):
            raise RuntimeError(
                f"置信度数量不一致: {npz_path}"
            )

        image_instance_count = 0

        for instance_index, (
            class_name,
            instance_mask,
            confidence,
        ) in enumerate(
            zip(
                classes,
                masks,
                confidences,
            )
        ):
            if class_name not in CLASS_TO_ID:
                raise ValueError(
                    f"发现未知类别: {class_name}, "
                    f"文件: {npz_path}"
                )

            if instance_mask.shape != (height, width):
                raise RuntimeError(
                    f"mask尺寸错误: {npz_path}, "
                    f"instance={instance_index}, "
                    f"mask={instance_mask.shape}, "
                    f"image={(height, width)}"
                )

            area = int(instance_mask.sum())

            # 完全空的mask直接跳过
            if area == 0:
                print(
                    "[Warning] 跳过空mask: "
                    f"{npz_path}, instance={instance_index}"
                )
                continue

            rle = encode_binary_mask(
                instance_mask
            )

            # COCO bbox格式：
            # [x, y, width, height]
            bbox = (
                mask_utils
                .toBbox(rle)
                .astype(float)
                .tolist()
            )

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": (
                        CLASS_TO_ID[class_name]
                    ),
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,

                    # CVAT将RLE实例作为mask导入。
                    # 这里的iscrowd主要用于格式表示，
                    # 并不代表该目标真的是一群物体。
                    "iscrowd": 1,

                    # 保留模型置信度，方便追溯。
                    "attributes": {
                        "sam3_confidence": float(
                            confidence
                        ),
                    },
                }
            )

            annotation_id += 1
            image_instance_count += 1
            total_instances += 1

        print(
            f"[OK] {image_path.name}: "
            f"{image_instance_count} instances"
        )

    coco_categories = [
        {
            "id": CLASS_TO_ID[class_name],
            "name": class_name,
            "supercategory": "",
        }
        for class_name in CLASS_NAMES
    ]

    coco_data = {
        "info": {
            "description": (
                "Original SAM3 prelabels for "
                "CVAT development test"
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            coco_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("[Done]")
    print(f"图片数: {len(coco_images)}")
    print(f"实例数: {total_instances}")
    print(f"输出文件: {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()