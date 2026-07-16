"""
准备 CVAT 正式开发集任务：350 张图片 + 原始 SAM3 预标注
=====================================================

输入：
1. development.txt
2. 原始图片
3. 原始 SAM3 .npz 预测

输出：
1. cvat_dev350_images.zip
2. cvat_dev350_prelabels.json
3. filename_mapping.csv
4. prepare_summary.json

说明：
- 不运行 SAM3。
- 不修改原始图片或原始 .npz。
- 图片在 ZIP 内使用扁平化唯一文件名：
    原目录__原文件名
  例如：
    fo-20240208_100047008_B__00000001.png
- COCO 中的 file_name 与 ZIP 中完全一致，便于 CVAT 匹配。
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ============================================================
# 路径配置
# ============================================================

IMAGE_ROOT = Path("/home/via/mai/datasets/images")

SAM3_LABEL_ROOT = Path(
    "/home/via/mai/datasets/sam3_single/sam3_label_single"
)

DEVELOPMENT_LIST = Path(
    "/home/via/mai/datasets/"
    "sam3_benchmark_v1/manifest/development.txt"
)

OUTPUT_ROOT = Path(
    "/home/via/mai/datasets/"
    "sam3_benchmark_v1/cvat_packages/development350"
)

OUTPUT_IMAGES_ZIP = OUTPUT_ROOT / "cvat_dev350_images.zip"
OUTPUT_COCO_JSON = OUTPUT_ROOT / "cvat_dev350_prelabels.json"
OUTPUT_MAPPING_CSV = OUTPUT_ROOT / "filename_mapping.csv"
OUTPUT_SUMMARY_JSON = OUTPUT_ROOT / "prepare_summary.json"

EXPECTED_IMAGE_COUNT = 350


# ============================================================
# 类别定义
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
    for class_id, class_name in enumerate(CLASS_NAMES, start=1)
}


def read_relative_paths() -> list[Path]:
    """读取 development.txt。"""
    if not DEVELOPMENT_LIST.exists():
        raise FileNotFoundError(
            f"development.txt 不存在: {DEVELOPMENT_LIST}"
        )

    relative_paths = []

    with DEVELOPMENT_LIST.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                relative_paths.append(Path(line))

    if len(relative_paths) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"期望 {EXPECTED_IMAGE_COUNT} 张，"
            f"实际读取到 {len(relative_paths)} 张"
        )

    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("development.txt 中存在重复图片路径")

    return relative_paths


def make_cvat_filename(relative_image_path: Path) -> str:
    """
    将：
        sequence/00000001.png
    转成：
        sequence__00000001.png
    """
    if len(relative_image_path.parts) < 2:
        raise ValueError(
            f"图片路径没有序列目录: {relative_image_path}"
        )

    sequence = relative_image_path.parts[0]
    return f"{sequence}__{relative_image_path.name}"


def npz_path_for(relative_image_path: Path) -> Path:
    """获取图片对应的原始 SAM3 .npz 路径。"""
    return (
        SAM3_LABEL_ROOT
        / relative_image_path.parent
        / f"{relative_image_path.stem}.npz"
    )


def normalize_class_name(value) -> str:
    """将 NumPy 字符串或 bytes 转为普通字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def encode_binary_mask(mask: np.ndarray) -> dict:
    """二维 bool mask -> COCO 压缩 RLE。"""
    mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_uint8)

    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    return {
        "size": [
            int(rle["size"][0]),
            int(rle["size"][1]),
        ],
        "counts": counts,
    }


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    for required_path in [
        IMAGE_ROOT,
        SAM3_LABEL_ROOT,
        DEVELOPMENT_LIST,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"必需路径不存在: {required_path}"
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    relative_paths = read_relative_paths()

    coco_images = []
    coco_annotations = []
    mapping_rows = []

    used_cvat_names = set()
    annotation_id = 1
    total_instances = 0
    empty_masks_skipped = 0
    class_instance_counts = {
        class_name: 0 for class_name in CLASS_NAMES
    }

    print(f"[Prepare] development 图片数: {len(relative_paths)}")
    print(f"[Prepare] 输出目录: {OUTPUT_ROOT}")

    # PNG/JPEG 已经压缩，ZIP_STORED 更快，不必重复压缩。
    with zipfile.ZipFile(
        OUTPUT_IMAGES_ZIP,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as image_zip:

        for image_id, relative_image_path in enumerate(
            relative_paths,
            start=1,
        ):
            image_path = IMAGE_ROOT / relative_image_path
            npz_path = npz_path_for(relative_image_path)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"图片不存在: {image_path}"
                )

            if not npz_path.exists():
                raise FileNotFoundError(
                    f"对应 .npz 不存在: {npz_path}"
                )

            cvat_filename = make_cvat_filename(
                relative_image_path
            )

            if cvat_filename in used_cvat_names:
                raise RuntimeError(
                    f"CVAT 文件名冲突: {cvat_filename}"
                )

            used_cvat_names.add(cvat_filename)

            with Image.open(image_path) as image:
                width, height = image.size

            # 直接把原图写进 ZIP，不复制到临时目录。
            image_zip.write(
                filename=image_path,
                arcname=cvat_filename,
            )

            coco_images.append(
                {
                    "id": image_id,
                    "file_name": cvat_filename,
                    "width": int(width),
                    "height": int(height),
                }
            )

            with np.load(
                npz_path,
                allow_pickle=True,
            ) as data:
                classes = [
                    normalize_class_name(value)
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

                prompt_ids = np.asarray(
                    data["prompt_ids"],
                    dtype=np.int16,
                ).reshape(-1)

            instance_count = len(classes)

            # 原推理脚本在“无实例”图片中保存 masks.shape=(0, 0, 0)。
            # 这是合法的空预测；转换为 COCO 前统一规范成 (0, H, W)。
            if instance_count == 0:
                if masks.size != 0:
                    raise RuntimeError(
                        f"无实例文件的 masks 却非空: {npz_path}, "
                        f"shape={masks.shape}"
                    )
                masks = np.empty(
                    (0, height, width),
                    dtype=bool,
                )

            if masks.shape != (
                instance_count,
                height,
                width,
            ):
                raise RuntimeError(
                    f"mask 尺寸异常: {npz_path}\n"
                    f"实际: {masks.shape}\n"
                    f"期望: {(instance_count, height, width)}"
                )

            if len(confidences) != instance_count:
                raise RuntimeError(
                    f"confidences 数量异常: {npz_path}"
                )

            if len(prompt_ids) != instance_count:
                raise RuntimeError(
                    f"prompt_ids 数量异常: {npz_path}"
                )

            imported_instance_count = 0

            for instance_index in range(instance_count):
                class_name = classes[instance_index]
                instance_mask = masks[instance_index]

                if class_name not in CLASS_TO_ID:
                    raise ValueError(
                        f"发现未知类别 {class_name!r}: {npz_path}"
                    )

                area = int(instance_mask.sum())

                if area == 0:
                    empty_masks_skipped += 1
                    print(
                        "[Warning] 跳过空 mask: "
                        f"{relative_image_path}, "
                        f"instance={instance_index}"
                    )
                    continue

                rle = encode_binary_mask(instance_mask)

                bbox = (
                    mask_utils.toBbox(rle)
                    .astype(float)
                    .tolist()
                )

                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CLASS_TO_ID[class_name],
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 1,
                    }
                )

                annotation_id += 1
                total_instances += 1
                imported_instance_count += 1
                class_instance_counts[class_name] += 1

            mapping_rows.append(
                {
                    "image_id": image_id,
                    "cvat_filename": cvat_filename,
                    "relative_image_path": str(
                        relative_image_path
                    ),
                    "relative_npz_path": str(
                        relative_image_path.parent
                        / f"{relative_image_path.stem}.npz"
                    ),
                    "num_prelabel_instances": (
                        imported_instance_count
                    ),
                }
            )

            if image_id % 25 == 0 or image_id == len(relative_paths):
                print(
                    f"[Progress] {image_id}/{len(relative_paths)} "
                    f"images, {total_instances} instances"
                )

    coco_categories = [
        {
            "id": CLASS_TO_ID[class_name],
            "name": class_name,
            "supercategory": "",
        }
        for class_name in CLASS_NAMES
    ]

    coco_payload = {
        "info": {
            "description": (
                "Original SAM3 prelabels for "
                "SAM3 Benchmark development split"
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    with OUTPUT_COCO_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            coco_payload,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    with OUTPUT_MAPPING_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "image_id",
            "cvat_filename",
            "relative_image_path",
            "relative_npz_path",
            "num_prelabel_instances",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    summary = {
        "split": "development",
        "image_count": len(coco_images),
        "annotation_count": len(coco_annotations),
        "category_count": len(coco_categories),
        "empty_masks_skipped": empty_masks_skipped,
        "class_instance_counts": class_instance_counts,
        "images_zip": str(OUTPUT_IMAGES_ZIP),
        "prelabels_json": str(OUTPUT_COCO_JSON),
        "mapping_csv": str(OUTPUT_MAPPING_CSV),
        "images_zip_sha256": sha256_file(
            OUTPUT_IMAGES_ZIP
        ),
        "prelabels_json_sha256": sha256_file(
            OUTPUT_COCO_JSON
        ),
    }

    with OUTPUT_SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("[Prepare Done]")
    print(f"图片数: {len(coco_images)}")
    print(f"实例数: {len(coco_annotations)}")
    print(f"类别数: {len(coco_categories)}")
    print(f"跳过空 mask: {empty_masks_skipped}")
    print(f"图片 ZIP: {OUTPUT_IMAGES_ZIP}")
    print(f"预标注 JSON: {OUTPUT_COCO_JSON}")
    print(f"文件名映射: {OUTPUT_MAPPING_CSV}")
    print(f"汇总: {OUTPUT_SUMMARY_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
