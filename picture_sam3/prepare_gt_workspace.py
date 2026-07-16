"""
建立 SAM3 Benchmark v1 人工标注工作区
====================================

功能：
1. 不修改原始 SAM3 预测结果。
2. 将原始 .npz 转换为独立的人工标注草稿。
3. 保留 SAM3 预标注，方便人工离线修改。
4. 增加统一评测类别 class_ids。
5. 增加人工标注状态和来源字段。
6. 创建 benchmark manifest 和类别定义文件。

注意：
- 本脚本不会加载 SAM3 模型。
- 本脚本不会进行任何模型推理。
- 本脚本不会更新模型权重。
- 原始 SAM3 结果只作为人工标注的初始草稿。
/home/via/mai/datasets/sam3_benchmark_v1/
├── gt_instances_draft/
│   └── 1703 个可以人工修改的 .npz
│
├── gt_instances_final/
│   └── 人工检查完成后才放入这里
│
├── gt_semantic/
│   └── 后续导出的单通道语义 GT
│
├── gt_vis/
│   └── 后续保存人工 GT 可视化
│
├── annotation_backup/
│   └── 每次保存前的备份
│
├── annotation_state/
│
└── manifest/
    ├── annotation_manifest.csv
    ├── class_definition.json
    └── prepare_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# 路径配置
# ============================================================

IMAGE_ROOT = Path(
    "/home/via/mai/datasets/images"
)

RAW_LABEL_ROOT = Path(
    "/home/via/mai/datasets/sam3_single/sam3_label_single"
)

AUDIT_ROOT = Path(
    "/home/via/mai/datasets/sam3_single/audit"
)

BENCHMARK_ROOT = Path(
    "/home/via/mai/datasets/sam3_benchmark_v1"
)

DRAFT_ROOT = BENCHMARK_ROOT / "gt_instances_draft"
FINAL_ROOT = BENCHMARK_ROOT / "gt_instances_final"
SEMANTIC_ROOT = BENCHMARK_ROOT / "gt_semantic"
VIS_ROOT = BENCHMARK_ROOT / "gt_vis"
MANIFEST_ROOT = BENCHMARK_ROOT / "manifest"
STATE_ROOT = BENCHMARK_ROOT / "annotation_state"
BACKUP_ROOT = BENCHMARK_ROOT / "annotation_backup"


# 已存在草稿时是否覆盖
OVERWRITE = False

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# 统一评测类别
# ============================================================

# 最终 mIoU 使用的类别编号
#
# background = 0
# 有效前景类 = 1~9
# ignore = 255
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
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}

IGNORE_INDEX = 255


# SAM3 原始文本提示词编号到统一评测类别编号的映射
#
# prompt_id=3: a pickup -> car=4
# prompt_id=5: a car    -> car=4
PROMPT_ID_TO_CLASS_ID = {
    0: 1,  # road
    1: 2,  # construction vehicle
    2: 3,  # truck
    3: 4,  # pickup -> car
    4: 5,  # bus
    5: 4,  # car
    6: 6,  # motorcycle
    7: 7,  # bicycle
    8: 8,  # rider
    9: 9,  # pedestrian
}


def collect_images(root: Path) -> list[Path]:
    """递归收集所有图片。"""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_audit_hashes() -> dict[str, str]:
    """
    读取上一步生成的原始标签 SHA256。

    返回：
        {
            "相对路径/xxx.npz": "sha256..."
        }
    """
    hash_path = AUDIT_ROOT / "raw_npz_sha256.jsonl"

    if not hash_path.exists():
        print(
            f"[Warning] 未找到原始标签哈希文件: {hash_path}"
        )
        return {}

    records = {}

    with hash_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            records[record["relative_path"]] = record["sha256"]

    return records


def normalize_classes(classes: np.ndarray) -> np.ndarray:
    """
    将类别名转换为固定长度 Unicode。

    避免以后修改类别时因为原数组字符串长度太短导致截断。
    """
    return np.asarray(
        [str(item) for item in classes.reshape(-1)],
        dtype="<U32",
    )


def classes_to_class_ids(classes: np.ndarray) -> np.ndarray:
    """将类别名转换成统一评测类别 ID。"""
    class_ids = []

    for class_name in classes:
        class_name = str(class_name)

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"未知类别名，无法生成 class_id: {class_name}"
            )

        class_id = CLASS_TO_ID[class_name]

        if class_id == 0:
            raise ValueError(
                "实例标签中不应该出现 background 实例"
            )

        class_ids.append(class_id)

    return np.asarray(class_ids, dtype=np.int16)


def normalize_empty_masks(
    masks: np.ndarray,
    image_height: int,
    image_width: int,
    dtype,
) -> np.ndarray:
    """
    将空实例标签统一成：

        (0, H, W)

    原始推理脚本中的空实例可能是：

        (0, 0, 0)
    """
    if masks.shape[0] == 0:
        return np.empty(
            (0, image_height, image_width),
            dtype=dtype,
        )

    return masks.astype(dtype, copy=False)


def create_draft_npz(
    image_path: Path,
    raw_npz_path: Path,
    draft_npz_path: Path,
) -> int:
    """
    从原始 SAM3 预测生成独立人工标注草稿。

    返回：
        实例数量
    """
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    with np.load(raw_npz_path, allow_pickle=True) as raw:
        classes = normalize_classes(
            np.asarray(raw["classes"])
        )

        prompt_ids = np.asarray(
            raw["prompt_ids"],
            dtype=np.int16,
        ).reshape(-1)

        confidences = np.asarray(
            raw["confidences"],
            dtype=np.float16,
        ).reshape(-1)

        boxes = np.asarray(
            raw["boxes"],
            dtype=np.float32,
        ).reshape(-1, 4)

        masks = np.asarray(
            raw["masks"],
            dtype=bool,
        )

        masks_prob = np.asarray(
            raw["masks_prob"],
            dtype=np.float16,
        )

        masks = normalize_empty_masks(
            masks=masks,
            image_height=image_height,
            image_width=image_width,
            dtype=bool,
        )

        masks_prob = normalize_empty_masks(
            masks=masks_prob,
            image_height=image_height,
            image_width=image_width,
            dtype=np.float16,
        )

        class_ids = classes_to_class_ids(classes)

        num_instances = len(classes)

        # 每个实例是否已经被人工修改
        manual_flags = np.zeros(
            num_instances,
            dtype=bool,
        )

        # 每个实例的来源
        #
        # 后续可以改成：
        # - sam3_prelabel
        # - manually_corrected
        # - manually_added
        annotation_source = np.full(
            num_instances,
            "sam3_prelabel",
            dtype="<U32",
        )

        # 创建草稿目录
        draft_npz_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            draft_npz_path,

            # 实例基本信息
            classes=classes,
            class_ids=class_ids,
            prompt_ids=prompt_ids,
            confidences=confidences,
            boxes=boxes,
            masks=masks,

            # 保留原字段，但不要将其作为真实概率使用
            masks_prob=masks_prob,

            # 原始配置和路径
            text_prompts=np.asarray(
                raw["text_prompts"]
            ),
            save_names=np.asarray(
                raw["save_names"]
            ),
            image_path=str(image_path),
            orig_shape=np.array(
                [image_height, image_width],
                dtype=np.int32,
            ),

            # 人工标注相关字段
            manual_flags=manual_flags,
            annotation_source=annotation_source,
            annotation_status=np.array(
                "unreviewed",
                dtype="<U32",
            ),
            benchmark_version=np.array(
                "sam3_benchmark_v1",
                dtype="<U32",
            ),
            prelabel_method=np.array(
                "original_sam3",
                dtype="<U32",
            ),
        )

    return num_instances


def create_class_definition() -> None:
    """保存统一类别定义。"""
    class_definition = {
        "benchmark_version": "sam3_benchmark_v1",
        "background_id": 0,
        "ignore_index": IGNORE_INDEX,
        "classes": [
            {
                "id": class_id,
                "name": class_name,
                "evaluate": class_id != 0,
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
        "prompt_id_to_class_id": {
            str(prompt_id): class_id
            for prompt_id, class_id
            in PROMPT_ID_TO_CLASS_ID.items()
        },
        "important_rules": [
            "prompt_ids 只表示原始 SAM3 文本提示编号。",
            "mIoU 必须使用 class_ids，不能直接使用 prompt_ids。",
            "a pickup 和 a car 都归并为 car 类别。",
            "最终人工 GT 不使用模型置信度作为判断标准。",
            "masks_prob 是二值结果，不是真正的连续概率图。",
            "ignore 区域使用像素值 255。",
        ],
    }

    output_path = (
        MANIFEST_ROOT / "class_definition.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_definition,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"图片目录不存在: {IMAGE_ROOT}"
        )

    if not RAW_LABEL_ROOT.exists():
        raise FileNotFoundError(
            f"原始标签目录不存在: {RAW_LABEL_ROOT}"
        )

    for directory in [
        BENCHMARK_ROOT,
        DRAFT_ROOT,
        FINAL_ROOT,
        SEMANTIC_ROOT,
        VIS_ROOT,
        MANIFEST_ROOT,
        STATE_ROOT,
        BACKUP_ROOT,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    create_class_definition()

    raw_hashes = load_audit_hashes()
    images = collect_images(IMAGE_ROOT)

    manifest_records = []

    total_instances = 0
    created_count = 0
    skipped_count = 0

    print(f"[Prepare] 图片数: {len(images)}")
    print(f"[Prepare] IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"[Prepare] RAW_LABEL_ROOT: {RAW_LABEL_ROOT}")
    print(f"[Prepare] BENCHMARK_ROOT: {BENCHMARK_ROOT}")
    print(f"[Prepare] OVERWRITE: {OVERWRITE}")

    for image_index, image_path in enumerate(
        tqdm(
            images,
            desc="Preparing GT drafts",
            unit="img",
        )
    ):
        relative_image_path = image_path.relative_to(
            IMAGE_ROOT
        )

        relative_npz_path = (
            relative_image_path.parent
            / f"{image_path.stem}.npz"
        )

        raw_npz_path = (
            RAW_LABEL_ROOT / relative_npz_path
        )

        draft_npz_path = (
            DRAFT_ROOT / relative_npz_path
        )

        if not raw_npz_path.exists():
            raise FileNotFoundError(
                f"缺少原始标签: {raw_npz_path}"
            )

        if draft_npz_path.exists() and not OVERWRITE:
            skipped_count += 1

            with np.load(
                draft_npz_path,
                allow_pickle=True,
            ) as existing:
                num_instances = len(
                    existing["classes"]
                )
        else:
            num_instances = create_draft_npz(
                image_path=image_path,
                raw_npz_path=raw_npz_path,
                draft_npz_path=draft_npz_path,
            )

            created_count += 1

        total_instances += num_instances

        raw_relative_string = str(relative_npz_path)

        manifest_records.append(
            {
                "image_id": image_index,
                "relative_image_path": str(
                    relative_image_path
                ),
                "relative_npz_path": str(
                    relative_npz_path
                ),
                "raw_sha256": raw_hashes.get(
                    raw_relative_string,
                    "",
                ),
                "num_prelabel_instances": num_instances,

                # 下一步再划分
                "split": "",

                # 当前均未人工检查
                "annotation_status": "unreviewed",

                # 第二个人复核时使用
                "review_status": "not_reviewed",
            }
        )

    manifest_path = (
        MANIFEST_ROOT / "annotation_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "image_id",
            "relative_image_path",
            "relative_npz_path",
            "raw_sha256",
            "num_prelabel_instances",
            "split",
            "annotation_status",
            "review_status",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_records)

    # 保存简单统计
    summary = {
        "benchmark_version": "sam3_benchmark_v1",
        "total_images": len(images),
        "total_prelabel_instances": total_instances,
        "created_drafts": created_count,
        "skipped_existing_drafts": skipped_count,
        "image_root": str(IMAGE_ROOT),
        "raw_label_root": str(RAW_LABEL_ROOT),
        "draft_root": str(DRAFT_ROOT),
        "final_root": str(FINAL_ROOT),
        "manifest": str(manifest_path),
    }

    summary_path = (
        MANIFEST_ROOT / "prepare_summary.json"
    )

    with summary_path.open(
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
    print("=" * 60)
    print("[Prepare Done]")
    print(f"图片总数: {len(images)}")
    print(f"预标注实例总数: {total_instances}")
    print(f"新建草稿数: {created_count}")
    print(f"跳过已有草稿数: {skipped_count}")
    print(f"Benchmark 根目录: {BENCHMARK_ROOT}")
    print(f"标注清单: {manifest_path}")
    print(
        "类别定义: "
        f"{MANIFEST_ROOT / 'class_definition.json'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()