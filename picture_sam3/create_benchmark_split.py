"""
创建 SAM3 Benchmark v1 固定数据划分
===================================

划分原则：
- 按完整采集时段划分。
- 同一采集时段的 B/F/L/R 四个视角必须处于同一个集合。
- 不进行逐图片随机划分，避免相邻帧和多视角场景泄漏。

固定划分：
- development:
    fo-20240208_100047008_B/F/L/R
- test:
    fo-20240208_090047852_B/F/L/R

输出：
1. 更新 annotation_manifest.csv 中的 split 字段。
2. development.txt
3. test.txt
4. split_summary.json
5. 保存原 manifest 备份。
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ============================================================
# 路径配置
# ============================================================

BENCHMARK_ROOT = Path(
    "/home/via/mai/datasets/sam3_benchmark_v1"
)

MANIFEST_ROOT = BENCHMARK_ROOT / "manifest"

MANIFEST_PATH = (
    MANIFEST_ROOT / "annotation_manifest.csv"
)

DEVELOPMENT_TXT = (
    MANIFEST_ROOT / "development.txt"
)

TEST_TXT = (
    MANIFEST_ROOT / "test.txt"
)

SUMMARY_PATH = (
    MANIFEST_ROOT / "split_summary.json"
)

BACKUP_ROOT = (
    MANIFEST_ROOT / "backup"
)


# ============================================================
# 固定划分
# ============================================================

DEVELOPMENT_SESSIONS = {
    "fo-20240208_100047008",
}

TEST_SESSIONS = {
    "fo-20240208_090047852",
}

VALID_VIEWS = {
    "B",
    "F",
    "L",
    "R",
}


def parse_sequence_folder(
    relative_image_path: str,
) -> tuple[str, str, str]:
    """
    从相对图片路径提取：

        sequence_folder:
            fo-20240208_090047852_B

        session_id:
            fo-20240208_090047852

        view:
            B

    参数示例：
        fo-20240208_090047852_B/00000001.png
    """
    path = Path(relative_image_path)

    if len(path.parts) < 2:
        raise ValueError(
            "图片路径不包含序列子目录: "
            f"{relative_image_path}"
        )

    sequence_folder = path.parts[0]

    match = re.fullmatch(
        r"(.+)_([BFLR])",
        sequence_folder,
    )

    if match is None:
        raise ValueError(
            "无法从目录名解析采集时段和视角: "
            f"{sequence_folder}"
        )

    session_id = match.group(1)
    view = match.group(2)

    if view not in VALID_VIEWS:
        raise ValueError(
            f"非法视角: {view}"
        )

    return sequence_folder, session_id, view


def determine_split(session_id: str) -> str:
    """根据采集时段决定数据划分。"""
    in_development = (
        session_id in DEVELOPMENT_SESSIONS
    )

    in_test = (
        session_id in TEST_SESSIONS
    )

    if in_development and in_test:
        raise RuntimeError(
            f"采集时段同时出现在两个集合: {session_id}"
        )

    if in_development:
        return "development"

    if in_test:
        return "test"

    raise ValueError(
        "发现没有配置划分的采集时段: "
        f"{session_id}"
    )


def read_manifest() -> tuple[list[dict], list[str]]:
    """读取原始标注清单。"""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"标注清单不存在: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "annotation_manifest.csv 没有表头"
            )

        fieldnames = list(reader.fieldnames)
        records = list(reader)

    required_fields = {
        "image_id",
        "relative_image_path",
        "relative_npz_path",
        "split",
    }

    missing_fields = (
        required_fields - set(fieldnames)
    )

    if missing_fields:
        raise RuntimeError(
            f"manifest 缺少字段: {sorted(missing_fields)}"
        )

    return records, fieldnames


def backup_manifest() -> Path:
    """更新 manifest 前先保存备份。"""
    BACKUP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_ROOT
        / f"annotation_manifest_before_split_{timestamp}.csv"
    )

    shutil.copy2(
        MANIFEST_PATH,
        backup_path,
    )

    return backup_path


def write_path_list(
    output_path: Path,
    image_paths: list[str],
) -> None:
    """保存相对图片路径列表。"""
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for image_path in image_paths:
            file.write(image_path + "\n")


def main() -> None:
    # 首先检查两个集合没有重复 session
    overlap = (
        DEVELOPMENT_SESSIONS
        & TEST_SESSIONS
    )

    if overlap:
        raise RuntimeError(
            f"development/test session 重复: {sorted(overlap)}"
        )

    records, fieldnames = read_manifest()
    backup_path = backup_manifest()

    split_counter = Counter()
    session_counter = Counter()
    sequence_counter = Counter()
    view_counter = defaultdict(Counter)

    development_paths = []
    test_paths = []

    all_sessions = set()
    split_sessions = defaultdict(set)

    for record in records:
        relative_image_path = (
            record["relative_image_path"].strip()
        )

        (
            sequence_folder,
            session_id,
            view,
        ) = parse_sequence_folder(
            relative_image_path
        )

        split = determine_split(
            session_id
        )

        # 更新 manifest
        record["split"] = split

        split_counter[split] += 1
        session_counter[session_id] += 1
        sequence_counter[sequence_folder] += 1
        view_counter[split][view] += 1

        all_sessions.add(session_id)
        split_sessions[split].add(session_id)

        if split == "development":
            development_paths.append(
                relative_image_path
            )
        elif split == "test":
            test_paths.append(
                relative_image_path
            )
        else:
            raise RuntimeError(
                f"内部错误，非法 split: {split}"
            )

    # ========================================================
    # 防泄漏检查
    # ========================================================

    leaked_sessions = (
        split_sessions["development"]
        & split_sessions["test"]
    )

    if leaked_sessions:
        raise RuntimeError(
            "检测到采集时段泄漏: "
            f"{sorted(leaked_sessions)}"
        )

    expected_sessions = (
        DEVELOPMENT_SESSIONS
        | TEST_SESSIONS
    )

    unconfigured_sessions = (
        all_sessions - expected_sessions
    )

    missing_sessions = (
        expected_sessions - all_sessions
    )

    if unconfigured_sessions:
        raise RuntimeError(
            "存在未配置的采集时段: "
            f"{sorted(unconfigured_sessions)}"
        )

    if missing_sessions:
        raise RuntimeError(
            "配置中的采集时段在数据中不存在: "
            f"{sorted(missing_sessions)}"
        )

    # 图片不能在两个列表中重复
    duplicated_images = (
        set(development_paths)
        & set(test_paths)
    )

    if duplicated_images:
        raise RuntimeError(
            "发现图片同时出现在 development 和 test 中"
        )

    # 检查 manifest 中是否有重复路径
    all_image_paths = (
        development_paths + test_paths
    )

    if len(all_image_paths) != len(set(all_image_paths)):
        duplicates = [
            path
            for path, count
            in Counter(all_image_paths).items()
            if count > 1
        ]

        raise RuntimeError(
            "manifest 中存在重复图片路径，例如: "
            f"{duplicates[:10]}"
        )

    # ========================================================
    # 保存更新后的 manifest
    # ========================================================

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    # 按路径排序，确保每次运行结果一致
    development_paths = sorted(
        development_paths
    )

    test_paths = sorted(
        test_paths
    )

    write_path_list(
        DEVELOPMENT_TXT,
        development_paths,
    )

    write_path_list(
        TEST_TXT,
        test_paths,
    )

    summary = {
        "benchmark_version": "sam3_benchmark_v1",

        "split_policy": (
            "Split by complete acquisition session. "
            "All B/F/L/R views from the same session "
            "remain in the same split."
        ),

        "development_sessions": sorted(
            DEVELOPMENT_SESSIONS
        ),

        "test_sessions": sorted(
            TEST_SESSIONS
        ),

        "total_images": len(records),

        "split_counts": {
            split_name: int(count)
            for split_name, count
            in sorted(split_counter.items())
        },

        "session_counts": {
            session_id: int(count)
            for session_id, count
            in sorted(session_counter.items())
        },

        "sequence_counts": {
            sequence_name: int(count)
            for sequence_name, count
            in sorted(sequence_counter.items())
        },

        "view_counts": {
            split_name: {
                view: int(count)
                for view, count
                in sorted(counts.items())
            }
            for split_name, counts
            in sorted(view_counter.items())
        },

        "development_session_ids": sorted(
            split_sessions["development"]
        ),

        "test_session_ids": sorted(
            split_sessions["test"]
        ),

        "session_overlap": sorted(
            leaked_sessions
        ),

        "development_list": str(
            DEVELOPMENT_TXT
        ),

        "test_list": str(
            TEST_TXT
        ),

        "manifest": str(
            MANIFEST_PATH
        ),

        "manifest_backup": str(
            backup_path
        ),
    }

    with SUMMARY_PATH.open(
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
    print("=" * 68)
    print("[Split Done]")
    print(
        f"总图片数: {len(records)}"
    )
    print(
        "Development: "
        f"{len(development_paths)}"
    )
    print(
        "Test: "
        f"{len(test_paths)}"
    )
    print()

    print("Development sessions:")
    for session_id in sorted(
        split_sessions["development"]
    ):
        print(
            f"  {session_id}: "
            f"{session_counter[session_id]}"
        )

    print()

    print("Test sessions:")
    for session_id in sorted(
        split_sessions["test"]
    ):
        print(
            f"  {session_id}: "
            f"{session_counter[session_id]}"
        )

    print()

    print("Development views:")
    for view in sorted(
        view_counter["development"]
    ):
        print(
            f"  {view}: "
            f"{view_counter['development'][view]}"
        )

    print()

    print("Test views:")
    for view in sorted(
        view_counter["test"]
    ):
        print(
            f"  {view}: "
            f"{view_counter['test'][view]}"
        )

    print()
    print(
        f"Session overlap: {sorted(leaked_sessions)}"
    )
    print(
        f"更新后的 manifest: {MANIFEST_PATH}"
    )
    print(
        f"Development list: {DEVELOPMENT_TXT}"
    )
    print(
        f"Test list: {TEST_TXT}"
    )
    print(
        f"划分报告: {SUMMARY_PATH}"
    )
    print(
        f"原 manifest 备份: {backup_path}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()