"""
将原始的非连续帧号图片，重命名为连续的 1, 2, 3, 4, 5...

用法:
    python make_old_images_continuous.py

默认从 /home/via/mai/datasets/images/ 读取，输出到 .../images_continuous/
也可在命令行指定源和目标:
    python make_old_images_continuous.py --src /path/to/source --dst /path/to/target

策略:
    - 每个视角文件夹 (B/F/L/R) 独立处理
    - 按原始帧号从小到大排序，依次重命名为 00000001.png, 00000002.png, ...
    - 不修改原始文件，只把改名后的文件复制到目标目录
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def collect_view_folders(src_dir: Path) -> List[Path]:
    """收集源目录下所有视角文件夹 (B/F/L/R)."""
    folders = sorted(src_dir.iterdir())
    view_folders = [f for f in folders if f.is_dir()]
    if not view_folders:
        print(f"[ERROR] 在 {src_dir} 下没有找到任何文件夹", file=sys.stderr)
        sys.exit(1)
    return view_folders


def sorted_images(folder: Path) -> List[Tuple[int, Path]]:
    """返回 [(frame_number, file_path), ...] 按帧号从小到大排序."""
    pairs = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        # 从文件名提取数字，例如 "00005368.png" -> 5368
        stem = f.stem  # 不含扩展名
        try:
            num = int(stem)
        except ValueError:
            print(f"[WARNING] 跳过无法解析数字的文件: {f.name}")
            continue
        pairs.append((num, f))
    pairs.sort(key=lambda x: x[0])  # 按帧号升序
    return pairs


def rename_continuous(
    src_dir: Path,
    dst_dir: Path,
    dry_run: bool = False,
    digits: int = 8,
) -> None:
    """核心逻辑: 将每个视角文件夹中的图片按帧号排序后连续重命名."""
    view_folders = collect_view_folders(src_dir)
    print(f"找到 {len(view_folders)} 个文件夹待处理\n")

    report_lines: List[str] = []

    for view_folder in view_folders:
        name = view_folder.name
        images = sorted_images(view_folder)
        n = len(images)

        if n == 0:
            print(f"  [{name}] 无图片，跳过")
            continue

        out_dir = dst_dir / name
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        # 检查连续性
        first_num = images[0][0]
        last_num = images[-1][0]
        expected = last_num - first_num + 1
        gaps = expected - n

        # 复制并重命名
        for new_idx, (old_num, old_path) in enumerate(images, start=1):
            new_name = f"{new_idx:0{digits}d}{old_path.suffix}"
            new_path = out_dir / new_name

            if dry_run:
                if new_idx <= 3 or new_idx > n - 3:
                    print(f"  [{name}] {old_path.name} -> {new_name}")
            else:
                shutil.copy2(old_path, new_path)

        # 汇总
        report = (
            f"{name}: {n} 张图片, "
            f"原帧号范围 {first_num}-{last_num}, "
            f"缺失 {gaps} 帧, "
            f"平均间隔 {gaps / n:.1f}"
        )
        report_lines.append(report)
        print(f"  [{name}] ✓ 完成 — {n} 张图片 (原范围 {first_num}-{last_num}, 缺失 {gaps} 帧)")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}共处理 {len(view_folders)} 个文件夹\n")

    # 写入报告
    if not dry_run:
        report_path = dst_dir / "continuous_report.txt"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"报告已写入 {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将非连续帧号图片重命名为连续 1,2,3..."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/via/mai/datasets/images"),
        help="源目录 (包含 B/F/L/R 视角文件夹)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/via/mai/datasets/images_continuous"),
        help="目标目录 (连续命名的图片输出到此处)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际复制文件",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=8,
        help="输出文件名的零填充位数 (默认 8，如 00000001.png)",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"[ERROR] 源目录不存在: {args.src}", file=sys.stderr)
        sys.exit(1)

    print(f"源目录: {args.src}")
    print(f"目标目录: {args.dst}")
    if args.dry_run:
        print("[模式] DRY RUN — 仅预览，不实际写入\n")

    rename_continuous(
        src_dir=args.src,
        dst_dir=args.dst,
        dry_run=args.dry_run,
        digits=args.digits,
    )


if __name__ == "__main__":
    main()
