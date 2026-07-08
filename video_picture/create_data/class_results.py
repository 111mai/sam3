from pathlib import Path
from tqdm import tqdm
import shutil

# 原始抽帧目录
SRC_ROOT = Path("/home/via/mai/all_video/images")

# 新的整理后目录
DST_ROOT = Path("/home/via/mai/all_video/images_video_view")

# 默认用软链接，不复制图片，不额外占 110G
# 可选: "symlink", "copy", "move"
OPERATION = "symlink"

# 是否覆盖已存在文件
OVERWRITE = True

# 允许的视角名
VALID_VIEWS = {"F", "B", "L", "R", "L1", "L2", "R1", "R2"}


def parse_name(img_path: Path):
    """
    解析文件名:
        F_00000001.png  -> view=F, frame=00000001.png
        R1_00000008.png -> view=R1, frame=00000008.png
    """
    stem = img_path.stem

    if "_" not in stem:
        return None, None

    view, frame_id = stem.split("_", 1)

    if view not in VALID_VIEWS:
        return None, None

    if not frame_id.isdigit():
        return None, None

    new_name = f"{int(frame_id):08d}.png"
    return view, new_name


def write_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if OVERWRITE:
            dst.unlink()
        else:
            return

    if OPERATION == "symlink":
        dst.symlink_to(src.resolve())

    elif OPERATION == "copy":
        shutil.copy2(src, dst)

    elif OPERATION == "move":
        shutil.move(str(src), str(dst))

    else:
        raise ValueError(f"未知 OPERATION: {OPERATION}")


def main():
    assert SRC_ROOT.exists(), f"源目录不存在: {SRC_ROOT}"

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()])

    print("=" * 60)
    print(f"源目录: {SRC_ROOT}")
    print(f"输出目录: {DST_ROOT}")
    print(f"操作方式: {OPERATION}")
    print(f"视频目录数量: {len(video_dirs)}")
    print("=" * 60)

    total_images = 0
    total_videos = 0
    total_skipped = 0

    report = []

    for video_dir in tqdm(video_dirs, desc="整理视频"):
        video_name = video_dir.name
        video_count = 0
        view_counts = {}

        img_paths = sorted(video_dir.glob("*.png"))

        for img_path in img_paths:
            view, new_name = parse_name(img_path)

            if view is None:
                total_skipped += 1
                continue

            dst_path = DST_ROOT / video_name / view / new_name
            write_file(img_path, dst_path)

            video_count += 1
            view_counts[view] = view_counts.get(view, 0) + 1

        if video_count > 0:
            total_videos += 1
            total_images += video_count
            report.append((video_name, view_counts, video_count))

    report_path = DST_ROOT / "reorganize_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"SRC_ROOT: {SRC_ROOT}\n")
        f.write(f"DST_ROOT: {DST_ROOT}\n")
        f.write(f"OPERATION: {OPERATION}\n")
        f.write(f"total_videos: {total_videos}\n")
        f.write(f"total_images: {total_images}\n")
        f.write(f"total_skipped: {total_skipped}\n\n")

        for video_name, view_counts, video_count in report:
            f.write(f"{video_name}\t总图片数={video_count}\t")
            f.write(", ".join([f"{k}:{v}" for k, v in sorted(view_counts.items())]))
            f.write("\n")

    print("\n整理完成！")
    print(f"有效视频数: {total_videos}")
    print(f"处理图片数: {total_images}")
    print(f"跳过文件数: {total_skipped}")
    print(f"报告文件: {report_path}")

    if OPERATION == "symlink":
        print("\n当前使用的是软链接，不会再复制 110G 图片。")


if __name__ == "__main__":
    main()