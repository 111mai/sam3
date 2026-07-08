import os
import subprocess
import time
from multiprocessing import Pool, cpu_count

# =======================
# 全局配置
# =======================
INPUT_DIR = "/home/via/mai/datasets/images_continuous"
OUTPUT_DIR = "/home/via/mai/datasets/videos_split"

CONFIGS = [
    {
        "video_prefix": "fo",
        "mode": "mode1",
    },
    {
        "video_prefix": "all",
        "mode": "mode2",
    },
]


# =======================
# 裁剪区域配置
# =======================
def get_positions(mode, root=None):
    """
    根据模式 / 路径返回裁剪区域。
    mode1: 2x2 四宫格，原始视频一般是 2560x1440，每个视角 1280x720。
    mode2: 兼容你原来的 all 数据格式。
    """
    if mode == "mode1":
        return {
            "F": "crop=1280:720:0:0",
            "L": "crop=1280:720:1280:0",
            "B": "crop=1280:720:0:720",
            "R": "crop=1280:720:1280:720",
        }

    elif mode == "mode2":
        if root and ("001ff226a8ed" in root or "CRH" in root):
            return {
                "F": "crop=640:360:0:0",
                "L": "crop=640:360:640:0",
                "B": "crop=640:360:0:360",
                "R": "crop=640:360:640:360",
            }
        else:
            return {
                "F":  "crop=640:360:0:0",
                "L1": "crop=640:360:640:0",
                "L2": "crop=640:360:0:360",
                "B":  "crop=640:360:640:360",
                "R2": "crop=640:360:0:720",
                "R1": "crop=640:360:640:720",
            }

    else:
        raise ValueError(f"未知模式: {mode}")


# =======================
# 单视频处理
# =======================
def process_video(video_path, video_name, positions):
    """
    输入一个四宫格视频，输出多个视角视频：
        OUTPUT_DIR/video_name/F.mp4
        OUTPUT_DIR/video_name/L.mp4
        OUTPUT_DIR/video_name/B.mp4
        OUTPUT_DIR/video_name/R.mp4
    """
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    filter_parts = []
    output_args = []

    for idx, (pos, crop_str) in enumerate(positions.items()):
        out_label = f"out{idx}"

        # 裁剪 + 统一缩放到 640x360
        # 如果你想保留原始子画面分辨率，把 ",scale=640:360" 删除即可
        filter_parts.append(
            f"[0:v]{crop_str},scale=640:360,setsar=1[{out_label}]"
        )

        out_path = os.path.join(video_output_dir, f"{pos}.mp4")

        output_args.extend([
            "-map", f"[{out_label}]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ])

    filter_complex = "; ".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-loglevel", "error",
    ] + output_args

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, video_name
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 处理失败: {video_path}")
        print(e.stderr)
        return False, video_path


# =======================
# 收集视频
# =======================
def collect_videos(config):
    video_list = []

    prefix = config["video_prefix"]
    mode = config["mode"]

    for root, _, files in os.walk(INPUT_DIR):
        positions = get_positions(mode, root)

        for file in files:
            if file.lower().endswith(".mp4") and file.startswith(prefix):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]

                video_list.append(
                    (video_path, video_name, positions)
                )

    print(f"配置 prefix={prefix}, mode={mode}: 找到 {len(video_list)} 个视频")
    return video_list


def _process_one(args):
    return process_video(*args)


# =======================
# 主入口
# =======================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_start = time.time()

    print("=" * 50)
    print("🔍 扫描视频文件...")

    all_videos = []
    for cfg in CONFIGS:
        vlist = collect_videos(cfg)
        all_videos.extend(vlist)

    if not all_videos:
        print("⚠️ 未找到任何视频")
        exit(0)

    total = len(all_videos)
    n_processes = min(cpu_count(), total)

    print(f"\n📹 共 {total} 个视频，使用 {n_processes} 进程并行裁剪")
    print("=" * 50)

    success, fail = 0, 0
    failed_list = []

    with Pool(processes=n_processes) as pool:
        for ok, name in pool.imap_unordered(_process_one, all_videos):
            if ok:
                success += 1
            else:
                fail += 1
                failed_list.append(name)

            done = success + fail
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            print(
                f"🎬 {done}/{total} ({done / total * 100:.1f}%)  "
                f"成功 {success}  失败 {fail}  "
                f"速度 {rate:.2f} video/s  "
                f"耗时 {elapsed / 60:.1f}min  "
                f"剩余 {eta / 60:.1f}min",
                flush=True
            )

    elapsed = time.time() - t_start
    mins, secs = divmod(elapsed, 60)

    print("\n" + "=" * 50)
    print(f"✅ 全部完成！成功 {success} 个，失败 {fail} 个")
    print(f"⏱  总耗时: {int(mins)} 分 {secs:.1f} 秒")
    print(f"📁 输出目录: {OUTPUT_DIR}")

    if failed_list:
        print(f"\n失败视频 ({len(failed_list)}):")
        for p in failed_list[:20]:
            print(f"  ⚠️ {p}")
        if len(failed_list) > 20:
            print(f"  ... 还有 {len(failed_list) - 20} 个")