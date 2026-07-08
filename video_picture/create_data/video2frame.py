import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

# =======================
# 全局配置
# =======================
INPUT_DIR = "/home/via/mai/all_video/video"
OUTPUT_DIR = "/home/via/mai/all_video/images"

# 不同数据源配置
CONFIGS = [
    {
        "fps": 2,
        "video_prefix": "fo",
        "mode": "mode1",
    },
    {
        "fps": 2,
        "video_prefix": "all",
        "mode": "mode2",
    },
]


# =======================
# 裁剪区域配置
# =======================
def get_positions(mode, root=None):
    """
    根据模式 / 路径返回裁剪区域
    """
    if mode == "mode1":
        # 第一个脚本
        return {
            "F": "crop=1280:720:0:0",
            "L": "crop=1280:720:1280:0",
            "B": "crop=1280:720:0:720",
            "R": "crop=1280:720:1280:720",
        }

    elif mode == "mode2":
        # 第二个脚本
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
        raise ValueError("未知模式")


# =======================
# 单视频处理
# =======================
def process_video(video_path, video_name, fps, positions):
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    filter_parts = []
    map_args = []

    for idx, (pos, crop_str) in enumerate(positions.items()):
        filter_parts.append(
            f"[0:v]{crop_str},scale=640:360,fps={fps}[out{idx}]"
        )
        map_args.extend([
            "-map", f"[out{idx}]",
            os.path.join(video_output_dir, f"{pos}_%08d.png")
        ])

    filter_complex = "; ".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-an",
        "-loglevel", "error",
    ] + map_args

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, video_name
    except subprocess.CalledProcessError:
        return False, video_path


# =======================
# 收集视频
# =======================
def collect_videos(config):
    video_list = []
    input_dir = INPUT_DIR
    fps = config["fps"]
    prefix = config["video_prefix"]
    mode = config["mode"]

    for root, _, files in os.walk(input_dir):
        positions = get_positions(mode, root)

        for file in files:
            if file.lower().endswith(".mp4") and file.startswith(prefix):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                video_list.append(
                    (video_path, video_name, fps, positions)
                )
    print(f"  配置 prefix={config['video_prefix']}, mode={config['mode']}: {len(video_list)} 个视频")
    return video_list


def _process_one(args):
    """Wrapper for imap_unordered — takes a single tuple."""
    return process_video(*args)


# =======================
# 主入口
# =======================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_start = time.time()

    # 收集
    print("=" * 50)
    print("🔍 扫描视频文件...")
    all_videos = []
    for cfg in CONFIGS:
        vlist = collect_videos(cfg)
        if vlist:
            print(f"  配置 prefix={cfg['video_prefix']}, mode={cfg['mode']}: {len(vlist)} 个视频")
        all_videos.extend(vlist)

    if not all_videos:
        print("⚠️ 未找到任何视频")
        exit(0)

    total = len(all_videos)
    n_processes = min(cpu_count(), total)
    print(f"\n📹 共 {total} 个视频，使用 {n_processes} 进程并行处理")
    print("=" * 50)

    # 处理（每隔一段时间报告进度）
    success, fail = 0, 0
    failed_list = []
    last_report = 0
    REPORT_INTERVAL = 30  # 每隔 30 秒打印一次

    with Pool(processes=n_processes) as pool:
        for ok, name in pool.imap_unordered(_process_one, all_videos):
            if ok:
                success += 1
            else:
                fail += 1
                failed_list.append(name)

            done = success + fail
            now = time.time()
            if now - last_report >= REPORT_INTERVAL or done == total:
                elapsed = now - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"🎬 {done}/{total} ({done/total*100:.1f}%)  "
                      f"成功 {success}  失败 {fail}  "
                      f"速度 {rate:.1f} video/s  "
                      f"耗时 {elapsed/60:.1f}min  剩余 {eta/60:.1f}min",
                      flush=True)
                last_report = now

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
