from pathlib import Path
from collections import Counter

ROOT = Path("/home/via/mai/all_video/images_video_view")

rows = []

for video_dir in sorted(ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    for view_dir in sorted(video_dir.iterdir()):
        if not view_dir.is_dir():
            continue

        num = len(list(view_dir.glob("*.png")))
        rows.append((num, video_dir.name, view_dir.name))

rows_sorted = sorted(rows, reverse=True)

print("=" * 80)
print(f"总视频-视角序列数: {len(rows_sorted)}")
print("=" * 80)

print("\n帧数最多的前 50 个序列：")
for num, video, view in rows_sorted[:50]:
    print(f"{num:5d} frames | {video} | {view}")

print("\n帧数分布：")
bins = {
    "<10": 0,
    "10-29": 0,
    "30-49": 0,
    "50-99": 0,
    "100-199": 0,
    ">=200": 0,
}

for num, _, _ in rows_sorted:
    if num < 10:
        bins["<10"] += 1
    elif num < 30:
        bins["10-29"] += 1
    elif num < 50:
        bins["30-49"] += 1
    elif num < 100:
        bins["50-99"] += 1
    elif num < 200:
        bins["100-199"] += 1
    else:
        bins[">=200"] += 1

for k, v in bins.items():
    print(f"{k:8s}: {v}")

report_path = ROOT / "frame_count_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("num_frames\tvideo\tview\n")
    for num, video, view in rows_sorted:
        f.write(f"{num}\t{video}\t{view}\n")

print(f"\n报告已保存: {report_path}")