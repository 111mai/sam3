"""
SAM3 批量视频语义分割 / 追踪脚本
================================
功能：
- 输入：一个目录下的多个 .mp4 视频，例如 B_000.mp4、F_000.mp4、L_000.mp4、R_000.mp4
- 每个视频单独追踪，互不共享 tracking memory
- 输出：每个视频对应一个文件夹，每帧保存一个 .npz 文件
- 同时使用 Ultralytics/SAM3 官方 save=True 保存可视化结果
- 保存内容：frame_idx, classes, confidences, track_ids, masks
"""

import os
import warnings
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import clip
from tqdm import tqdm

# 修复 clip tokenizer 兼容问题
clip.simple_tokenizer.SimpleTokenizer = lambda: clip.tokenize

from ultralytics.models.sam import SAM3VideoSemanticPredictor


# =======================
# 全局配置
# =======================

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore")

# 关闭 Ultralytics 每帧日志输出
logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
except Exception:
    pass


# 输入：你刚刚切好的 30s 视频目录
INPUT_VIDEO_DIR = "/home/via/mai/datasets/images_continuous/videos_split/fo-20240212_150156736/chunks_30s"

# 输出：SAM3 tracking 标签目录
OUTPUT_ROOT = "/home/via/mai/datasets/sam3_track_label_continuous"

# SAM3 权重路径
MODEL_PATH = "/home/via/sam3/sam3.pt"

# 使用哪张 GPU
DEVICE = "cuda:0"

# 是否跳过已经处理过的视频
# False：强制重新跑，会覆盖同名 npz，并重新生成官方可视化
# True ：如果该视频目录里已有 npz，则跳过，避免重复跑
SKIP_EXISTING = False

# 目标级置信度阈值
CONF_THRES = 0.5

# mask 二值化阈值
MASK_THRES = 0.5

# 视频抽帧步长
# 1 表示每帧都处理
# 如果显存爆了或太慢，可以改成 2 / 3 / 5
VID_STRIDE = 1


TEXT_PROMPTS = [
    "road",
    "a construction vehicle",
    "a truck",
    "a pickup",
    "a bus",
    "a car",
    "a motorcycle",
    "a bicycle",
    "a rider",
    "a pedestrian",
]

SAVE_NAMES = [
    "road",
    "construction vehicle",
    "truck",
    "car",
    "bus",
    "car",
    "motorcycle",
    "bicycle",
    "rider",
    "pedestrian",
]


def get_video_total_frames(video_path: Path):
    """
    获取视频总帧数，并根据 VID_STRIDE 估计实际处理帧数。

    返回：
        int 或 None
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return None

    processed_frames = (total_frames + VID_STRIDE - 1) // VID_STRIDE
    return processed_frames


def safe_class_names(cls_indices):
    """
    将 SAM3 输出的类别索引转成保存用类别名。
    防止意外索引越界。
    """
    names = []

    for i in cls_indices:
        if 0 <= i < len(SAVE_NAMES):
            names.append(SAVE_NAMES[i])
        else:
            names.append(f"unknown_{i}")

    return names


def make_predictor(output_dir: Path):
    """
    创建一个新的 SAM3VideoSemanticPredictor。
    每个视频单独创建一次，避免 tracking memory 串到下一个视频。
    同时使用 Ultralytics 官方 save=True 保存可视化结果。
    """

    overrides = dict(
        conf=CONF_THRES,
        task="segment",
        mode="predict",
        model=MODEL_PATH,
        half=True,

        # 官方可视化
        save=True,
        show=False,

        device=DEVICE,

        # 关键：关闭 Ultralytics 每帧输出
        verbose=False,

        vid_stride=VID_STRIDE,

        # 官方可视化输出路径：
        # 最终会保存到 output_dir / official_vis
        project=str(output_dir),
        name="official_vis",
        exist_ok=True,
    )

    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    return predictor


def process_one_video(video_path: Path, input_root: Path, output_root: Path):
    """
    对单个视频做 SAM3 视频追踪，并保存每帧 npz。
    """

    rel_path = video_path.relative_to(input_root)
    rel_parent = rel_path.parent
    video_stem = video_path.stem

    # 例如：
    # 输入 INPUT_ROOT/B/B_000.mp4
    # 输出 OUTPUT_ROOT/B/B_000/000000.npz
    output_dir = output_root / rel_parent / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    official_vis_dir = output_dir / "official_vis"

    if SKIP_EXISTING:
        existing_npz = list(output_dir.glob("*.npz"))
        if len(existing_npz) > 0:
            print(f"[Skip] {video_path} 已有 {len(existing_npz)} 个 npz")
            return True

    total_frames = get_video_total_frames(video_path)

    print(f"\n[Video] 开始处理: {video_path}")
    print(f"[NPZ  ] 输出目录: {output_dir}")
    print(f"[VIS  ] 官方可视化目录: {official_vis_dir}")
    print(f"[Info ] 预计处理帧数: {total_frames if total_frames is not None else 'Unknown'}")

    predictor = None

    try:
        predictor = make_predictor(output_dir)

        with torch.inference_mode():
            results = predictor(
                source=str(video_path),
                text=TEXT_PROMPTS,
                stream=True,
            )

            frame_count = 0

            progress_bar = tqdm(
                results,
                total=total_frames,
                desc=video_stem,
                unit="frame",
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            for frame_idx, r in enumerate(progress_bar):
                out_npz = output_dir / f"{frame_idx:06d}.npz"

                if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
                    cls_indices = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy().astype(np.float16)

                    masks_prob = r.masks.data.cpu().numpy().astype(np.float16)
                    masks_bool = masks_prob > MASK_THRES
                    class_names = safe_class_names(cls_indices)

                    if hasattr(r.boxes, "id") and r.boxes.id is not None:
                        track_ids = r.boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.full(len(class_names), -1, dtype=int)

                else:
                    class_names = []
                    confs = np.array([], dtype=np.float16)
                    masks_bool = np.array([], dtype=bool)
                    track_ids = np.array([], dtype=int)

                np.savez_compressed(
                    out_npz,
                    frame_idx=np.array(frame_idx),
                    classes=np.array(class_names),
                    confidences=confs,
                    track_ids=track_ids,
                    masks=masks_bool,
                )

                frame_count += 1

        print(f"[Done] {video_path.name} 完成，共保存 {frame_count} 帧")
        print(f"[VIS ] 官方可视化结果请查看: {official_vis_dir}")

        return True

    except RuntimeError as e:
        print(f"\n[CUDA/Runtime Error] 处理失败: {video_path}")
        print(e)
        return False

    except Exception as e:
        print(f"\n[Error] 处理失败: {video_path}")
        print(e)
        return False

    finally:
        # 释放当前视频的 predictor 和显存缓存
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    input_root = Path(INPUT_VIDEO_DIR)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"输入视频目录不存在: {input_root}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"SAM3 权重不存在: {MODEL_PATH}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("没有检测到 CUDA GPU，请先检查 PyTorch CUDA 环境")

    video_paths = sorted(input_root.rglob("*.mp4"))

    if len(video_paths) == 0:
        print(f"[Warn] 没有在 {input_root} 下找到 mp4 视频")
        return

    print("=" * 60)
    print(f"[Main] 输入目录: {input_root}")
    print(f"[Main] 输出目录: {output_root}")
    print(f"[Main] 共找到 {len(video_paths)} 个视频")
    print(f"[Main] 使用设备: {DEVICE}")
    print(f"[Main] vid_stride={VID_STRIDE}")
    print(f"[Main] skip_existing={SKIP_EXISTING}")
    print("=" * 60)

    success = 0
    fail = 0
    failed_videos = []

    for idx, video_path in enumerate(video_paths, start=1):
        print(f"\n[{idx}/{len(video_paths)}] {video_path}")

        ok = process_one_video(video_path, input_root, output_root)

        if ok:
            success += 1
        else:
            fail += 1
            failed_videos.append(str(video_path))

    print("\n" + "=" * 60)
    print("[Finish] 全部完成")
    print(f"[Finish] 成功: {success}")
    print(f"[Finish] 失败: {fail}")
    print(f"[Finish] 输出目录: {output_root}")

    if failed_videos:
        print("\n失败视频:")
        for p in failed_videos:
            print(f"  - {p}")


if __name__ == "__main__":
    main()