"""
SAM3 语义分割推理脚本 v2
========================

功能：
- 使用 SAM3 对图片进行单帧语义/实例分割。
- 输入：INPUT_FOLDER 下的所有 .jpg / .jpeg / .png 图片。
- 输出：
  1. .npz 结构化标签文件：
     classes, prompt_ids, confidences, boxes, masks, masks_prob, text_prompts, save_names, image_path, orig_shape
  2. 官方 Ultralytics 可视化结果：
     每个输出目录下的 _vis/*.jpg

说明：
- 官方可视化只用于肉眼检查。
- .npz 文件用于后续 CoCo-SAM3 复现、semantic map 生成、mIoU 计算、类别冲突分析。
- 多 GPU 并行处理，每个 GPU 独立加载一份 SAM3 模型。
"""

import warnings
import logging
import os
from pathlib import Path
import time
import queue

import numpy as np
import torch
import torch.multiprocessing as mp
import clip
from tqdm import tqdm

# monkey-patch:
# SAM3 期望 SimpleTokenizer() 可调用，但 clip>=1.0 的 SimpleTokenizer 不可调用
# 用 clip.tokenize 替代，接口兼容
clip.simple_tokenizer.SimpleTokenizer = lambda: clip.tokenize

from ultralytics.models.sam import SAM3SemanticPredictor


# =========================
# 基本配置
# =========================

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

INPUT_FOLDER = "/home/via/mai/datasets/images_continuous"
OUTPUT_FOLDER = "/home/via/mai/datasets/sam3_label_continuous"

# 调试阶段先跑 100 张，确认输出没问题后改成 None 跑全量
MAX_IMAGES = None

# 是否保存官方可视化 overlay
SAVE_VIS = True

# 是否跳过已经存在的 npz
SKIP_EXISTING = False

# SAM3 权重路径
MODEL_PATH = str(Path(__file__).parent.parent / "sam3.pt")

# 文本提示词列表
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

# 存到 .npz 里的类别名
# 注意：a pickup 被归并为 car
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


def gpu_worker(gpu_id, image_paths, root_folder, output_root, progress_queue):
    """
    单个 GPU 进程：
    - 加载一份 SAM3 模型
    - 处理分配给该 GPU 的图片
    - 保存 .npz 和官方可视化图
    """
    if not image_paths:
        return

    device = f"cuda:{gpu_id}"

    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model=MODEL_PATH,
        half=True,
        save=False,   # 不自动保存到 runs/，后面手动 result.save(filename=...)
        device=device,
        verbose=False,
    )

    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print(f"[GPU {gpu_id}] 模型加载成功，分配图片数: {len(image_paths)}")
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return

    with torch.no_grad():
        for img_path in image_paths:
            try:
                relative_path = img_path.relative_to(root_folder).parent
                save_dir = output_root / relative_path
                save_dir.mkdir(parents=True, exist_ok=True)

                out_npz = save_dir / f"{img_path.stem}.npz"

                if SKIP_EXISTING and out_npz.exists():
                    progress_queue.put(1)
                    continue

                # =========================
                # 1. SAM3 推理
                # =========================
                predictor.set_image(str(img_path))
                results = predictor(text=TEXT_PROMPTS)
                result = results[0]

                # =========================
                # 2. 官方可视化保存
                # =========================
                if SAVE_VIS:
                    vis_dir = save_dir / "_vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"{img_path.stem}.jpg"

                    try:
                        # Ultralytics 官方 Results.save()
                        # 保存内容：原图 + mask + box + label + conf
                        result.save(filename=str(vis_path))
                    except Exception as e:
                        print(f"[GPU {gpu_id}] 可视化保存失败: {img_path}, error={e}")

                # =========================
                # 3. 保存结构化 .npz
                # =========================
                if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
                    # 类别索引：对应 TEXT_PROMPTS 的 index
                    cls_indices = result.boxes.cls.cpu().numpy().astype(np.int16)

                    # 实例置信度
                    confs = result.boxes.conf.cpu().numpy().astype(np.float16)

                    # box: xyxy
                    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)

                    # mask 概率图/分数图
                    masks_prob = result.masks.data.cpu().numpy().astype(np.float16)

                    # bool mask
                    masks_bool = masks_prob > 0.5

                    # 保存类别名
                    class_names = np.array([SAVE_NAMES[int(i)] for i in cls_indices])
                else:
                    cls_indices = np.array([], dtype=np.int16)
                    confs = np.array([], dtype=np.float16)
                    boxes = np.empty((0, 4), dtype=np.float32)
                    masks_prob = np.empty((0, 0, 0), dtype=np.float16)
                    masks_bool = np.empty((0, 0, 0), dtype=bool)
                    class_names = np.array([])

                # 原图尺寸，方便后续检查 mask 尺寸是否对齐
                if hasattr(result, "orig_shape") and result.orig_shape is not None:
                    orig_shape = np.array(result.orig_shape, dtype=np.int32)
                else:
                    orig_shape = np.array([-1, -1], dtype=np.int32)

                np.savez_compressed(
                    out_npz,
                    classes=class_names,
                    prompt_ids=cls_indices,
                    confidences=confs,
                    boxes=boxes,
                    masks=masks_bool,
                    masks_prob=masks_prob,
                    text_prompts=np.array(TEXT_PROMPTS),
                    save_names=np.array(SAVE_NAMES),
                    image_path=str(img_path),
                    orig_shape=orig_shape,
                )

            except Exception as e:
                print(f"[GPU {gpu_id}] 处理失败: {img_path}, error={e}")

            finally:
                progress_queue.put(1)


def collect_images(folder_path):
    """
    收集输入目录下所有图片。
    支持 .jpg / .jpeg / .png
    """
    exts = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in folder_path.rglob("*") if p.suffix.lower() in exts]
    all_images = sorted(all_images)

    if MAX_IMAGES is not None:
        all_images = all_images[:MAX_IMAGES]

    return all_images


def split_list(items, num_chunks):
    """
    将图片列表切成 num_chunks 份，分给多个 GPU。
    """
    if num_chunks <= 0:
        return []

    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    while len(chunks) < num_chunks:
        chunks.append([])

    return chunks


def main():
    folder_path = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not folder_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {folder_path}")

    all_images = collect_images(folder_path)
    total_imgs = len(all_images)

    if total_imgs == 0:
        print(f"[Main] 没有找到图片: {folder_path}")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("没有检测到 CUDA GPU，请先检查 PyTorch CUDA 环境")

    print(f"[Main] 输入目录: {folder_path}")
    print(f"[Main] 输出目录: {output_folder}")
    print(f"[Main] 总计找到 {total_imgs} 张图片")
    print(f"[Main] 使用 {num_gpus} 个 GPU")
    print(f"[Main] MODEL_PATH = {MODEL_PATH}")
    print(f"[Main] SAVE_VIS = {SAVE_VIS}")
    print(f"[Main] MAX_IMAGES = {MAX_IMAGES}")
    print(f"[Main] SKIP_EXISTING = {SKIP_EXISTING}")

    chunks = split_list(all_images, num_gpus)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    progress_queue = manager.Queue()

    processes = []
    total_start = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, chunks[rank], folder_path, output_folder, progress_queue),
        )
        p.start()
        processes.append(p)

    with tqdm(total=total_imgs, desc="Processing", unit="img") as pbar:
        processed_count = 0
        last_progress = time.time()

        while processed_count < total_imgs:
            try:
                _ = progress_queue.get(timeout=5)
                pbar.update(1)
                processed_count += 1
                last_progress = time.time()
            except queue.Empty:
                # 如果所有进程都死了，就退出
                if not any(p.is_alive() for p in processes):
                    print("\n[Warning] 所有 worker 已结束，但进度未达到 total_imgs")
                    break

                # 如果长时间没有进度，强制终止
                if time.time() - last_progress > 600:
                    print("\n[Error] Workers 可能卡住，10 分钟无进度，准备终止...")
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    break

    for p in processes:
        p.join()

    total_end = time.time()
    print(f"\n[Done] 全部完成! 总耗时: {total_end - total_start:.2f} 秒")
    print(f"[Done] 输出目录: {output_folder}")


if __name__ == "__main__":
    main()