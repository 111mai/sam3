"""
SAM3 语义分割推理脚本
======================
功能：使用 SAM3 模型对图片进行语义分割，生成标签文件。
- 输入：images/ 目录下的所有图片（.jpg/.png/.jpeg）
- 输出：sam3_label/ 目录下的 .npz 标签文件（classes, confidences, masks）
- 使用多 GPU 并行处理，每个 GPU 独立加载模型并分配一部分图片
- 通过文本提示词（road, car, truck, bus 等）定义要分割的类别
输入图片路径
    ↓
main() 搜索所有图片
    ↓
按 GPU 数量切分图片列表
    ↓
每张 GPU 启动一个进程
    ↓
每个进程加载一份 SAM3 模型
    ↓
对每张图片 set_image()
    ↓
用 TEXT_PROMPTS 做文本引导分割
    ↓
得到 boxes.cls / boxes.conf / masks.data
    ↓
类别索引 → SAVE_NAMES 类别名
    ↓
mask 概率图 → bool mask
    ↓
保存成 .npz
"""

import warnings
import logging
import os
from pathlib import Path
from PIL import Image
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import clip

# monkey-patch: SAM3 期望 SimpleTokenizer() 可调用，但 clip>=1.0 的 SimpleTokenizer 不可调用
# 用 clip.tokenize 替代，接口完全兼容
clip.simple_tokenizer.SimpleTokenizer = lambda: clip.tokenize

from ultralytics.models.sam import SAM3SemanticPredictor
from tqdm import tqdm
import queue

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

INPUT_FOLDER = "/home/via/mai/datasets/images_continuous"
OUTPUT_FOLDER = "/home/via/mai/datasets/sam3_label_continuous"   

TEXT_PROMPTS = [
    'road', 'a construction vehicle',  'a truck','a pickup','a bus', 'a car',
    'a motorcycle', 'a bicycle', 'a rider', 'a pedestrian'
]#文本提示词列表
SAVE_NAMES = [
    'road',
    'construction vehicle',
    'truck',
    'car',
    'bus',
    'car',
    'motorcycle',
    'bicycle',
    'rider',
    'pedestrian'
]#存到 .npz 里的类别名。SAM3 检测到 pickup 时，保存的时候把它归为 car

#多进程 worker 函数
def gpu_worker(gpu_id, image_paths, root_folder, output_root, progress_queue):
    if not image_paths:
        return

    device = f"cuda:{gpu_id}"

    overrides = dict(
        conf=0.5,#置信度阈值，目标/实例级别的置信度阈值，road=-.4,car=0.6,那只保留car,它影响的是最终有多少个 mask 被输出
        task="segment",#任务类型是分割
        mode="predict",#模式是预测/推理
        model=str(Path(__file__).parent.parent / "sam3.pt"),#指定模型权重路径
        half=True,#使用半精度 FP16 推理
        save=False,#不让 Ultralytics 自动保存可视化结果
        device=device,
        verbose=False,#关闭详细输出
    )

    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return
    
    with torch.no_grad():#进入无梯度模式，因为这是推理，不需要反向传播
        for img_path in image_paths:
            predictor.set_image(str(img_path))
            results = predictor(text=TEXT_PROMPTS)#用文本 prompt 进行语义分割推理

            relative_path = img_path.relative_to(root_folder).parent
            save_dir = output_root / relative_path
            save_dir.mkdir(parents=True, exist_ok=True)
            out_npz = save_dir / f"{img_path.stem}.npz"

            if results[0].masks is not None:
                cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)#取出每个检测/分割结果对应的类别索引
                confs = results[0].boxes.conf.cpu().numpy().astype(np.float16)#取出每个结果的置信度
                masks_prob = results[0].masks.data.cpu().numpy().astype(np.float16)#取出 mask 数据

                class_names = [SAVE_NAMES[i] for i in cls_indices]

                # 将概率图转为 bool
                masks_bool = masks_prob > 0.5#如果某个像素值大于 0.5，则认为属于该目标.假设模型已经保留了一个 car 的 mask，这个 mask 里面每个像素可能有一个概率值.这个像素属于不属于该目标。
            else:
                class_names = np.array([])
                confs = np.array([], dtype=np.float16)
                masks_bool = np.array([], dtype=bool)

            np.savez_compressed(
                out_npz,
                classes=np.array(class_names),
                confidences=confs,
                masks=masks_bool,
            )

            progress_queue.put(1)

def main():
    folder_path = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_images = list(folder_path.rglob("*.[jp][pn]g"))
    total_imgs = len(all_images)
    if total_imgs == 0:
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("没有检测到 CUDA GPU，请先检查 PyTorch CUDA 环境")
    print(f"[Main] 总计找到 {total_imgs} 张图片，使用 {num_gpus} 个 GPU 处理...")

    chunk_size = (total_imgs + num_gpus - 1) // num_gpus
    chunks = [all_images[i:i + chunk_size] for i in range(0, total_imgs, chunk_size)]
    while len(chunks) < num_gpus:
        chunks.append([])

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    progress_queue = manager.Queue()

    processes = []
    total_start = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, chunks[rank], folder_path, output_folder, progress_queue)
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
                if not any(p.is_alive() for p in processes):
                    break
                if time.time() - last_progress > 600:
                    print("\n[Error] Workers appear stuck (no progress for 10 min), terminating...")
                    for p in processes:
                        p.terminate()
                    break

    for p in processes:
        p.join()

    total_end = time.time()
    print(f"\n[Done] 全部完成! 总耗时: {total_end - total_start:.2f} 秒")

if __name__ == "__main__":
    main()
