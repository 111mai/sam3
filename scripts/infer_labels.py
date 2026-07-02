import warnings
import logging
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

warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

INPUT_FOLDER = "/home/via/mai/datasets/images"
OUTPUT_FOLDER = "/home/via/mai/datasets/sam3_label"   

TEXT_PROMPTS = [
    'road', 'a construction vehicle',  'a truck','a pickup','a bus', 'a car',
    'a motorcycle', 'a bicycle', 'a rider', 'a pedestrian'
]
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
]


def gpu_worker(gpu_id, image_paths, root_folder, output_root, progress_queue):
    if not image_paths:
        return

    device = f"cuda:{gpu_id}"

    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model=str(Path(__file__).parent.parent / "sam3.pt"),
        half=False,
        save=False,
        device=device,
        verbose=False,
    )

    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return
    
    with torch.no_grad():
        for img_path in image_paths:
            predictor.set_image(str(img_path))
            results = predictor(text=TEXT_PROMPTS)

            relative_path = img_path.relative_to(root_folder).parent
            save_dir = output_root / relative_path
            save_dir.mkdir(parents=True, exist_ok=True)
            out_npz = save_dir / f"{img_path.stem}.npz"

            if results[0].masks is not None:
                cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy().astype(np.float16)
                masks_prob = results[0].masks.data.cpu().numpy().astype(np.float16)

                class_names = [SAVE_NAMES[i] for i in cls_indices]

                # 将概率图转为 bool
                masks_bool = masks_prob > 0.5
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
    print(f"[Main] 总计找到 {total_imgs} 张图片，准备使用 2 个 GPU 处理...")

    if total_imgs == 0:
        return

    # num_gpus = 2
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("没有检测到 CUDA GPU，请先检查 PyTorch CUDA 环境")
    print(f"[Main] 检测到 {num_gpus} 个 GPU")

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
