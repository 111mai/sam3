import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import warnings
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import clip

# monkey-patch: SAM3 期望 SimpleTokenizer() 可调用，但 clip>=1.0 的 SimpleTokenizer 不可调用
# 用 clip.tokenize 替代，接口完全兼容
clip.simple_tokenizer.SimpleTokenizer = lambda: clip.tokenize

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils import ops
from tqdm import tqdm
import queue
#os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "sam3.pt"
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

INPUT_FOLDER = "/home/via/mai/datasets/images"
OUTPUT_FOLDER = "/home/via/mai/datasets/sam3_coco_label"   


# 是否保存官方可视化 overlay
SAVE_VIS = False

# 调试阶段先限制图片数量；正式跑全量时改成 None
MAX_IMAGES = 10


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
# ========== CoCo-SAM3 融合参数 ==========
# 论文中 synonym aggregation 使用 tau_s = 0.10
TAU_S = 0.10

# semantic prior 项的权重，先用 0.7，后续可以根据可视化调参
LAMBDA_PRIOR = 0.7

# 背景阈值：最大融合分数低于这个值的像素设为 background = -1
BG_SCORE_THRESH = 0.0

# 最终语义类别表：去掉 SAVE_NAMES 里的重复 car
SEMANTIC_NAMES = np.array(list(dict.fromkeys(SAVE_NAMES)))

SEMANTIC_NAME_TO_ID = {
    name: i for i, name in enumerate(SEMANTIC_NAMES)
}

# 每个 prompt 映射到哪个最终语义类
# 例如 a pickup 和 a car 都映射到 car
PROMPT_TO_SEMANTIC = np.array(
    [SEMANTIC_NAME_TO_ID[name] for name in SAVE_NAMES],
    dtype=np.int64,
)
def build_coco_sam3_semantic(
    predictor,
    pred_masks,
    presence_logit,
    final_masks_logits_low,
    final_cls,
    prompt_embed,
    src_shape,
):
    """
    根据 CoCo-SAM3 公式生成 semantic_label。

    输出：
    semantic_label: (H, W), int16, -1 表示 background
    coco_score: (H, W), float16, 每个像素最大融合分数
    """

    device = pred_masks.device
    mask_h, mask_w = pred_masks.shape[-2:]
    num_semantic = len(SEMANTIC_NAMES)

    prompt_to_semantic = torch.tensor(
        PROMPT_TO_SEMANTIC,
        device=device,
        dtype=torch.long,
    )

    # 1. 结构证据：SAM3 mask logit
    structural_logits = torch.full(
        (num_semantic, mask_h, mask_w),
        -1e4,
        dtype=torch.float32,
        device=device,
    )

    for i in range(final_masks_logits_low.shape[0]):
        prompt_id = int(final_cls[i].item())
        sem_id = int(prompt_to_semantic[prompt_id].item())

        logit_map = final_masks_logits_low[i].float()

        # 同一语义类可能有多个 mask，用 max 聚合
        structural_logits[sem_id] = torch.maximum(
            structural_logits[sem_id],
            logit_map,
        )

    # 2. presence logit：每个语义类一个全图存在性分数
    z_prompt = presence_logit.squeeze(-1).float()

    z_semantic = torch.full(
        (num_semantic,),
        -1e4,
        dtype=torch.float32,
        device=device,
    )

    for sid in range(num_semantic):
        prompt_ids = torch.where(prompt_to_semantic == sid)[0]
        z_semantic[sid] = z_prompt[prompt_ids].max()

    # 3. semantic prior π_c(x)
    image_feat = predictor.features["backbone_fpn"][0].float()  # (1, 256, Hf, Wf)

    if image_feat.shape[-2:] != (mask_h, mask_w):
        image_feat = F.interpolate(
            image_feat,
            size=(mask_h, mask_w),
            mode="bilinear",
            align_corners=False,
        )

    image_feat = image_feat[0]        # (256, Hm, Wm)
    text_feat = prompt_embed.float()  # (num_prompt, 256)

    image_feat = F.normalize(image_feat, dim=0)
    text_feat = F.normalize(text_feat, dim=-1)

    # u_s(x) = e_s^T f(x)
    u_prompt = torch.einsum(
        "pc,chw->phw",
        text_feat,
        image_feat,
    )

    # prompt → semantic，类内 LogSumExp 聚合
    u_semantic = torch.full(
        (num_semantic, mask_h, mask_w),
        -1e4,
        dtype=torch.float32,
        device=device,
    )

    for sid in range(num_semantic):
        prompt_ids = torch.where(prompt_to_semantic == sid)[0]

        u_semantic[sid] = torch.logsumexp(
            u_prompt[prompt_ids] / TAU_S,
            dim=0,
        )

    # log π_c(x)
    log_pi = u_semantic - torch.logsumexp(
        u_semantic,
        dim=0,
        keepdim=True,
    )

    # 4. CoCo-SAM3 融合公式
    score_low = (
        structural_logits
        + LAMBDA_PRIOR * log_pi
        + z_semantic[:, None, None]
    )

    score_up = F.interpolate(
        score_low[None],
        size=src_shape,
        mode="bilinear",
        align_corners=False,
    )[0]

    max_score, semantic_label = score_up.max(dim=0)
    semantic_label = semantic_label.to(torch.int16)

    semantic_label[max_score < BG_SCORE_THRESH] = -1

    return (
        semantic_label.detach().cpu().numpy().astype(np.int16),
        max_score.detach().cpu().numpy().astype(np.float16),
    )
def sam3_raw_predict(predictor, img_path):
    """
    复刻 SAM3SemanticPredictor.postprocess，
    但额外保留 raw mask logits、mask probability、query_id、presence logits。
    """

    # 1. 读取原图尺寸
    with Image.open(img_path) as im:
        image_w, image_h = im.size

    src_shape = (image_h, image_w)

    # 2. 对当前图片提取 SAM3 图像特征
    predictor.set_image(str(img_path))

    # 3. 准备几何 prompt。这里没有 box/point，所以 bboxes=None, labels=None
    prompts = predictor._prepare_geometric_prompts(
        src_shape[:2],
        bboxes=None,
        labels=None,
    )

    # 4. 直接调用 SAM3 内部 inference，拿到 postprocess 前的原始输出
    with torch.no_grad():
        preds = predictor._inference_features(
            predictor.features,
            *prompts,
            text=TEXT_PROMPTS,
        )
    captured = {}

    module_dict = dict(predictor.model.named_modules())
    prompt_proj = module_dict["dot_prod_scoring.prompt_proj"]

    def hook_prompt_proj(module, inputs, output):
        captured["prompt_embed"] = output.detach()

    hook_handle = prompt_proj.register_forward_hook(hook_prompt_proj)

    with torch.no_grad():
        preds = predictor._inference_features(
            predictor.features,
            *prompts,
            text=TEXT_PROMPTS,
        )

    hook_handle.remove()

    if "prompt_embed" not in captured:
        raise RuntimeError("没有捕获到 dot_prod_scoring.prompt_proj 的文本 embedding")

    prompt_embed = captured["prompt_embed"]

    pred_boxes = preds["pred_boxes"]              # (num_prompt, num_query, 4)
    pred_logits = preds["pred_logits"]            # (num_prompt, num_query, 1)
    pred_masks = preds["pred_masks"]              # (num_prompt, num_query, H, W), raw logits
    presence_logit = preds["presence_logit_dec"]  # (num_prompt, 1)

    # 5. 复刻官方置信度计算
    pred_scores = pred_logits.sigmoid()
    presence_score = presence_logit.sigmoid().unsqueeze(1)
    pred_scores = (pred_scores * presence_score).squeeze(-1)  # (num_prompt, num_query)

    nc, num_query = pred_scores.shape

    # 6. 自己构造 prompt id 和 query id 网格，后面跟着筛选一起走
    cls_grid = torch.arange(nc, device=pred_scores.device)[:, None].expand(nc, num_query)
    query_grid = torch.arange(num_query, device=pred_scores.device)[None, :].expand(nc, num_query)

    # 7. 第一轮筛选：置信度阈值
    keep_thr = pred_scores > predictor.args.conf

    if keep_thr.sum().item() == 0:
        return dict(
            cls_indices=np.array([], dtype=np.int16),
            query_ids=np.array([], dtype=np.int16),
            confs=np.array([], dtype=np.float16),
            boxes=np.empty((0, 4), dtype=np.float32),
            masks_logits=np.empty((0, image_h, image_w), dtype=np.float16),
            masks_prob=np.empty((0, image_h, image_w), dtype=np.float16),
            masks_bool=np.empty((0, image_h, image_w), dtype=bool),
            presence_logits=presence_logit.squeeze(-1).detach().cpu().numpy().astype(np.float16),
            image_shape=np.array([image_h, image_w], dtype=np.int32),
        )

    kept_masks_logits = pred_masks[keep_thr]
    kept_boxes = pred_boxes[keep_thr]
    kept_scores = pred_scores[keep_thr]
    kept_cls = cls_grid[keep_thr]
    kept_query = query_grid[keep_thr]

    # 8. box 从 xywh 转成 xyxy，和官方源码一致
    kept_boxes_xyxy = ops.xywh2xyxy(kept_boxes)

    # 9. 按类别偏移做 NMS，和官方源码一致
    class_offset = kept_cls[:, None].to(kept_boxes_xyxy.dtype) * (
        0 if predictor.args.agnostic_nms else 7680
    )

    nms_boxes = kept_boxes_xyxy + class_offset

    keep_nms = torchvision.ops.nms(
        nms_boxes.float(),
        kept_scores.float(),
        predictor.args.iou,
    )

    final_masks_logits_low = kept_masks_logits[keep_nms]
    final_boxes_xyxy = kept_boxes_xyxy[keep_nms]
    final_scores = kept_scores[keep_nms]
    final_cls = kept_cls[keep_nms]
    final_query = kept_query[keep_nms]

    # 10. 把 raw mask logits 插值到原图大小
    final_masks_logits_up = F.interpolate(
        final_masks_logits_low.float()[None],
        size=src_shape,
        mode="bilinear",
        align_corners=False,
    )[0]

    # 11. raw logits -> probability -> bool mask
    final_masks_prob_up = final_masks_logits_up.sigmoid()
    final_masks_bool = final_masks_prob_up > 0.5

    # 12. box 从归一化坐标变成原图像素坐标
    final_boxes_xyxy = final_boxes_xyxy.float()
    final_boxes_xyxy[..., [0, 2]] *= image_w
    final_boxes_xyxy[..., [1, 3]] *= image_h
    # 13. CoCo-SAM3 融合，生成 semantic_label
    semantic_label, coco_score = build_coco_sam3_semantic(
        predictor=predictor,
        pred_masks=pred_masks,
        presence_logit=presence_logit,
        final_masks_logits_low=final_masks_logits_low,
        final_cls=final_cls,
        prompt_embed=prompt_embed,
        src_shape=src_shape,
    )
    return dict(
        cls_indices=final_cls.detach().cpu().numpy().astype(np.int16),
        query_ids=final_query.detach().cpu().numpy().astype(np.int16),
        confs=final_scores.detach().cpu().numpy().astype(np.float16),
        boxes=final_boxes_xyxy.detach().cpu().numpy().astype(np.float32),
        masks_logits=final_masks_logits_up.detach().cpu().numpy().astype(np.float16),
        masks_prob=final_masks_prob_up.detach().cpu().numpy().astype(np.float16),
        masks_bool=final_masks_bool.detach().cpu().numpy().astype(bool),
        presence_logits=presence_logit.squeeze(-1).detach().cpu().numpy().astype(np.float16),
        semantic_label=semantic_label,
        semantic_names=SEMANTIC_NAMES.copy(),
        coco_score=coco_score,
        image_shape=np.array([image_h, image_w], dtype=np.int32),
    )

#多进程 worker 函数
def gpu_worker(gpu_id, image_paths, root_folder, output_root, progress_queue):
    if not image_paths:
        return

    device = f"cuda:{gpu_id}"

    overrides = dict(
        conf=0.5,#置信度阈值，目标/实例级别的置信度阈值，road=-.4,car=0.6,那只保留car,它影响的是最终有多少个 mask 被输出
        task="segment",#任务类型是分割
        mode="predict",#模式是预测/推理
        # model=str(Path(__file__).parent.parent / "sam3.pt"),#指定模型权重路径
        model=str(MODEL_PATH),
        half=True,#使用半精度 FP16 推理
        save=False,
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
            try:
                raw = sam3_raw_predict(predictor, img_path)

                relative_path = img_path.relative_to(root_folder).parent
                save_dir = output_root / relative_path
                save_dir.mkdir(parents=True, exist_ok=True)

                out_npz = save_dir / f"{img_path.stem}.npz"

                # ========== 1. 官方可视化保存 ==========
                # 保存到每个子目录下的 _vis 文件夹
        
                if SAVE_VIS:
                    vis_dir = save_dir / "_vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"{img_path.stem}.jpg"

                    try:
                        results = predictor(text=TEXT_PROMPTS)
                        result = results[0]
                        result.save(filename=str(vis_path))
                    except Exception as e:
                        print(f"[GPU {gpu_id}] 可视化保存失败: {img_path}, error={e}")
                # ========== 2. 保存结构化 npz，供后续 CoCo-SAM3 / 评价使用 ==========
                # if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
                cls_indices = raw["cls_indices"]
                class_names = np.array([SAVE_NAMES[int(i)] for i in cls_indices])
                if len(cls_indices) > 0:
                    print(
                        f"[GPU {gpu_id}] {img_path.name}: "
                        f"classes={class_names.tolist()}, "
                        f"query_ids={raw['query_ids'].tolist()}, "
                        f"confs={raw['confs'].tolist()}"
                    )

                np.savez_compressed(
                    out_npz,
                    classes=class_names,
                    prompt_ids=cls_indices,
                    query_ids=raw["query_ids"],
                    confidences=raw["confs"],
                    boxes=raw["boxes"],
                    masks=raw["masks_bool"],
                    masks_prob=raw["masks_prob"],
                    masks_logits=raw["masks_logits"],
                    presence_logits=raw["presence_logits"],
                    semantic_label=raw["semantic_label"],
                    semantic_names=raw["semantic_names"],
                    coco_score=raw["coco_score"],
                    text_prompts=np.array(TEXT_PROMPTS),
                    save_names=np.array(SAVE_NAMES),
                    image_path=str(img_path),
                    image_shape=raw["image_shape"],
                )

            except Exception as e:
                print(f"[GPU {gpu_id}] 处理失败: {img_path}, error={e}")

            progress_queue.put(1)

def main():
    folder_path = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in folder_path.rglob("*") if p.suffix.lower() in exts]
    all_images = sorted(all_images)
    if MAX_IMAGES is not None:
        all_images = all_images[:MAX_IMAGES]
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
