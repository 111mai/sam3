"""
探测 SAM3 内部结构
=================
目的：搞清楚 SAM3 推理时内部有哪些数据可用，为写 CoCo-SAM3 做准备。

只跑一张图，打印：
  1. predictor.features 里有什么（图像特征）
  2. predictor.model 的子模块结构（找 PE 中间层、text encoder）
  3. _inference_features 返回了什么
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import torch
import clip
clip.simple_tokenizer.SimpleTokenizer = lambda: clip.tokenize

from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "/home/via/sam3/sam3.pt"

# 找一张测试图片
IMAGE_DIR = Path("/home/via/mai/datasets/images")
test_imgs = sorted(IMAGE_DIR.rglob("*.png"))
if not test_imgs:
    test_imgs = sorted(IMAGE_DIR.rglob("*.jpg"))
IMG_PATH = str(test_imgs[0])
print(f"测试图片: {IMG_PATH}")

TEXT_PROMPTS = [
    'road', 'a construction vehicle', 'a truck', 'a pickup', 'a bus',
    'a car', 'a motorcycle', 'a bicycle', 'a rider', 'a pedestrian',
]

# -------- 加载模型 --------
overrides = dict(conf=0.5, task="segment", mode="predict",
                 model=MODEL_PATH, half=True, save=False, device="cuda:0", verbose=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

# ============================================================
# 1. 探测 predictor.features（set_image 之后）
# ============================================================
print("\n" + "=" * 60)
print("1. predictor.features 里有什么？")
print("=" * 60)
predictor.set_image(IMG_PATH)

for key, val in predictor.features.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        print(f"  {key}: list/tuple, len={len(val)}")
        if len(val) > 0 and isinstance(val[0], torch.Tensor):
            print(f"    [0]: shape={val[0].shape}")
    else:
        print(f"  {key}: type={type(val).__name__}, value={str(val)[:100]}")

# ============================================================
# 2. 探测 predictor.model 结构
# ============================================================
print("\n" + "=" * 60)
print("2. predictor.model 顶层子模块")
print("=" * 60)
for name, module in predictor.model.named_children():
    print(f"  {name}: {type(module).__name__}")

# ============================================================
# 3. 列出所有 named_modules（筛选关键的）
# ============================================================
print("\n" + "=" * 60)
print("3. 关键子模块（按名称筛选）")
print("=" * 60)
all_modules = dict(predictor.model.named_modules())
keywords = ["encoder", "perception", "backbone", "text", "prompt", "proj",
            "fpn", "neck", "dot_prod", "presence", "mask", "decoder", "image"]
for kw in keywords:
    matches = [(n, type(m).__name__) for n, m in all_modules.items() if kw.lower() in n.lower()]
    if matches:
        print(f"\n  [{kw}]:")
        for n, t in matches:
            print(f"    {n}  →  {t}")

# ============================================================
# 4. 探测 _inference_features 返回了什么
# ============================================================
print("\n" + "=" * 60)
print("4. _inference_features 返回的字典")
print("=" * 60)
prompts = predictor._prepare_geometric_prompts(
    (predictor.features["orig_shape"][0].item(), predictor.features["orig_shape"][1].item()),
    bboxes=None, labels=None,
)
with torch.no_grad():
    preds = predictor._inference_features(predictor.features, *prompts, text=TEXT_PROMPTS)

for key, val in preds.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        print(f"  {key}: list/tuple, len={len(val)}")
        if len(val) > 0 and isinstance(val[0], torch.Tensor):
            print(f"    [0]: shape={val[0].shape}")
    else:
        print(f"  {key}: type={type(val).__name__}")

# ============================================================
# 5. 看看 image_encoder 有没有中间层可以 hook
# ============================================================
print("\n" + "=" * 60)
print("5. image_encoder 内部结构")
print("=" * 60)
if "image_encoder" in dict(predictor.model.named_children()):
    ie = predictor.model.image_encoder
    print(f"  image_encoder type: {type(ie).__name__}")
    for name, _ in ie.named_children():
        print(f"    {name}")
    # 看看有没有 blocks/layers
    for name, mod in ie.named_modules():
        if "block" in name.lower() or "layer" in name.lower():
            print(f"    {name}: {type(mod).__name__}")
            break  # 只打印第一个，太多了

# ============================================================
# 6. 用 hook 捕获 image_encoder 中间层输出
# ============================================================
print("\n" + "=" * 60)
print("6. Hook image_encoder 各层输出")
print("=" * 60)

captured_blocks = {}

def make_hook(name):
    def hook_fn(module, input, output):
        captured_blocks[name] = output.detach()
    return hook_fn

handles = []
ie = predictor.model.image_encoder
for name, module in ie.named_modules():
    # 找所有 transformer block 层
    if hasattr(module, 'norm1') or hasattr(module, 'attn'):
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

# 换张图跑一次推理，触发 hook
predictor.set_image(str(test_imgs[1]) if len(test_imgs) > 1 else IMG_PATH)
with torch.no_grad():
    _ = predictor._inference_features(predictor.features, *prompts, text=TEXT_PROMPTS)

for name, tensor in captured_blocks.items():
    print(f"  {name}: shape={tensor.shape}")

for h in handles:
    h.remove()

print("\n" + "=" * 60)
print("[DONE] 探测完成")
print("=" * 60)
