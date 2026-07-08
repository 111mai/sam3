"""
共享模型定义与工具函数
======================
本文件是项目的模型仓库，定义了多种模型架构和辅助工具：

模型类：
  - ResNetClassifier    — ResNet101 + Linear 分类头（用于车辆分类）
  - SegmentModel        — ViT backbone + LinearHead 分割头（用于语义分割）
  - Classifier          — ViT backbone + LinearClassifier（利用中间层 token 做分类）
  - LinearHead          — 分割任务的 lightweight 解码头
  - LinearClassifier    — 基于 ViT 多层的线性分类器
  - AngleRegressor      — ResNet101 + Linear(2048,2) 方位角回归
  - build_seg_model()   — FCN-ResNet101 语义分割模型工厂

工具函数：
  - compute_bbox() / expand_bbox() — mask → 边界框计算与扩展
  - ImageDirDataset     — 通用图片文件夹数据集

常量：
  - CLASSES             — 语义分割的 9 个类别
  - COLORS              — 各类别对应的 BGR 颜色
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights

IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
IMAGENET_STD_255 = (58.395, 57.12, 57.375)

CLASSES = [
    "road", "construction vehicle", "truck", "bus", "car",
    "motorcycle", "bicycle", "rider", "pedestrian",
]

COLORS: dict[str, tuple[int, int, int]] = {
    "road": (128, 64, 128),
    "construction vehicle": (255, 220, 0),
    "truck": (139, 90, 40),
    "bus": (0, 200, 80),
    "car": (0, 80, 220),
    "motorcycle": (255, 220, 0),
    "bicycle": (255, 220, 0),
    "rider": (230, 60, 60),
    "pedestrian": (230, 60, 60),
}


def _forward_backbone(backbone, x, **kwargs):
    with torch.no_grad(), torch.autocast("cuda", enabled=x.is_cuda):
        return backbone.get_intermediate_layers(x, **kwargs)


class LinearHead(nn.Module):
    def __init__(self, in_channels, num_classes, use_batchnorm=True, dropout=0.1):
        super().__init__()
        self.channels = sum(in_channels) if isinstance(in_channels, list) else in_channels
        self.bn = nn.BatchNorm2d(self.channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(self.channels, num_classes, 1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        if isinstance(x, list):
            x = [F.interpolate(f, size=x[0].shape[2:], mode="bilinear", align_corners=False) for f in x]
            x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)
        return x


class SegmentModel(nn.Module):
    def __init__(self, backbone, num_classes, out_indices=None, backbone_out_layers="last"):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        n_blocks = backbone.n_blocks

        if out_indices is not None:
            self.out_indices = out_indices
        elif backbone_out_layers == "last":
            self.out_indices = [n_blocks - 1]
        elif backbone_out_layers == "four_last":
            self.out_indices = list(range(n_blocks - 4, n_blocks))
        elif backbone_out_layers == "four_even_intervals":
            self.out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
        else:
            raise ValueError(f"Unknown backbone_out_layers: {backbone_out_layers}")

        embed_dims = [backbone.embed_dim] * len(self.out_indices)
        self.head = LinearHead(embed_dims, num_classes)

    def forward(self, x):
        features = _forward_backbone(
            self.backbone, x, n=self.out_indices, reshape=True, return_class_token=False,
        )
        return self.head(list(features))


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, use_n_blocks=1, use_avgpool=True):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.use_n_blocks = use_n_blocks
        embed_dim = backbone.embed_dim
        out_dim = use_n_blocks * embed_dim + (embed_dim if use_avgpool else 0)
        self.head = LinearClassifier(out_dim, use_n_blocks, use_avgpool, num_classes)

    def forward(self, x):
        n = self.backbone.n_blocks
        block_indices = list(range(n - self.use_n_blocks, n))
        x_tokens_list = _forward_backbone(
            self.backbone, x, n=block_indices, norm=True, reshape=False, return_class_token=True,
        )
        return self.head(x_tokens_list)


def compute_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def expand_bbox(bbox, img_w, img_h, pad_ratio):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    return max(0, x1 - px), max(0, y1 - py), min(img_w, x2 + px), min(img_h, y2 + py)


class AngleRegressor(nn.Module):
    """ResNet101 backbone + Linear(2048, 2) head for azimuth angle regression."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet101(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class ResNetClassifier(nn.Module):
    """ResNet101 backbone + Linear(2048, num_classes) for classification."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet101(weights=weights)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_seg_model(num_classes, pretrained=True):
    """FCN-ResNet101 with frozen backbone, trainable classifier head."""
    weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = fcn_resnet101(weights=weights)
    if num_classes != 21:
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, 1)
        if model.aux_classifier is not None:
            in_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_aux, num_classes, 1)
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


class ImageDirDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.paths = [p for p in self.paths if os.path.isfile(p)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]
