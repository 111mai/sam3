"""
车辆分类器训练脚本
==================
功能：训练 ResNet101 分类器，用于区分不同车辆类型。
- 输入：cls_dataset/ 目录（ImageFolder 结构，由 build_cls_dataset.py 生成）
- 输出：checkpoints/best_model_cls.pth（最佳模型权重 + 类别名称）
- 模型：ResNet101（ImageNet 预训练）+ Linear(2048, num_classes)
- 数据划分：90% 训练 / 10% 验证（固定随机种子）
- 优化策略：AdamW + CosineAnnealingLR，保存验证集准确率最高的模型
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.transforms import v2
from tqdm import tqdm

# ====== Constants ======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_DIR = "/home/via/mai/datasets/cls_dataset"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


# ====== Model ======

class ResNetClassifier(nn.Module):
    """ResNet101 backbone + Linear(2048, num_classes)."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet101(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ====== Data ======

class TransformSubset(Dataset):
    """Subset with a custom transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.classes = dataset.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y


def make_train_transform(crop_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_eval_transform(crop_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(crop_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_loaders(data_dir, batch_size, num_workers):
    full_ds = ImageFolder(data_dir)
    rng = torch.Generator().manual_seed(42)
    n_train = int(len(full_ds) * 0.9)
    indices = torch.randperm(len(full_ds), generator=rng).tolist()

    train_ds = TransformSubset(full_ds, indices[:n_train], make_train_transform())
    val_ds = TransformSubset(full_ds, indices[n_train:], make_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds


# ====== Training loop ======

def run_epoch(model, loader, criterion, device, optimizer=None,
              epoch=None, total_epochs=None):
    training = optimizer is not None
    model.train(training)

    desc = f"Train {epoch}/{total_epochs}" if training else "Val"
    total_loss, correct, total = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=desc, ncols=100)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(imgs)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


# ====== Main ======

def train_classifier(data_dir, output_dir, num_epochs=20, batch_size=64,
                     learning_rate=1e-2, weight_decay=1e-4, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_ds = build_loaders(data_dir, batch_size, num_workers)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = ResNetClassifier(len(train_ds.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    os.makedirs(output_dir, exist_ok=True)

    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            epoch=epoch + 1, total_epochs=num_epochs)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "val_acc": val_acc,
            }, os.path.join(output_dir, "best_model_cls.pth"))
            print(f"  -> saved best (acc={val_acc:.4f})")

    print(f"\nBest val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    train_classifier(data_dir=args.data_dir, output_dir=args.output_dir,
                     num_epochs=args.epochs, batch_size=args.batch_size,
                     learning_rate=args.lr)
