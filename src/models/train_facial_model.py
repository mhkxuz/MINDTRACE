"""
Reproducible training script for the Facial Emotion CNN.
Faithfully reproduces the experimental pipeline from:
notebooks/facial_emotion_label.ipynb

Dataset:
FER2013 (28,709 train, 3,589 test images, 48x48)
Labels (in exact order from notebooks/facial_emotion_label.ipynb):
0: angry
1: disgust
2: fear
3: happy
4: neutral
5: sad
6: surprise

Checkpoint saved to:
models/facial/fer_model.pt
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import argparse
import random
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from datasets import Dataset as HFDataset

from src.models.facial_model import (
    FacialEmotionCNN,
    EMOTION_LABELS,
)

# Label mapping from AutumnQiu/fer2013 ClassLabel index to MINDTRACE EMOTION_LABELS index
# AutumnQiu: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
AUTUMNQIU_TO_MINDTRACE = {
    0: 0,  # Angry -> angry
    1: 1,  # Disgust -> disgust
    2: 2,  # Fear -> fear
    3: 3,  # Happy -> happy
    4: 5,  # Sad -> sad
    5: 6,  # Surprise -> surprise
    6: 4,  # Neutral -> neutral
}

DEFAULT_ARROW_TRAIN = os.path.expanduser(
    "~/.cache/huggingface/datasets/AutumnQiu___fer2013/default/0.0.0/9c0267afc93fa4d4bfc05d7ff1108a04f72e8392/fer2013-train.arrow"
)
DEFAULT_ARROW_TEST = os.path.expanduser(
    "~/.cache/huggingface/datasets/AutumnQiu___fer2013/default/0.0.0/9c0267afc93fa4d4bfc05d7ff1108a04f72e8392/fer2013-test.arrow"
)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArrowFERDataset(Dataset):
    """
    Dataset wrapper for FER2013 loaded from HuggingFace Arrow files.
    Converts images to RGB 48x48 and maps labels to exact MINDTRACE EMOTION_LABELS.
    """
    def __init__(self, arrow_path: str, transform=None, max_samples: Optional[int] = None):
        self.hf_ds = HFDataset.from_file(arrow_path)
        if max_samples and max_samples < len(self.hf_ds):
            self.hf_ds = self.hf_ds.select(range(max_samples))
        self.transform = transform

        # Precompute mapped labels for class weighting
        raw_labels = self.hf_ds["label"]
        self.mapped_labels = [AUTUMNQIU_TO_MINDTRACE[l] for l in raw_labels]

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.hf_ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = AUTUMNQIU_TO_MINDTRACE[item["label"]]
        return img, label


def get_train_transforms():
    """Exact train transform pipeline from facial_emotion_label.ipynb."""
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_test_transforms():
    """Exact test transform pipeline from facial_emotion_label.ipynb."""
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def train_fer_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    checkpoint_dir: str = "models/facial",
) -> None:
    """Train FER CNN model for specified number of epochs."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\nStarting FER training on {device} for {epochs} epoch(s)...")
    print(f"Batches per epoch: {len(train_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - start_time
                remaining = (elapsed / (batch_idx + 1)) * (len(train_loader) - batch_idx - 1)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {correct / total:.4f} | "
                    f"ETA: {remaining:.1f}s"
                )

        train_acc = correct / total if total > 0 else 0.0
        epoch_time = (time.time() - start_time) / 60
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} Complete | "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Duration: {epoch_time:.2f} min"
        )


def evaluate_fer_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    emotion_labels: List[str],
) -> Dict[str, Any]:
    """Evaluate FER model on hold-out test set."""
    model.eval()
    all_preds = []
    all_labels = []

    print("\nEvaluating FER model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=emotion_labels, output_dict=True)

    print("\n" + "=" * 55)
    print("FER MODEL — FINAL TEST RESULTS")
    print("=" * 55)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1_macro:.4f}")
    print(f"F1 Weight: {f1_weighted:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Train FER2013 7-class Facial Emotion CNN")
    parser.add_argument("--train-arrow", type=str, default=DEFAULT_ARROW_TRAIN)
    parser.add_argument("--test-arrow", type=str, default=DEFAULT_ARROW_TEST)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample dataset for quick training/testing")
    parser.add_argument("--output-path", type=str, default="models/facial/fer_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Datasets and Loaders
    print("Loading FER2013 datasets...")
    train_ds = ArrowFERDataset(
        args.train_arrow,
        transform=get_train_transforms(),
        max_samples=args.max_samples,
    )
    test_ds = ArrowFERDataset(
        args.test_arrow,
        transform=get_test_transforms(),
        max_samples=args.max_samples,
    )

    print(f"Train images: {len(train_ds)}")
    print(f"Test images : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Compute balanced class weights (exact Cell 7)
    train_labels = train_ds.mapped_labels
    unique_classes = np.unique(train_labels)
    unique_weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
    class_weights = np.ones(len(EMOTION_LABELS), dtype=np.float32)
    for c, w in zip(unique_classes, unique_weights):
        class_weights[c] = w
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    print("Class weights:")
    for idx, (label_name, w) in enumerate(zip(EMOTION_LABELS, class_weights)):
        print(f"  {idx} ({label_name:10s}): {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 3. Model & Optimizer
    model = FacialEmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FacialEmotionCNN initialized with {total_params:,} parameters (expected: 1,274,951)")
    assert total_params == 1274951, f"Parameter count mismatch: {total_params} != 1274951"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Train
    checkpoint_dir = os.path.dirname(args.output_path) or "models/facial"
    train_fer_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
    )

    # 5. Save model
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"\nFinal FER model state_dict saved to: {args.output_path}")

    # 6. Evaluate
    _ = evaluate_fer_model(
        model=model,
        test_loader=test_loader,
        device=device,
        emotion_labels=EMOTION_LABELS,
    )

    print("\nFacial emotion training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
