"""
Reproducible training script for the MentalBERT 7-class text classifier.
Faithfully reproduces the experimental pipeline from:
notebooks/text_model.ipynb

Dataset:
Kaggle Sentiment Analysis for Mental Health (approx 52,680 records)
7 Target Classes:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

Checkpoint saved to:
models/text/text_model_final.pt
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import argparse
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv

load_dotenv()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Expected classes in exact alphabetical order from LabelEncoder
EXPECTED_CLASSES = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

MODEL_NAME = "mental/mental-bert-base-uncased"
DEFAULT_CSV_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--AhmedSSoliman--sentiment-analysis-for-mental-health-Combined-Data"
    "/snapshots/014f442ffa00109fbd96ce285ab3b2e5c0fe86fc/sentiment-analysis-for-mental-health-Combined Data.csv"
)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for tokenized mental health statements."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_and_preprocess_data(
    csv_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], LabelEncoder]:
    """
    Load dataset from local CSV or fallback, clean text, and encode labels.
    """
    path_to_use = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path_to_use):
        alt_path = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--TungHamHoc--sentiment-mental-health"
            "/snapshots/a273145c5f1bb5e958ecce1da4cd906c532f6845/sentiment-mental-health.csv"
        )
        if os.path.exists(alt_path):
            path_to_use = alt_path

    if not os.path.exists(path_to_use):
        print("Loading via datasets library: AhmedSSoliman/sentiment-analysis-for-mental-health-Combined-Data")
        from datasets import load_dataset
        ds = load_dataset("AhmedSSoliman/sentiment-analysis-for-mental-health-Combined-Data")["train"]
        df = ds.to_pandas()
    else:
        print(f"Loading dataset from: {path_to_use}")
        df = pd.read_csv(path_to_use, encoding="latin-1")

    # Standardize columns: statement, status
    text_col = "statement" if "statement" in df.columns else ("text" if "text" in df.columns else df.columns[1])
    label_col = "status" if "status" in df.columns else ("label" if "label" in df.columns else df.columns[2])

    df_clean = pd.DataFrame({
        "text": df[text_col],
        "label": df[label_col]
    }).dropna()

    print(f"Total raw records after dropna: {len(df_clean)}")
    print("Class distribution:\n", df_clean["label"].value_counts())

    le = LabelEncoder()
    df_clean["label_encoded"] = le.fit_transform(df_clean["label"])

    print(f"\nLabelEncoder classes ({len(le.classes_)}):")
    for idx, c in enumerate(le.classes_):
        print(f"  {idx} -> {c}")

    assert list(le.classes_) == EXPECTED_CLASSES, (
        f"Encoded class mismatch! Expected {EXPECTED_CLASSES}, got {list(le.classes_)}"
    )

    if max_samples and max_samples < len(df_clean):
        print(f"Stratified subsampling to approximately {max_samples} records...")
        frac = max_samples / len(df_clean)
        df_clean = (
            df_clean.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(max(10, int(np.rint(len(x) * frac))), random_state=seed))
            .reset_index(drop=True)
        )

    # Force clean all text data (reproducing Cell 6B)
    texts = df_clean["text"].tolist()
    cleaned_texts = [str(x) if x is not None else "" for x in texts]
    cleaned_texts = [x.strip() for x in cleaned_texts]
    cleaned_texts = [x if x else "empty" for x in cleaned_texts]

    labels = df_clean["label_encoded"].tolist()

    return cleaned_texts, labels, le


def tokenize_texts(tokenizer: Any, texts: List[str], max_len: int = 128) -> Dict[str, torch.Tensor]:
    """Tokenize a list of texts into PyTorch tensors."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3,
    checkpoint_dir: str = "models/text",
) -> None:
    """Train sequence classification model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\nStarting training on {device} for {epochs} epoch(s)...")
    print(f"Batches per epoch: {len(train_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - start_time
                remaining = (elapsed / (batch_idx + 1)) * (len(train_loader) - batch_idx - 1)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {correct / total:.4f} | "
                    f"ETA: {remaining / 60:.1f}m"
                )

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        epoch_time = (time.time() - start_time) / 60

        print(f"\n{'=' * 55}")
        print(f"Epoch {epoch + 1}/{epochs} Complete")
        print(f"  Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")
        print(f"  Avg Train Loss : {total_loss / len(train_loader):.4f}")
        print(f"  Avg Val Loss   : {val_loss / len(val_loader):.4f}")
        print(f"  Duration       : {epoch_time:.1f} minutes")
        print(f"{'=' * 55}\n")

        epoch_save_path = os.path.join(checkpoint_dir, f"text_model_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Saved checkpoint: {epoch_save_path}")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate model on hold-out test set."""
    model.eval()
    all_preds = []
    all_labels = []

    print("\nRunning hold-out test evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = outputs.logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print("\n" + "=" * 55)
    print("TEXT MODEL — FINAL TEST RESULTS")
    print("=" * 55)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1_macro:.4f}")
    print(f"F1 Weight: {f1_weighted:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    suicidal_metrics = report_dict.get("Suicidal", {})
    print("Suicidal Class Metrics:")
    print(f"  Precision: {suicidal_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall   : {suicidal_metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score : {suicidal_metrics.get('f1-score', 0.0):.4f}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MentalBERT 7-class text classifier")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to sentiment CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample dataset for quick training/testing")
    parser.add_argument("--output-path", type=str, default="models/text/text_model_final.pt", help="Path to save final weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load and clean data
    X, y, le = load_and_preprocess_data(
        csv_path=args.dataset_path,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # 2. 80 / 10 / 10 stratified train/val/test split (exact notebook logic)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    print(f"\nSplit Sizes: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # 3. Load Tokenizer
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token or True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token or True)

    print(f"Loaded tokenizer: {MODEL_NAME}")

    # 4. Tokenize sets
    print("Tokenizing train, val, and test splits (max_len=128)...")
    train_enc = tokenize_texts(tokenizer, X_train)
    val_enc = tokenize_texts(tokenizer, X_val)
    test_enc = tokenize_texts(tokenizer, X_test)

    train_dataset = MentalHealthDataset(train_enc, y_train)
    val_dataset = MentalHealthDataset(val_enc, y_val)
    test_dataset = MentalHealthDataset(test_enc, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Initialize Model
    print("Initializing AutoModelForSequenceClassification with 7 labels...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(EXPECTED_CLASSES),
            token=token or True,
            local_files_only=True,
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(EXPECTED_CLASSES),
            token=token or True,
        )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 6. Train
    output_dir = os.path.dirname(args.output_path) or "models/text"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=output_dir,
    )

    # 7. Save final model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"\nFinal model state_dict successfully saved to: {args.output_path}")

    # 8. Evaluate on test set
    _ = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=EXPECTED_CLASSES,
    )

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
