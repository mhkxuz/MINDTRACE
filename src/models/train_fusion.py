"""
Reproducible training script for the Multimodal Distress Fusion Model.
Reconstructs the exact experimental pipeline from:
notebooks/data_preparation+behavioral_features.ipynb and notebooks/fusion_model.ipynb
Saves:
- artifacts/scalers/behav_scaler_esconv.pkl
- models/fusion/fusion_model.pt
"""

import os
import json
import copy
import random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dotenv import load_dotenv
load_dotenv()

from src.features.behavioral import (
    extract_behavioral_features,
    clean_text,
    FUSION_BEHAVIORAL_COLS,
)
from src.models.fusion_model import FusionModelWeighted


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def parse_raw_esconv_data():
    """Load and parse ESConv dataset from Hugging Face."""
    from datasets import load_dataset
    print("Loading thu-coai/esconv dataset...")
    ds = load_dataset("thu-coai/esconv")["train"]

    conversation_records = []
    turns_records = []

    for i, item in enumerate(ds):
        text_str = item.get("text", "")
        try:
            record = json.loads(text_str) if isinstance(text_str, str) else text_str
        except Exception:
            continue

        if not isinstance(record, dict):
            continue

        conv_id = i + 1
        survey = record.get("survey_score", {})
        seeker_score = survey.get("seeker", {}) if isinstance(survey, dict) else {}

        raw_sit = record.get("situation", "")
        situation = clean_text(raw_sit)

        init_intensity = seeker_score.get("initial_emotion_intensity") if isinstance(seeker_score, dict) else None

        conversation_records.append({
            "conversation_id": conv_id,
            "situation": situation,
            "initial_emotion_intensity": init_intensity,
        })

        dialog = record.get("dialog", [])
        for t_idx, turn in enumerate(dialog, start=1):
            if not isinstance(turn, dict):
                continue
            msg = clean_text(turn.get("text", ""))
            if not msg:
                continue
            spk = turn.get("speaker", "")
            speaker_norm = "user" if spk in ["usr", "seeker", "user"] else "assistant"
            turns_records.append({
                "conversation_id": conv_id,
                "turn_number": t_idx,
                "speaker": speaker_norm,
                "message": msg,
            })

    conv_df = pd.DataFrame(conversation_records)
    turns_df = pd.DataFrame(turns_records)

    # Filter out records with missing initial_emotion_intensity
    conv_df = conv_df.dropna(subset=["initial_emotion_intensity"])
    conv_df["initial_emotion_intensity"] = conv_df["initial_emotion_intensity"].astype(int)

    return conv_df, turns_df


def extract_features_for_all(conv_df, turns_df):
    """Extract 8 behavioral features for each conversation."""
    print("Extracting behavioral features for all conversations...")
    records = []
    for conv_id in conv_df["conversation_id"].unique():
        sub_turns = turns_df[turns_df["conversation_id"] == conv_id].sort_values("turn_number")
        turn_list = [{"speaker": r["speaker"], "message": r["message"]} for _, r in sub_turns.iterrows()]
        feats = extract_behavioral_features(turn_list)
        feats["conversation_id"] = conv_id
        records.append(feats)

    behav_df = pd.DataFrame(records)
    merged_df = behav_df.merge(conv_df, on="conversation_id", how="inner")
    return merged_df


def generate_text_embeddings(situation_texts, hf_token=None):
    """Generate 768-d embeddings using MentalBERT encoder."""
    from transformers import AutoTokenizer, AutoModel
    model_name = "mental/mental-bert-base-uncased"
    token = hf_token or os.getenv("HF_TOKEN")

    print(f"Loading {model_name} for text embedding extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token if token else True)
    model = AutoModel.from_pretrained(model_name, token=token if token else True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    print(f"Extracting embeddings for {len(situation_texts)} situations on {device}...")
    batch_size = 32
    for start_idx in range(0, len(situation_texts), batch_size):
        batch_texts = situation_texts[start_idx : start_idx + batch_size]
        batch_clean = [t.strip() if t.strip() else "empty" for t in batch_texts]
        inputs = tokenizer(
            batch_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outs = model(**inputs)
            cls_emb = outs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)

    return np.vstack(embeddings).astype(np.float32)


def train_and_save_fusion():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    conv_df, turns_df = parse_raw_esconv_data()
    final_df = extract_features_for_all(conv_df, turns_df)

    # Binary distress label: initial_emotion_intensity >= 4
    final_df["distress_label"] = (final_df["initial_emotion_intensity"] >= 4).astype(int)
    y = final_df["distress_label"].values

    print(f"Total dataset samples: {len(final_df)}")
    print(f"Distress distribution:\n{final_df['distress_label'].value_counts(normalize=True)}")

    # Behavioral features
    X_behav_raw = final_df[FUSION_BEHAVIORAL_COLS].fillna(0).values
    scaler = StandardScaler()
    X_behav = scaler.fit_transform(X_behav_raw)

    os.makedirs("artifacts/scalers", exist_ok=True)
    scaler_path = "artifacts/scalers/behav_scaler_esconv.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved behavioral scaler to {scaler_path}")

    # Text embeddings
    situation_texts = final_df["situation"].tolist()
    embeddings = generate_text_embeddings(situation_texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Stratified Splits: 80% train, 10% val, 10% test (matching notebook Cell 18/19)
    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(indices, test_size=0.10, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1111, random_state=42, stratify=y[idx_trainval])

    print(f"Split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    # Class weights matching notebook: tensor([2.2039, 0.6467])
    classes = np.unique(y[idx_train])
    cw = compute_class_weight("balanced", classes=classes, y=y[idx_train])
    cw_tensor = torch.FloatTensor(cw).to(device)
    print(f"Class weights: {cw_tensor}")

    # Build Model C
    model = FusionModelWeighted(text_dim=768, behav_dim=8, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    Xtr_text = torch.FloatTensor(embeddings[idx_train]).to(device)
    Xtr_behav = torch.FloatTensor(X_behav[idx_train]).to(device)
    ytr = torch.LongTensor(y[idx_train]).to(device)

    Xva_text = torch.FloatTensor(embeddings[idx_val]).to(device)
    Xva_behav = torch.FloatTensor(X_behav[idx_val]).to(device)
    yva = y[idx_val]

    epochs = 60
    best_score = -1.0
    best_state = None

    print("\nStarting Fusion Training (60 epochs)...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr_text, Xtr_behav)
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xva_text, Xva_behav)
            val_pred = val_out.argmax(dim=1).cpu().numpy()

        val_bal_acc = balanced_accuracy_score(yva, val_pred)
        if val_bal_acc > best_score:
            best_score = val_bal_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.4f} | Val BalAcc: {val_bal_acc:.4f}")

    print(f"\nTraining Complete! Best Val Balanced Accuracy: {best_score:.4f}")
    model.load_state_dict(best_state)

    # Test Evaluation
    model.eval()
    Xte_text = torch.FloatTensor(embeddings[idx_test]).to(device)
    Xte_behav = torch.FloatTensor(X_behav[idx_test]).to(device)
    yte = y[idx_test]

    with torch.no_grad():
        test_out = model(Xte_text, Xte_behav)
        probs = torch.softmax(test_out, dim=1)[:, 1].cpu().numpy()
        preds = test_out.argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(yte, preds)
    test_bal_acc = balanced_accuracy_score(yte, preds)
    test_f1 = f1_score(yte, preds, zero_division=0)
    test_prec = precision_score(yte, preds, zero_division=0)
    test_rec = recall_score(yte, preds, zero_division=0)
    test_auroc = roc_auc_score(yte, probs)

    print("\n" + "="*50)
    print("FUSION MODEL TEST EVALUATION RESULTS:")
    print("="*50)
    print(f"Accuracy          : {test_acc:.4f}")
    print(f"Balanced Accuracy : {test_bal_acc:.4f}")
    print(f"F1-Score          : {test_f1:.4f}")
    print(f"Precision         : {test_prec:.4f}")
    print(f"Recall            : {test_rec:.4f}")
    print(f"AUROC             : {test_auroc:.4f}")
    print("="*50)

    # Save model checkpoint
    os.makedirs("models/fusion", exist_ok=True)
    save_path = "models/fusion/fusion_model.pt"
    torch.save(best_state, save_path)
    print(f"\nSuccessfully saved trained fusion checkpoint to: {save_path}")


if __name__ == "__main__":
    train_and_save_fusion()
