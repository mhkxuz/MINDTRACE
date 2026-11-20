"""
Multimodal Text + Behavioral Fusion Architecture and Inference Service.
Faithfully reproduces the models and evaluation from:
notebooks/fusion_model.ipynb
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import joblib

# Binary distress classes
DISTRESS_CLASSES = [
    "Lower Distress",
    "Higher Distress",
]

# Experimental results extracted from fusion_model.ipynb (5-seed evaluation)
FIVE_SEED_BENCHMARK = {
    "Model A - Text Only": {
        "acc_mean": 0.554, "acc_std": 0.117,
        "bal_acc_mean": 0.527, "bal_acc_std": 0.033,
        "f1_mean": 0.649, "f1_std": 0.155,
        "precision_mean": 0.781, "precision_std": 0.020,
        "recall_mean": 0.577, "recall_std": 0.191,
        "auroc_mean": 0.496, "auroc_std": 0.023,
    },
    "Model B - Behavioral Only": {
        "acc_mean": 0.448, "acc_std": 0.030,
        "bal_acc_mean": 0.385, "bal_acc_std": 0.018,
        "f1_mean": 0.583, "f1_std": 0.035,
        "precision_mean": 0.695, "precision_std": 0.016,
        "recall_mean": 0.503, "recall_std": 0.043,
        "auroc_mean": 0.370, "auroc_std": 0.022,
    },
    "Model C - Fusion (Text + Behavioral)": {
        "acc_mean": 0.576, "acc_std": 0.035,
        "bal_acc_mean": 0.514, "bal_acc_std": 0.035,
        "f1_mean": 0.694, "f1_std": 0.034,
        "precision_mean": 0.777, "precision_std": 0.020,
        "recall_mean": 0.629, "recall_std": 0.049,
        "auroc_mean": 0.477, "auroc_std": 0.025,
    },
}

# 5-fold Cross-Validation benchmark from fusion_model.ipynb
FIVE_FOLD_CV_BENCHMARK = {
    "Model A - Text Only": {
        "bal_acc_mean": 0.556, "bal_acc_std": 0.039,
        "f1_mean": 0.655, "f1_std": 0.153,
        "auroc_mean": 0.573, "auroc_std": 0.046,
    },
    "Model B - Behavioral Only": {
        "bal_acc_mean": 0.525, "bal_acc_std": 0.009,
        "f1_mean": 0.527, "f1_std": 0.280,
        "auroc_mean": 0.506, "auroc_std": 0.073,
    },
    "Model C - Fusion (Text + Behavioral)": {
        "bal_acc_mean": 0.549, "bal_acc_std": 0.043,
        "f1_mean": 0.710, "f1_std": 0.052,
        "auroc_mean": 0.559, "auroc_std": 0.052,
    },
    "paired_ttest_C_vs_A": {
        "mean_diff": -0.0138,
        "p_value": 0.2957,
        "interpretation": "Not statistically significant at alpha=0.05",
    },
}


class TextOnlyModel(nn.Module):
    """Model A: Text-only baseline (768 -> 128 -> 2). Total params: 98,690."""
    def __init__(self, text_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)


class BehavioralOnlyModel(nn.Module):
    """Model B: Behavioral-only baseline (8 -> 32 -> 16 -> 2). Total params: 850."""
    def __init__(self, behav_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(behav_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, behav_feat: torch.Tensor) -> torch.Tensor:
        return self.net(behav_feat)


class FusionModelWeighted(nn.Module):
    """
    Model C: Weighted Fusion classifier (768 text + 8 behavioral = 776 input dims).
    Total params: 25,970.
    """
    def __init__(self, text_dim: int = 768, behav_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.behav_branch = nn.Sequential(
            nn.Linear(behav_dim, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
        )

    def forward(self, text_emb: torch.Tensor, behav_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_branch(text_emb)
        b = self.behav_branch(behav_feat)
        combined = torch.cat([t, b], dim=1)
        return self.classifier(combined)


class FusionService:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path or os.getenv(
            "FUSION_MODEL_PATH", "models/fusion/fusion_model.pt"
        )
        self.scaler_path = scaler_path or os.getenv(
            "BEHAV_SCALER_PATH", "artifacts/scalers/behav_scaler_esconv.pkl"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FusionModelWeighted] = None
        self.scaler = None
        self.is_model_loaded = False
        self.is_scaler_loaded = False

        self.load_scaler()
        self.load_model()

    def load_scaler(self) -> bool:
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                self.is_scaler_loaded = True
                return True
            except Exception as e:
                print(f"Error loading behavioral scaler from {self.scaler_path}: {e}")
                self.scaler = None
                self.is_scaler_loaded = False
                return False
        return False

    def load_model(self) -> bool:
        if os.path.exists(self.checkpoint_path):
            try:
                model = FusionModelWeighted(text_dim=768, behav_dim=8, num_classes=2)
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
                self.is_model_loaded = True
                return True
            except Exception as e:
                print(f"Error loading fusion model checkpoint from {self.checkpoint_path}: {e}")
                self.model = None
                self.is_model_loaded = False
                return False
        return False

    def predict(
        self,
        text_embedding: np.ndarray,
        raw_behavioral_vector: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run inference on text embedding (768,) and raw behavioral vector (8,).
        """
        if not self.is_model_loaded or self.model is None:
            return {
                "success": False,
                "error": f"Fusion model checkpoint not found at '{self.checkpoint_path}'. Trained weights must be placed at this path.",
                "prediction": None,
                "probabilities": {},
                "benchmarks": {
                    "five_seed": FIVE_SEED_BENCHMARK,
                    "five_fold_cv": FIVE_FOLD_CV_BENCHMARK,
                },
            }

        if text_embedding is None or text_embedding.shape[-1] != 768:
            return {
                "success": False,
                "error": f"Invalid text embedding dimension: expected 768, got {None if text_embedding is None else text_embedding.shape}.",
                "prediction": None,
                "probabilities": {},
            }

        if raw_behavioral_vector is None or raw_behavioral_vector.shape[-1] != 8:
            return {
                "success": False,
                "error": f"Invalid behavioral feature vector dimension: expected 8, got {None if raw_behavioral_vector is None else raw_behavioral_vector.shape}.",
                "prediction": None,
                "probabilities": {},
            }

        try:
            # Scale behavioral features
            b_vec = raw_behavioral_vector.reshape(1, -1)
            if self.scaler is not None:
                b_scaled = self.scaler.transform(b_vec)
            else:
                b_scaled = b_vec  # fallback if scaler unpickling failed

            t_tensor = torch.FloatTensor(text_embedding).reshape(1, 768).to(self.device)
            b_tensor = torch.FloatTensor(b_scaled).to(self.device)

            with torch.no_grad():
                logits = self.model(t_tensor, b_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            prob_dict = {
                DISTRESS_CLASSES[0]: float(probs[0]),
                DISTRESS_CLASSES[1]: float(probs[1]),
            }
            pred_idx = int(np.argmax(probs))
            predicted_class = DISTRESS_CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            return {
                "success": True,
                "prediction": predicted_class,
                "confidence": confidence,
                "distress_probability": float(probs[1]),
                "probabilities": prob_dict,
                "input_dimensions": {
                    "text_dim": 768,
                    "behav_dim": 8,
                    "total_fusion_dim": 776,
                },
                "benchmarks": {
                    "five_seed": FIVE_SEED_BENCHMARK,
                    "five_fold_cv": FIVE_FOLD_CV_BENCHMARK,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fusion inference failed: {str(e)}",
                "prediction": None,
                "probabilities": {},
            }
