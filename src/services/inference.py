"""
Unified inference service for the MINDTRACE Research Platform.
Coordinates text, facial, behavioral, and multimodal fusion modalities.
"""

import os
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv

load_dotenv()
from PIL import Image
import numpy as np
import pandas as pd

from src.utils.config import (
    get_model_repo,
    get_artifact_url,
    get_hf_token,
)
from src.features.behavioral import (
    extract_behavioral_features,
    get_fusion_feature_vector,
    parse_conversation_transcript,
    process_conversation_dataframe,
    FUSION_BEHAVIORAL_COLS,
)
from src.models.facial_model import (
    FacialEmotionService,
    EMOTION_LABELS,
)
from src.models.text_model import (
    MentalBERTService,
    TEXT_CLASSES,
)
from src.models.fusion_model import (
    FusionService,
)


class MindTracePlatform:
    """
    Central orchestration service for MINDTRACE.
    Manages lazy-loaded models, artifact inspection, and unified predictions.
    """
    def __init__(self):
        self.text_service = MentalBERTService()
        self.facial_service = FacialEmotionService()
        self.fusion_service = FusionService()

    def get_system_status(self) -> Dict[str, Any]:
        """Inspect disk checkpoints, cloud configuration, and operational readiness of all models."""
        text_ckpt_exists = self.text_service.is_classifier_available()
        base_encoder_available = self.text_service.is_base_encoder_available()
        facial_ckpt_exists = os.path.exists(self.facial_service.checkpoint_path)
        fusion_ckpt_exists = os.path.exists(self.fusion_service.checkpoint_path)
        scaler_exists = os.path.exists(self.fusion_service.scaler_path)

        # Status text determination
        if text_ckpt_exists:
            text_status_text = "Ready for Inference (Local Best Checkpoint - Epoch 2)"
        elif get_model_repo("text") or get_artifact_url("text_model"):
            text_status_text = "Cloud Configured (Will Download on Demand)"
        else:
            text_status_text = "Checkpoint Missing on Disk"

        if base_encoder_available:
            base_status_text = "Available in Local Cache"
        elif get_hf_token():
            base_status_text = "Cloud Configured (Will Download via HF_TOKEN)"
        else:
            base_status_text = "Missing in Local Cache"

        if facial_ckpt_exists:
            facial_status_text = "Ready for Inference"
        elif get_model_repo("artifacts") or get_artifact_url("fer_model"):
            facial_status_text = "Cloud Configured (Will Download on Demand)"
        else:
            facial_status_text = "Checkpoint Missing on Disk"

        if fusion_ckpt_exists and scaler_exists:
            fusion_status_text = "Ready for Inference"
        elif get_model_repo("artifacts") or get_artifact_url("fusion_model"):
            fusion_status_text = "Cloud Configured (Will Download on Demand)"
        else:
            fusion_status_text = "Checkpoint Missing on Disk"

        return {
            "text_model": {
                "name": "MentalBERT 7-Class Classifier (Fine-Tuned Safetensors)",
                "checkpoint_path": self.text_service.checkpoint_path,
                "checkpoint_exists": text_ckpt_exists,
                "is_loaded": self.text_service.is_classifier_loaded,
                "status": text_status_text,
                "modality": "Conversational Text Categorization",
                "classes": TEXT_CLASSES,
                "benchmark_accuracy": "83.30%",
                "benchmark_macro_f1": "80.88%",
                "suicidal_recall": "75.77%",
                "benchmark_type": "Reproduced held-out dataset benchmark (Kaggle sentiment test partition)",
            },
            "text_base_encoder": {
                "name": "Frozen Base MentalBERT Encoder (mental/mental-bert-base-uncased)",
                "role": "768-d CLS text feature extraction for Multimodal Distress Fusion",
                "is_available": base_encoder_available,
                "is_loaded": self.text_service.is_embedding_loaded,
                "status": base_status_text,
                "modality": "Frozen Base CLS Embedding (768 dims)",
            },
            "facial_model": {
                "name": "FER2013 3-Block CNN",
                "checkpoint_path": self.facial_service.checkpoint_path,
                "checkpoint_exists": facial_ckpt_exists,
                "is_loaded": self.facial_service.is_loaded,
                "status": facial_status_text,
                "modality": "Facial Affect (Static Image / Frame)",
                "classes": EMOTION_LABELS,
                "benchmark_accuracy": "58.79%",
                "benchmark_macro_f1": "56.47%",
            },
            "behavioral_module": {
                "name": "Interaction Feature Extractor",
                "scaler_path": self.fusion_service.scaler_path,
                "scaler_exists": scaler_exists,
                "is_loaded": True,
                "status": "Fully Operational (Rule & Regression Engine)",
                "modality": "Dialogue Dynamics",
                "feature_count": 8,
            },
            "fusion_model": {
                "name": "Text + Behavioral Distress Fusion (Model C)",
                "checkpoint_path": self.fusion_service.checkpoint_path,
                "checkpoint_exists": fusion_ckpt_exists,
                "is_loaded": self.fusion_service.is_model_loaded,
                "status": fusion_status_text,
                "modality": "Multimodal (768 base text + 8 behavioral = 776 dims)",
                "benchmark_accuracy": "57.6%",
                "benchmark_f1": "0.694",
                "benchmark_auroc": "0.477",
            },
            "proposed_tri_modal": {
                "name": "Proposed 3-Branch Architecture (783 dims)",
                "status": "Proposed Architecture (Not Yet Implemented/Validated)",
                "modality": "Base Text (768) + Behavioral (8) + Facial Affect (7)",
                "is_loaded": False,
            },
        }

    # Text Analysis
    def analyze_text(self, text: str) -> Dict[str, Any]:
        return self.text_service.predict(text)

    def analyze_text_conversation(
        self,
        transcript_or_turns: Union[str, List[Dict[str, str]]],
        aggregate_mode: str = "user_only"
    ) -> Dict[str, Any]:
        if isinstance(transcript_or_turns, str):
            turns = parse_conversation_transcript(transcript_or_turns)
        else:
            turns = transcript_or_turns
        return self.text_service.predict_conversation(turns, aggregate_mode=aggregate_mode)

    # Facial Emotion Analysis
    def analyze_facial_image(self, image_input: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        return self.facial_service.predict(image_input)

    # Behavioral Feature Extraction
    def analyze_behavioral_transcript(self, transcript_text: str) -> Dict[str, Any]:
        turns = parse_conversation_transcript(transcript_text)
        features = extract_behavioral_features(turns)
        vector = get_fusion_feature_vector(features)
        return {
            "features": features,
            "fusion_vector": vector,
            "fusion_vector_cols": FUSION_BEHAVIORAL_COLS,
        }

    def analyze_behavioral_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        return process_conversation_dataframe(df)

    # Multimodal Fusion Analysis
    def run_fusion_analysis(
        self,
        transcript_text: str,
        situation_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full two-branch fusion pipeline:
        1. Extract behavioral features from dialogue turns.
        2. Extract 768-d text embedding from situation text (or user dialogue text).
        3. Pass 776-dimensional representation to FusionModelWeighted.
        """
        behav_result = self.analyze_behavioral_transcript(transcript_text)
        features = behav_result["features"]
        raw_b_vec = behav_result["fusion_vector"]

        # Text selection: if situation text provided (like ESConv 'situation'), use it; else aggregate user messages
        if situation_text and situation_text.strip():
            target_text = situation_text.strip()
        else:
            user_msgs = [t["message"] for t in features.get("turns_data", []) if t["speaker"] == "user"]
            target_text = " ".join(user_msgs) if user_msgs else transcript_text

        text_emb = self.text_service.extract_text_embedding(target_text)

        if text_emb is None:
            return {
                "success": False,
                "error": "Failed to extract 768-d text embedding using MentalBERT encoder.",
                "behavioral_features": features,
                "text_embedding": None,
                "fusion_prediction": None,
            }

        fusion_pred = self.fusion_service.predict(text_emb, raw_b_vec)

        return {
            "success": fusion_pred.get("success", False),
            "target_text": target_text,
            "behavioral_features": features,
            "raw_behavioral_vector": raw_b_vec.tolist(),
            "fusion_vector_cols": FUSION_BEHAVIORAL_COLS,
            "fusion_prediction": fusion_pred,
            "input_dimensions": {
                "text_dim": 768,
                "behav_dim": 8,
                "total_fusion_dim": 776,
            },
        }
