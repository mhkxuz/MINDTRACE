"""
MentalBERT text classification model and 768-d embedding extraction service.
Faithfully reproduces the text modeling logic from:
notebooks/text_model.ipynb and notebooks/fusion_model.ipynb
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
import torch
import numpy as np

# Exact 7 classes from text_model.ipynb (le_text.classes_)
TEXT_CLASSES = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

# Benchmark metrics from test set evaluation in text_model.ipynb
TEXT_BENCHMARK_METRICS = {
    "dataset": "Kaggle Sentiment Analysis for Mental Health",
    "dataset_size": 52680,
    "test_accuracy": 0.8426,
    "macro_f1": 0.8290,
    "weighted_f1": 0.8421,
    "epochs_trained": 3,
    "class_metrics": {
        "Anxiety": {"precision": 0.90, "recall": 0.86, "f1": 0.88},
        "Bipolar": {"precision": 0.88, "recall": 0.85, "f1": 0.87},
        "Depression": {"precision": 0.78, "recall": 0.81, "f1": 0.79},
        "Normal": {"precision": 0.95, "recall": 0.97, "f1": 0.96},
        "Personality disorder": {"precision": 0.85, "recall": 0.72, "f1": 0.78},
        "Stress": {"precision": 0.79, "recall": 0.80, "f1": 0.79},
        "Suicidal": {"precision": 0.75, "recall": 0.71, "f1": 0.73},
    },
}

MODEL_NAME = "mental/mental-bert-base-uncased"


class MentalBERTService:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or os.getenv(
            "TEXT_MODEL_PATH", "models/text/text_model_final.pt"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.classifier_model = None
        self.embedding_model = None
        self.is_classifier_loaded = False
        self.is_embedding_loaded = False

    def get_hf_token(self) -> Optional[str]:
        """Fetch Hugging Face token safely from environment."""
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    def load_tokenizer(self) -> bool:
        """Load tokenizer with optional authentication token."""
        if self.tokenizer is not None:
            return True
        try:
            from transformers import AutoTokenizer
            token = self.get_hf_token()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME,
                    token=token if token else True,
                    local_files_only=True
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME,
                    token=token if token else True
                )
            return True
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            return False

    def load_classifier(self) -> bool:
        """Load fine-tuned 7-class sequence classifier."""
        if not os.path.exists(self.checkpoint_path):
            self.is_classifier_loaded = False
            return False

        if not self.load_tokenizer():
            return False

        try:
            from transformers import AutoModelForSequenceClassification
            token = self.get_hf_token()
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(TEXT_CLASSES),
                    token=token if token else True,
                    local_files_only=True
                )
            except Exception:
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(TEXT_CLASSES),
                    token=token if token else True
                )
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.classifier_model = model
            self.is_classifier_loaded = True
            return True
        except Exception as e:
            print(f"Error loading text classifier checkpoint from {self.checkpoint_path}: {e}")
            self.is_classifier_loaded = False
            return False

    def load_embedding_encoder(self) -> bool:
        """
        Load frozen base MentalBERT model for generating 768-dim embeddings,
        matching Cell 6 of fusion_model.ipynb.
        """
        if self.embedding_model is not None:
            return True

        if not self.load_tokenizer():
            return False

        try:
            from transformers import AutoModel
            token = self.get_hf_token()
            try:
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    token=token if token else True,
                    local_files_only=True
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    MODEL_NAME,
                    token=token if token else True
                )
            model.to(self.device)
            model.eval()
            self.embedding_model = model
            self.is_embedding_loaded = True
            return True
        except Exception as e:
            print(f"Error loading base MentalBERT encoder: {e}")
            return False

    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract 768-dimensional embedding from text using the [CLS] representation,
        matching:
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        from fusion_model.ipynb.
        """
        if not self.load_embedding_encoder() or self.tokenizer is None:
            return None

        clean_t = text.strip() if text else "empty"
        inputs = self.tokenizer(
            clean_t,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Take CLS token embedding (shape: 1 x 768)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

        return cls_embedding.astype(np.float32)

    def predict(self, text_input: str) -> Dict[str, Any]:
        """
        Classify single text into 7 mental health categories.
        """
        if not os.path.exists(self.checkpoint_path):
            return {
                "success": False,
                "error": f"Text model checkpoint not found at '{self.checkpoint_path}'. Trained weights must be placed at this path.",
                "predicted_class": None,
                "probabilities": {},
                "benchmark": TEXT_BENCHMARK_METRICS,
            }

        if not self.is_classifier_loaded:
            loaded = self.load_classifier()
            if not loaded:
                return {
                    "success": False,
                    "error": f"Failed to initialize or load weights from '{self.checkpoint_path}'.",
                    "predicted_class": None,
                    "probabilities": {},
                    "benchmark": TEXT_BENCHMARK_METRICS,
                }

        clean_text = text_input.strip() if text_input else "empty"
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_dict = {
            TEXT_CLASSES[i]: float(probs[i])
            for i in range(len(TEXT_CLASSES))
        }
        pred_idx = int(np.argmax(probs))
        pred_class = TEXT_CLASSES[pred_idx]
        top_prob = float(probs[pred_idx])

        # Prediction uncertainty via normalized entropy and margin
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(len(TEXT_CLASSES)))
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0

        return {
            "success": True,
            "predicted_class": pred_class,
            "confidence": top_prob,
            "entropy": entropy,
            "margin": margin,
            "probabilities": prob_dict,
            "benchmark": TEXT_BENCHMARK_METRICS,
        }

    def predict_conversation(
        self,
        turns: List[Dict[str, str]],
        aggregate_mode: str = "user_only"
    ) -> Dict[str, Any]:
        """
        Analyze conversation text with turn-level breakdown and aggregate analysis.
        aggregate_mode: 'user_only' (concatenates user turns) or 'all' (all turns).
        """
        if not turns:
            return {
                "success": False,
                "error": "No conversation turns provided.",
            }

        if aggregate_mode == "user_only":
            relevant_turns = [t for t in turns if t.get("speaker") in ["user", "seeker"]]
        else:
            relevant_turns = turns

        if not relevant_turns:
            relevant_turns = turns

        aggregated_text = " ".join([t.get("message", "") for t in relevant_turns])
        agg_result = self.predict(aggregated_text)

        # Optional turn-by-turn predictions for user turns
        turn_predictions = []
        if self.is_classifier_loaded:
            for idx, turn in enumerate(turns):
                if turn.get("speaker") in ["user", "seeker"] and turn.get("message"):
                    res = self.predict(turn["message"])
                    if res.get("success"):
                        turn_predictions.append({
                            "turn_index": idx + 1,
                            "message": turn["message"],
                            "predicted_class": res["predicted_class"],
                            "confidence": res["confidence"],
                        })

        return {
            "success": agg_result.get("success", False),
            "aggregate_mode": aggregate_mode,
            "aggregated_text_length": len(aggregated_text),
            "aggregate_prediction": agg_result,
            "turn_predictions": turn_predictions,
            "benchmark": TEXT_BENCHMARK_METRICS,
        }
