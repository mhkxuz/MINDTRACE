"""
MentalBERT text classification model and 768-d embedding extraction service.
Faithfully reproduces the text modeling logic from:
notebooks/text_model.ipynb, notebooks/train_text_model_colab.ipynb, and notebooks/fusion_model.ipynb

Architectural Design (Decoupled Dual-Encoder Pipeline):
1. 7-Class Mental Health Text Categorization (Page 2):
   - Uses the fine-tuned MentalBERT classifier (Epoch 2 best checkpoint) loaded locally
     from models/text/mentalbert_best_export/ (config.json, model.safetensors, tokenizer.json).
   - Operates offline via local_files_only=True without internet downloads during startup.
2. Multimodal Distress Fusion Feature Extraction (Page 5):
   - Uses the frozen base MentalBERT encoder (mental/mental-bert-base-uncased) loaded
     from local Hugging Face cache (local_files_only=True).
   - Preserves exact 768-dimensional CLS embedding extraction protocol from fusion_model.ipynb Cell 7,
     preventing representation degradation from fine-tuning distribution shifts (cosine sim ~0.3120).
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

# Historical benchmark metrics from original text_model.ipynb experiment
TEXT_HISTORICAL_BENCHMARK_METRICS = {
    "dataset": "Kaggle Sentiment Analysis for Mental Health (Original)",
    "dataset_size": 52680,
    "test_accuracy": 0.8426,
    "macro_f1": 0.8290,
    "weighted_f1": 0.8421,
    "suicidal_recall": 0.7100,
    "epochs_trained": 3,
    "benchmark_type": "Original Colab Experiment Benchmark",
}

# Reproduced held-out test evaluation metrics (from models/text/mentalbert_best_export/training_metadata.json)
TEXT_REPRODUCED_BENCHMARK_METRICS = {
    "dataset": "Kaggle Sentiment Analysis for Mental Health (52,680 cleaned records)",
    "dataset_size": 52680,
    "test_split_size": 5268,
    "selected_checkpoint": "text_model_best.pt",
    "selected_epoch": 2,
    "test_accuracy": 0.8330,
    "macro_f1": 0.8088,
    "weighted_f1": 0.8325,
    "suicidal_recall": 0.7577,
    "benchmark_type": "Reproduced Held-Out Dataset Benchmark",
    "scientific_qualification": (
        "Evaluated on 5,268 held-out test samples from the Kaggle sentiment dataset. "
        "Represents statistical text categorization performance across dataset-defined categories, "
        "not a clinical diagnosis or validated clinical suicide risk assessment."
    ),
    "class_names": TEXT_CLASSES,
    "class_metrics": {
        "Anxiety": {"precision": 0.90, "recall": 0.86, "f1": 0.88},
        "Bipolar": {"precision": 0.88, "recall": 0.85, "f1": 0.87},
        "Depression": {"precision": 0.78, "recall": 0.81, "f1": 0.79},
        "Normal": {"precision": 0.95, "recall": 0.97, "f1": 0.96},
        "Personality disorder": {"precision": 0.85, "recall": 0.72, "f1": 0.78},
        "Stress": {"precision": 0.79, "recall": 0.80, "f1": 0.79},
        "Suicidal": {"precision": 0.75, "recall": 0.7577, "f1": 0.7538},
    },
}

# Default benchmark alias for backward compatibility
TEXT_BENCHMARK_METRICS = TEXT_REPRODUCED_BENCHMARK_METRICS

DEFAULT_EXPORTED_MODEL_DIR = "models/text/mentalbert_best_export"
DEFAULT_BASE_MODEL_NAME = "mental/mental-bert-base-uncased"


class MentalBERTService:
    """
    Decoupled dual-encoder service:
    - Fine-tuned 7-class classifier (Safetensors) for conversational text categorization.
    - Frozen base MentalBERT encoder for 768-dimensional CLS feature extraction for multimodal fusion.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        # 1. Path configuration: Prefer exported safetensors model directory if present
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        elif os.path.isdir(DEFAULT_EXPORTED_MODEL_DIR):
            self.checkpoint_path = DEFAULT_EXPORTED_MODEL_DIR
        else:
            self.checkpoint_path = os.getenv("TEXT_MODEL_PATH", DEFAULT_EXPORTED_MODEL_DIR)

        self.base_model_path = base_model_path or os.getenv(
            "BASE_MENTALBERT_PATH", DEFAULT_BASE_MODEL_NAME
        )

        # 2. Device selection
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Offline enforcement: online download allowed only behind explicit opt-in
        self.allow_online_download = os.getenv(
            "ALLOW_ONLINE_MODEL_DOWNLOAD", "0"
        ).strip().lower() in ("1", "true", "yes")

        # 4. Decoupled lazy models
        self.classifier_tokenizer = None
        self.classifier_model = None
        self.is_classifier_loaded = False

        self.base_tokenizer = None
        self.base_embedding_model = None
        self.is_embedding_loaded = False

    def is_classifier_available(self) -> bool:
        """
        Check whether the fine-tuned classifier is present on disk
        without loading heavy model weights.
        """
        path = self.checkpoint_path
        if not path:
            return False

        if os.path.isdir(path):
            has_config = os.path.exists(os.path.join(path, "config.json"))
            has_weights = (
                os.path.exists(os.path.join(path, "model.safetensors"))
                or os.path.exists(os.path.join(path, "pytorch_model.bin"))
            )
            return has_config and has_weights
        elif os.path.isfile(path):
            return os.path.exists(path)

        return False

    def is_base_encoder_available(self) -> bool:
        """
        Check whether the frozen base MentalBERT encoder is available locally in the
        local Hugging Face cache or local filesystem without loading tensor weights.
        """
        # If it's a local directory path
        if os.path.isdir(self.base_model_path):
            return os.path.exists(os.path.join(self.base_model_path, "config.json"))

        # Check standard Hugging Face cache location
        hf_cache_hub = os.path.expanduser("~/.cache/huggingface/hub")
        expected_repo = "models--" + self.base_model_path.replace("/", "--")
        repo_dir = os.path.join(hf_cache_hub, expected_repo)
        if os.path.exists(repo_dir):
            snapshots_dir = os.path.join(repo_dir, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                return True

        # If online download is explicitly opted-in, consider it resolvable
        if self.allow_online_download:
            return True

        return False

    def load_classifier(self) -> bool:
        """
        Load fine-tuned 7-class sequence classifier from local exported model directory
        or legacy .pt checkpoint file.
        Strictly enforces offline loading by default (local_files_only=True).
        """
        if self.is_classifier_loaded and self.classifier_model is not None:
            return True

        if not self.is_classifier_available():
            self.is_classifier_loaded = False
            return False

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            local_only = not self.allow_online_download

            # Case A: Local directory export (models/text/mentalbert_best_export/)
            if os.path.isdir(self.checkpoint_path):
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint_path,
                    local_files_only=local_only,
                )
                self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    self.checkpoint_path,
                    local_files_only=local_only,
                )
            # Case B: Legacy single .pt state-dict file
            elif os.path.isfile(self.checkpoint_path):
                base_name = self.base_model_path
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                    base_name,
                    local_files_only=local_only,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_name,
                    num_labels=len(TEXT_CLASSES),
                    local_files_only=local_only,
                )
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.classifier_model = model
            else:
                raise FileNotFoundError(
                    f"Text classifier checkpoint not found at: {self.checkpoint_path}"
                )

            self.classifier_model.to(self.device)
            self.classifier_model.eval()
            self.is_classifier_loaded = True
            return True

        except Exception as e:
            print(f"Error loading fine-tuned text classifier from {self.checkpoint_path}: {e}")
            self.is_classifier_loaded = False
            return False

    def load_embedding_encoder(self) -> bool:
        """
        Load frozen base MentalBERT model for generating 768-dim embeddings for multimodal fusion,
        strictly matching Cell 6 / Cell 7 of notebooks/fusion_model.ipynb.
        Enforces local offline loading from local cache (local_files_only=True).
        """
        if self.is_embedding_loaded and self.base_embedding_model is not None:
            return True

        if not self.is_base_encoder_available():
            raise FileNotFoundError(
                f"Base MentalBERT model '{self.base_model_path}' not found in local cache. "
                "Fusion requires the frozen base encoder to preserve exact feature alignment."
            )

        try:
            from transformers import AutoTokenizer, AutoModel

            local_only = not self.allow_online_download

            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                local_files_only=local_only,
            )
            model = AutoModel.from_pretrained(
                self.base_model_path,
                local_files_only=local_only,
            )
            model.to(self.device)
            model.eval()
            self.base_embedding_model = model
            self.is_embedding_loaded = True
            return True

        except Exception as e:
            print(f"Error loading base MentalBERT encoder from {self.base_model_path}: {e}")
            self.is_embedding_loaded = False
            return False

    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract 768-dimensional CLS token embedding using the FROZEN BASE MentalBERT encoder,
        strictly reproducing Cell 6 / Cell 7 of notebooks/fusion_model.ipynb.

        Returns:
            np.ndarray: Finite float32 array of shape (768,).
        """
        if not self.load_embedding_encoder() or self.base_tokenizer is None:
            return None

        clean_t = str(text).strip() if text else "empty"
        inputs = self.base_tokenizer(
            clean_t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.base_embedding_model(**inputs)
            cls_embedding = (
                outputs.last_hidden_state[:, 0, :]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return cls_embedding

    def predict(self, text_input: str) -> Dict[str, Any]:
        """
        Classify single text into 7 mental health categories using the fine-tuned
        MentalBERT classifier (models/text/mentalbert_best_export).
        """
        if not self.is_classifier_available():
            return {
                "success": False,
                "error": (
                    f"Text model artifact not found at '{self.checkpoint_path}'. "
                    "Ensure models/text/mentalbert_best_export/ contains config.json and model.safetensors."
                ),
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

        clean_text = str(text_input).strip() if text_input else "empty"
        inputs = self.classifier_tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
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
            "model_info": {
                "architecture": "MentalBERT 7-Class Sequence Classifier",
                "checkpoint": "Epoch 2 Best Export (Safetensors)",
                "held_out_accuracy": "83.30%",
                "held_out_macro_f1": "80.88%",
                "held_out_suicidal_recall": "75.77%",
                "qualification": "Statistical text categorization on Kaggle test set; not a clinical diagnosis.",
            },
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
