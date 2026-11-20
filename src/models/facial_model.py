"""
Facial Emotion Recognition CNN architecture and inference service.
Faithfully reproduces the custom PyTorch CNN from:
notebooks/facial_emotion_label.ipynb
"""

import os
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Emotion labels in the exact order from facial_emotion_label.ipynb
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Benchmark metrics from test evaluation in facial_emotion_label.ipynb
FER_BENCHMARK_METRICS = {
    "dataset": "FER2013",
    "test_samples": 7178,
    "test_accuracy": 0.5879,
    "macro_f1": 0.5647,
    "weighted_f1": 0.59,
    "epochs_trained": 10,
    "class_f1": {
        "angry": 0.51,
        "disgust": 0.51,
        "fear": 0.43,
        "happy": 0.80,
        "neutral": 0.55,
        "sad": 0.41,
        "surprise": 0.74,
    },
}


class FacialEmotionCNN(nn.Module):
    """
    3-block Convolutional Neural Network for 48x48 facial emotion classification.
    Total parameters: 1,274,951.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_inference_transforms():
    """Exact test transform pipeline from facial_emotion_label.ipynb."""
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class FacialEmotionService:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or os.getenv(
            "FER_MODEL_PATH", "models/facial/fer_model.pt"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FacialEmotionCNN] = None
        self.is_loaded = False
        self.transform = get_inference_transforms()
        self.load_checkpoint()

    def load_checkpoint(self) -> bool:
        if os.path.exists(self.checkpoint_path):
            try:
                model = FacialEmotionCNN(num_classes=len(EMOTION_LABELS))
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Error loading facial model checkpoint from {self.checkpoint_path}: {e}")
                self.model = None
                self.is_loaded = False
                return False
        else:
            self.model = None
            self.is_loaded = False
            return False

    def predict(self, image_input: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        image_input can be a PIL Image, numpy array (OpenCV BGR or RGB), or path string.
        """
        if not self.is_loaded or self.model is None:
            return {
                "success": False,
                "error": f"Facial model checkpoint not found at '{self.checkpoint_path}'. Trained weights must be placed at this path.",
                "dominant_emotion": None,
                "probabilities": {},
            }

        try:
            if isinstance(image_input, str):
                pil_img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 2:
                    pil_img = Image.fromarray(image_input).convert("RGB")
                elif image_input.shape[2] == 3:
                    # Assume RGB
                    pil_img = Image.fromarray(image_input)
                else:
                    pil_img = Image.fromarray(image_input[:, :, :3])
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                return {
                    "success": False,
                    "error": "Unsupported image format provided.",
                    "dominant_emotion": None,
                    "probabilities": {},
                }

            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_img)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            prob_dict = {
                label: float(probs[i])
                for i, label in enumerate(EMOTION_LABELS)
            }
            dominant_idx = int(np.argmax(probs))
            dominant_emotion = EMOTION_LABELS[dominant_idx]

            return {
                "success": True,
                "dominant_emotion": dominant_emotion,
                "confidence": float(probs[dominant_idx]),
                "probabilities": prob_dict,
                "benchmark": FER_BENCHMARK_METRICS,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference failed: {str(e)}",
                "dominant_emotion": None,
                "probabilities": {},
            }
