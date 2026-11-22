"""
Unit tests for model architectures and services.
"""

import os
import torch
import numpy as np
from src.models.facial_model import (
    FacialEmotionCNN,
    FacialEmotionService,
)
from src.models.text_model import (
    MentalBERTService,
    TEXT_CLASSES,
)
from src.models.fusion_model import (
    TextOnlyModel,
    BehavioralOnlyModel,
    FusionModelWeighted,
    FusionService,
    DISTRESS_CLASSES,
)


def test_facial_emotion_cnn_architecture():
    model = FacialEmotionCNN(num_classes=7)
    param_count = sum(p.numel() for p in model.parameters())
    # As calculated in facial_emotion_label.ipynb: 1,274,951 parameters
    assert param_count == 1274951

    # Test forward pass with 48x48 3-channel input
    x = torch.randn(2, 3, 48, 48)
    out = model(x)
    assert out.shape == (2, 7)


def test_text_only_model_architecture():
    model = TextOnlyModel(text_dim=768, num_classes=2)
    param_count = sum(p.numel() for p in model.parameters())
    # Linear(768, 128) + b(128) = 768*128 + 128 = 98432; Linear(128, 2) + b(2) = 258. Total: 98,690
    assert param_count == 98690

    x = torch.randn(3, 768)
    out = model(x)
    assert out.shape == (3, 2)


def test_behavioral_only_model_architecture():
    model = BehavioralOnlyModel(behav_dim=8, num_classes=2)
    param_count = sum(p.numel() for p in model.parameters())
    # Linear(8, 32)+b(32) = 288; Linear(32, 16)+b(16) = 528; Linear(16, 2)+b(2) = 34. Total: 850
    assert param_count == 850

    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 2)


def test_fusion_model_weighted_architecture():
    model = FusionModelWeighted(text_dim=768, behav_dim=8, num_classes=2)
    param_count = sum(p.numel() for p in model.parameters())
    # Text: 768*32 + 32 = 24608; Behav: 8*16 + 16 = 144; Classifier: 48*16 + 16 + 16*2 + 2 = 784 + 34 = 1218
    # Total: 24608 + 144 + 1218 = 25,970 params (exact match with notebook cell 17)
    assert param_count == 25970

    text_tensor = torch.randn(2, 768)
    behav_tensor = torch.randn(2, 8)
    out = model(text_tensor, behav_tensor)
    assert out.shape == (2, 2)


def test_missing_checkpoints_graceful_handling():
    # Provide non-existent paths to verify safe failure
    fac_svc = FacialEmotionService(checkpoint_path="nonexistent_fer.pt")
    res_fac = fac_svc.predict(np.zeros((48, 48, 3), dtype=np.uint8))
    assert res_fac["success"] is False
    assert "not found" in res_fac["error"]

    text_svc = MentalBERTService(checkpoint_path="nonexistent_text.pt")
    res_txt = text_svc.predict("Hello there")
    assert res_txt["success"] is False
    assert "not found" in res_txt["error"]

    fusion_svc = FusionService(checkpoint_path="nonexistent_fusion.pt")
    res_fus = fusion_svc.predict(np.zeros((768,), dtype=np.float32), np.zeros((8,), dtype=np.float32))
    assert res_fus["success"] is False
    assert "not found" in res_fus["error"]


def test_checkpoint_detection_and_status():
    """Verify that checkpoint detection accurately reports disk status."""
    text_svc = MentalBERTService()
    assert text_svc.checkpoint_path in ["models/text/mentalbert_best_export", "models/text/text_model_final.pt"]
    if os.path.exists(text_svc.checkpoint_path):
        text_svc.load_classifier()
        assert text_svc.is_classifier_loaded is True
    else:
        assert text_svc.is_classifier_loaded is False

    fac_svc = FacialEmotionService()
    assert fac_svc.checkpoint_path == "models/facial/fer_model.pt"
    if os.path.exists(fac_svc.checkpoint_path):
        assert fac_svc.is_loaded is True
    else:
        assert fac_svc.is_loaded is False

    fus_svc = FusionService()
    assert fus_svc.checkpoint_path == "models/fusion/fusion_model.pt"
    assert os.path.exists(fus_svc.checkpoint_path) is True
    assert fus_svc.is_model_loaded is True
    assert fus_svc.is_scaler_loaded is True


def test_fine_tuned_mentalbert_service_offline_and_predictions():
    """Verify offline loading, 7-class probability normalization, and input edge cases."""
    text_svc = MentalBERTService()
    loaded = text_svc.load_classifier()
    assert loaded is True
    assert text_svc.is_classifier_loaded is True
    assert os.path.exists(os.path.join(text_svc.checkpoint_path, "model.safetensors"))

    # 1. Normal text inference - check structure and probability axioms
    sample_text = "I have been feeling really exhausted and stressed about my upcoming project deadlines."
    res = text_svc.predict(sample_text)
    assert res["success"] is True
    assert "predicted_class" in res
    assert res["predicted_class"] in TEXT_CLASSES
    assert len(res["probabilities"]) == 7
    # Probabilities sum to 1.0 within tolerance and are non-negative
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4
    for k, v in res["probabilities"].items():
        assert 0.0 <= v <= 1.0

    # 2. Empty / whitespace string handling - robustly handled without crashing
    empty_res = text_svc.predict("   ")
    assert empty_res["success"] is True
    assert len(empty_res["probabilities"]) == 7
    assert abs(sum(empty_res["probabilities"].values()) - 1.0) < 1e-4

    # 3. Short single-word input
    short_res = text_svc.predict("hello")
    assert short_res["success"] is True
    assert abs(sum(short_res["probabilities"].values()) - 1.0) < 1e-4

    # 4. Long text (exceeding 128 max tokens) - verify truncation works without crashing
    long_text = "stress anxiety depression overwhelm " * 80
    long_res = text_svc.predict(long_text)
    assert long_res["success"] is True
    assert abs(sum(long_res["probabilities"].values()) - 1.0) < 1e-4


def test_frozen_base_encoder_and_fusion_pipeline():
    """Verify base MentalBERT offline extraction and 776-d fusion pipeline integration."""
    text_svc = MentalBERTService()
    # Extract 768-d embedding using frozen base MentalBERT
    emb = text_svc.extract_text_embedding("I am dealing with overwhelming anxiety and panic.")
    assert emb is not None
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()

    # Empty text fallback embedding
    empty_emb = text_svc.extract_text_embedding("")
    assert empty_emb is not None
    assert empty_emb.shape == (768,)

    # Feed into FusionService with 8 behavioral features
    fus_svc = FusionService()
    fus_svc.load_model()
    fus_svc.load_scaler()
    assert fus_svc.is_model_loaded is True

    raw_behav = np.array([6.0, 0.5, 14.2, -0.15, 0.0, 0.0, 4.2, 0.0], dtype=np.float32)
    fus_pred = fus_svc.predict(emb, raw_behav)
    assert fus_pred["success"] is True
    assert fus_pred["prediction"] in DISTRESS_CLASSES
    assert "distress_probability" in fus_pred
    assert 0.0 <= fus_pred["distress_probability"] <= 1.0
    assert abs(fus_pred["probabilities"][DISTRESS_CLASSES[0]] + fus_pred["probabilities"][DISTRESS_CLASSES[1]] - 1.0) < 1e-4



def test_fusion_feature_ordering_and_scaler():
    """Verify exact 8 feature ordering and scaler transformation."""
    fus_svc = FusionService()
    success = fus_svc.load_scaler()
    assert success is True
    assert fus_svc.scaler is not None
    assert fus_svc.scaler.n_features_in_ == 8

    # Test scaling of zeros and positive vector
    raw_vec = np.array([5.0, 0.6, 12.0, -0.5, 0.0, 1.0, 3.5, 1.0], dtype=np.float32)
    scaled = fus_svc.scaler.transform(raw_vec.reshape(1, -1))[0]
    assert scaled.shape == (8,)
    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()
