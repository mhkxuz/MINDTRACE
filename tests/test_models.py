"""
Unit tests for model architectures and services.
"""

import os
import torch
import numpy as np
import pytest
from src.models.facial_model import (
    FacialEmotionCNN,
    FacialEmotionService,
    EMOTION_LABELS,
    FER_BENCHMARK_METRICS,
)
from src.models.text_model import (
    MentalBERTService,
    TEXT_CLASSES,
    TEXT_BENCHMARK_METRICS,
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
