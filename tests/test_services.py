"""
Integration tests for MindTracePlatform service layer.
"""

import os
from src.services.inference import MindTracePlatform


def test_platform_initialization_and_status():
    platform = MindTracePlatform()
    status = platform.get_system_status()

    assert "text_model" in status
    assert "facial_model" in status
    assert "behavioral_module" in status
    assert "fusion_model" in status
    assert "proposed_tri_modal" in status

    # Behavioral module is rule/regression based and always active
    assert status["behavioral_module"]["is_loaded"] is True
    assert status["proposed_tri_modal"]["is_loaded"] is False


def test_platform_behavioral_analysis():
    platform = MindTracePlatform()
    transcript = """
    User: I have been struggling to get out of bed this morning.
    Assistant: I hear you. Low energy can be really draining.
    User: yeah, nothing matters.
    Assistant: It's okay to feel that way. I am here with you.
    """
    res = platform.analyze_behavioral_transcript(transcript)
    assert "features" in res
    assert "fusion_vector" in res
    assert res["features"]["total_turn_count"] == 4
    assert res["features"]["user_turn_count"] == 2
    assert res["fusion_vector"].shape == (8,)


def test_platform_fusion_inference_live():
    platform = MindTracePlatform()
    if os.path.exists("models/fusion/fusion_model.pt"):
        res = platform.run_fusion_analysis("User: hello\nAssistant: hi", situation_text="Mild stress")
        assert res["success"] is True
        assert "prediction" in res["fusion_prediction"]
        assert res["input_dimensions"]["total_fusion_dim"] == 776
        assert res["fusion_prediction"]["prediction"] in ["Lower Distress", "Higher Distress"]
        assert 0.0 <= res["fusion_prediction"]["distress_probability"] <= 1.0
