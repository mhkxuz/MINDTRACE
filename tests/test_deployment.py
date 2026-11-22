"""
Deployment and cloud configuration unit tests.
Verifies secrets resolution, artifact integrity verification, offline preservation,
and graceful error handling for missing models.
"""

import os
import re
from src.utils.config import (
    get_secret_or_env,
    get_hf_token,
    get_model_repo,
    mask_token,
    is_cloud_environment,
)
from src.utils.artifact_manager import (
    verify_pt_checkpoint,
    verify_scaler,
    verify_text_model_export,
    ensure_fine_tuned_mentalbert,
    ensure_facial_model,
    ensure_fusion_model,
    ensure_behavioral_scaler,
    DEFAULT_TEXT_EXPORT_DIR,
    DEFAULT_FER_MODEL_PATH,
    DEFAULT_FUSION_MODEL_PATH,
    DEFAULT_SCALER_PATH,
)


def test_config_retrieval_and_masking():
    """Verify secrets retrieval priority and token masking."""
    # Test fallback to default
    val = get_secret_or_env("NONEXISTENT_KEY_12345", default="fallback_val")
    assert val == "fallback_val"

    # Test get_model_repo default
    repo = get_model_repo("text")
    assert repo is None or isinstance(repo, str)

    # Test get_hf_token
    token = get_hf_token()
    assert token is None or isinstance(token, str)

    # Test masking of tokens
    dummy_token = "mock_dummy_token_for_unit_tests_9876"
    masked = mask_token(dummy_token)
    assert masked.startswith("mock...")
    assert masked.endswith("9876 (Configured)")
    assert dummy_token not in masked

    assert mask_token(None) == "Not Configured"
    assert mask_token("") == "Not Configured"
    assert mask_token("short") == "Not Configured"


def test_cloud_environment_detection():
    """Verify cloud detection handles flags and defaults safely."""
    # Test with custom flag
    os.environ["STREAMLIT_SHARING_MODE"] = "1"
    assert is_cloud_environment() is True
    del os.environ["STREAMLIT_SHARING_MODE"]


def test_artifact_integrity_verification_local():
    """Verify that all four required local artifacts pass integrity checks."""
    assert verify_text_model_export(DEFAULT_TEXT_EXPORT_DIR) is True
    assert verify_pt_checkpoint(DEFAULT_FER_MODEL_PATH, min_bytes=1_000_000) is True
    assert verify_pt_checkpoint(DEFAULT_FUSION_MODEL_PATH, min_bytes=10_000) is True
    assert verify_scaler(DEFAULT_SCALER_PATH) is True


def test_artifact_integrity_failure_on_invalid():
    """Verify that corrupt or missing files fail integrity checks gracefully."""
    assert verify_pt_checkpoint("nonexistent_path.pt") is False
    assert verify_scaler("nonexistent_path.pkl") is False
    assert verify_text_model_export("nonexistent_dir") is False


def test_ensure_artifacts_local_resolution():
    """
    Verify that ensure_* functions resolve immediately on disk
    without attempting any network downloads in local development.
    """
    s_text, msg_text = ensure_fine_tuned_mentalbert(DEFAULT_TEXT_EXPORT_DIR)
    assert s_text is True
    assert "verified on disk" in msg_text

    s_fer, msg_fer = ensure_facial_model(DEFAULT_FER_MODEL_PATH)
    assert s_fer is True
    assert "verified on disk" in msg_fer

    s_fus, msg_fus = ensure_fusion_model(DEFAULT_FUSION_MODEL_PATH)
    assert s_fus is True
    assert "verified on disk" in msg_fus

    s_scl, msg_scl = ensure_behavioral_scaler(DEFAULT_SCALER_PATH)
    assert s_scl is True
    assert "verified on disk" in msg_scl


def test_missing_artifact_graceful_handling():
    """
    Verify that when an artifact is missing and no cloud repo is configured,
    ensure_* returns False with actionable setup instructions.
    """
    fake_path = "models/facial/nonexistent_dummy.pt"
    # Temporarily ensure no repo is set for this test
    old_repo = os.environ.get("HF_ARTIFACTS_REPO")
    if "HF_ARTIFACTS_REPO" in os.environ:
        del os.environ["HF_ARTIFACTS_REPO"]

    try:
        success, msg = ensure_facial_model(fake_path)
        assert success is False
        assert "not found" in msg.lower() or "secrets" in msg.lower()
    finally:
        if old_repo:
            os.environ["HF_ARTIFACTS_REPO"] = old_repo


def test_templates_contain_no_real_secrets():
    """Verify template files exist and do not contain real Hugging Face tokens."""
    hf_token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")

    templates = [".env.example", ".streamlit/secrets.toml.example"]
    for tmpl in templates:
        assert os.path.isfile(tmpl), f"Missing template: {tmpl}"
        with open(tmpl) as f:
            content = f.read()
        matches = hf_token_pattern.findall(content)
        assert len(matches) == 0, f"Template {tmpl} contains a real token pattern!"


def test_deployment_configuration_files():
    """Verify requirements.txt and packages.txt for Streamlit Community Cloud."""
    assert os.path.isfile("packages.txt")
    with open("packages.txt") as f:
        pkg_content = f.read()
    assert "libgl1" in pkg_content

    with open("requirements.txt") as f:
        req_content = f.read()
    assert "opencv-python-headless" in req_content
    assert "streamlit" in req_content
    assert "huggingface-hub" in req_content
