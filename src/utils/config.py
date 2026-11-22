"""
MINDTRACE Configuration & Secrets Manager
Handles secure configuration and credential retrieval across local and cloud environments.
Supports Streamlit Community Cloud Secrets (st.secrets), environment variables, and local .env files.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Ensure local .env is loaded if present
load_dotenv()


def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a configuration value or secret.
    Priority order:
    1. Streamlit Cloud Secrets (st.secrets) if running within Streamlit runtime.
    2. OS Environment variables.
    3. Default value provided.
    Never raises an exception if secrets file is absent.
    """
    # 1. Check Streamlit Secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass

    # 2. Check OS environment variables
    env_val = os.getenv(key)
    if env_val is not None and str(env_val).strip():
        return str(env_val).strip()

    return default


def get_hf_token() -> Optional[str]:
    """
    Retrieve Hugging Face API access token securely from Streamlit Secrets or environment.
    Never prints or logs the token.
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface_token"):
        token = get_secret_or_env(key)
        if token:
            return token
    return None


def get_model_repo(repo_type: str = "all") -> Optional[str]:
    """
    Retrieve configured Hugging Face repository containing fine-tuned model or artifacts.
    Keys checked:
    - For text: HF_TEXT_MODEL_REPO, MINDTRACE_TEXT_MODEL_REPO, HF_MODEL_REPO, MINDTRACE_MODEL_REPO
    - For artifacts: HF_ARTIFACTS_REPO, MINDTRACE_ARTIFACTS_REPO, HF_MODEL_REPO, MINDTRACE_MODEL_REPO
    - For all: HF_MODEL_REPO, MINDTRACE_MODEL_REPO
    """
    if repo_type == "text":
        keys = ["HF_TEXT_MODEL_REPO", "MINDTRACE_TEXT_MODEL_REPO", "HF_MODEL_REPO", "MINDTRACE_MODEL_REPO"]
    elif repo_type == "artifacts":
        keys = ["HF_ARTIFACTS_REPO", "MINDTRACE_ARTIFACTS_REPO", "HF_MODEL_REPO", "MINDTRACE_MODEL_REPO"]
    else:
        keys = ["HF_MODEL_REPO", "MINDTRACE_MODEL_REPO", "HF_TEXT_MODEL_REPO", "HF_ARTIFACTS_REPO"]

    for k in keys:
        repo = get_secret_or_env(k)
        if repo:
            return repo
    return None


def get_artifact_url(artifact_name: str) -> Optional[str]:
    """
    Retrieve optional direct URL for downloading a specific model artifact.
    artifact_name: 'fer_model', 'fusion_model', 'behav_scaler', 'text_model'
    """
    key_mapping = {
        "fer_model": ["FER_MODEL_URL", "MINDTRACE_FER_MODEL_URL"],
        "fusion_model": ["FUSION_MODEL_URL", "MINDTRACE_FUSION_MODEL_URL"],
        "behav_scaler": ["BEHAV_SCALER_URL", "MINDTRACE_BEHAV_SCALER_URL"],
        "text_model": ["TEXT_MODEL_URL", "MINDTRACE_TEXT_MODEL_URL"],
    }
    keys = key_mapping.get(artifact_name, [f"{artifact_name.upper()}_URL"])
    for k in keys:
        url = get_secret_or_env(k)
        if url:
            return url
    return None


def is_cloud_environment() -> bool:
    """
    Detect whether the application is running in Streamlit Community Cloud
    or another cloud container runtime.
    """
    cloud_indicators = [
        "STREAMLIT_SHARING_MODE",
        "STREAMLIT_SERVER_HEADLESS",
        "IS_STREAMLIT_CLOUD",
        "STREAMLIT_CLOUD",
        "HEROKU",
        "RENDER",
        "SPACE_ID",  # Hugging Face Spaces
    ]
    for ind in cloud_indicators:
        val = os.getenv(ind, "").lower()
        if val in ("1", "true", "yes", "online"):
            return True

    # Check if running in typical Streamlit Cloud mount path
    if os.path.exists("/mount/src") or os.getenv("HOME", "").startswith("/home/appuser"):
        return True

    return False


def mask_token(token: Optional[str]) -> str:
    """
    Safely format a token for UI presentation without exposing credential characters.
    """
    if not token or len(token) < 8:
        return "Not Configured"
    return f"{token[:4]}...{token[-4:]} (Configured)"
