"""
MINDTRACE Artifact & Model Manager
Handles verification, lazy downloading, and integrity checking for all model checkpoints
and scalers across local and Streamlit Community Cloud deployments.
"""

import os
import shutil
import urllib.request
from typing import Tuple, Optional

import torch
import joblib

from src.utils.config import (
    get_hf_token,
    get_model_repo,
    get_artifact_url,
)

DEFAULT_TEXT_EXPORT_DIR = "models/text/mentalbert_best_export"
DEFAULT_FER_MODEL_PATH = "models/facial/fer_model.pt"
DEFAULT_FUSION_MODEL_PATH = "models/fusion/fusion_model.pt"
DEFAULT_SCALER_PATH = "artifacts/scalers/behav_scaler_esconv.pkl"


def verify_pt_checkpoint(path: str, min_bytes: int = 1000) -> bool:
    """Verify that a PyTorch checkpoint exists, is non-trivial in size, and can be loaded."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        # Try loading state dict on CPU with safe weights_only flag
        state = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(state, dict) and len(state) > 0
    except Exception:
        # Fallback without weights_only if older PyTorch
        try:
            state = torch.load(path, map_location="cpu")
            return isinstance(state, dict) and len(state) > 0
        except Exception:
            return False


def verify_scaler(path: str) -> bool:
    """Verify that a scikit-learn StandardScaler pickle exists and is valid for 8 features."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < 50:
            return False
        scaler = joblib.load(path)
        return hasattr(scaler, "transform") and getattr(scaler, "n_features_in_", None) == 8
    except Exception:
        return False


def verify_text_model_export(dir_path: str) -> bool:
    """Verify that the fine-tuned MentalBERT export directory contains config and weights."""
    if not os.path.isdir(dir_path):
        return False

    has_config = os.path.isfile(os.path.join(dir_path, "config.json"))
    safetensors_path = os.path.join(dir_path, "model.safetensors")
    bin_path = os.path.join(dir_path, "pytorch_model.bin")

    has_weights = False
    if os.path.isfile(safetensors_path) and os.path.getsize(safetensors_path) > 10_000_000:
        has_weights = True
    elif os.path.isfile(bin_path) and os.path.getsize(bin_path) > 10_000_000:
        has_weights = True

    return has_config and has_weights


def download_file_from_url(url: str, destination_path: str, timeout: int = 60) -> bool:
    """
    Download a file from an HTTP/HTTPS URL into destination_path.
    Downloads to a temporary file first and renames atomically to prevent corrupted files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    tmp_path = destination_path + ".download.tmp"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MINDTRACE-Research-Platform/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        # Atomic move
        shutil.move(tmp_path, destination_path)
        return True
    except Exception as e:
        print(f"Error downloading file from URL '{url}' to '{destination_path}': {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def download_from_hf_hub(
    repo_id: str,
    filename: str,
    destination_path: str,
    token: Optional[str] = None
) -> bool:
    """
    Download an individual artifact from a Hugging Face repository using huggingface_hub.
    Supports subfolder search and direct filename mapping.
    """
    try:
        from huggingface_hub import hf_hub_download

        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)

        # Try exact filename first, then basename if full path was given
        candidates = [filename, os.path.basename(filename)]

        downloaded_file = None
        for cand in candidates:
            try:
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=cand,
                    token=token,
                )
                if downloaded_file and os.path.exists(downloaded_file):
                    break
            except Exception:
                continue

        if downloaded_file and os.path.exists(downloaded_file):
            shutil.copy(downloaded_file, destination_path)
            return True

        return False
    except Exception as e:
        print(f"Error downloading '{filename}' from Hugging Face repo '{repo_id}': {e}")
        return False


# =========================================================================
# High-Level Ensure Functions
# =========================================================================

def ensure_fine_tuned_mentalbert(target_dir: str = DEFAULT_TEXT_EXPORT_DIR) -> Tuple[bool, str]:
    """
    Ensure fine-tuned MentalBERT 7-class classifier is present and verified.
    If missing, attempts downloading from configured Hugging Face repository.
    """
    if verify_text_model_export(target_dir):
        return True, f"Fine-tuned MentalBERT export verified on disk at '{target_dir}'."

    os.makedirs(target_dir, exist_ok=True)
    repo_id = get_model_repo("text")
    token = get_hf_token()

    if repo_id:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                token=token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
            )
            if verify_text_model_export(target_dir):
                return True, f"Successfully downloaded fine-tuned MentalBERT from Hugging Face repo '{repo_id}'."
        except Exception as e:
            print(f"Hugging Face snapshot download failed for '{repo_id}': {e}")

    # Check direct URL for model.safetensors
    weights_url = get_artifact_url("text_model")
    if weights_url:
        safetensors_target = os.path.join(target_dir, "model.safetensors")
        success = download_file_from_url(weights_url, safetensors_target)
        if success and verify_text_model_export(target_dir):
            return True, "Successfully downloaded fine-tuned MentalBERT weights from URL."

    return False, (
        f"Fine-tuned MentalBERT weights not found at '{target_dir}'. "
        "For Streamlit Community Cloud: set 'HF_TEXT_MODEL_REPO' in Streamlit Secrets, "
        "or place 'model.safetensors' into 'models/text/mentalbert_best_export/'."
    )


def ensure_facial_model(target_path: str = DEFAULT_FER_MODEL_PATH) -> Tuple[bool, str]:
    """
    Ensure FER2013 facial CNN checkpoint is present and verified.
    """
    if verify_pt_checkpoint(target_path, min_bytes=1_000_000):
        return True, f"Facial emotion CNN checkpoint verified on disk at '{target_path}'."

    # Try downloading from Hugging Face repo
    repo_id = get_model_repo("artifacts")
    token = get_hf_token()
    if repo_id:
        if download_from_hf_hub(repo_id, "fer_model.pt", target_path, token=token):
            if verify_pt_checkpoint(target_path, min_bytes=1_000_000):
                return True, f"Downloaded and verified 'fer_model.pt' from Hugging Face repo '{repo_id}'."

    # Try downloading from direct URL
    url = get_artifact_url("fer_model")
    if url:
        if download_file_from_url(url, target_path):
            if verify_pt_checkpoint(target_path, min_bytes=1_000_000):
                return True, f"Downloaded and verified 'fer_model.pt' from configured URL '{url}'."

    return False, (
        f"Facial CNN checkpoint not found at '{target_path}'. "
        "Configure 'HF_ARTIFACTS_REPO' or 'FER_MODEL_URL' in Streamlit Secrets to download automatically."
    )


def ensure_fusion_model(target_path: str = DEFAULT_FUSION_MODEL_PATH) -> Tuple[bool, str]:
    """
    Ensure multimodal distress fusion model (Model C) checkpoint is present and verified.
    """
    if verify_pt_checkpoint(target_path, min_bytes=10_000):
        return True, f"Fusion model checkpoint verified on disk at '{target_path}'."

    # Try downloading from Hugging Face repo
    repo_id = get_model_repo("artifacts")
    token = get_hf_token()
    if repo_id:
        if download_from_hf_hub(repo_id, "fusion_model.pt", target_path, token=token):
            if verify_pt_checkpoint(target_path, min_bytes=10_000):
                return True, f"Downloaded and verified 'fusion_model.pt' from Hugging Face repo '{repo_id}'."

    # Try downloading from direct URL
    url = get_artifact_url("fusion_model")
    if url:
        if download_file_from_url(url, target_path):
            if verify_pt_checkpoint(target_path, min_bytes=10_000):
                return True, f"Downloaded and verified 'fusion_model.pt' from configured URL '{url}'."

    return False, (
        f"Fusion model checkpoint not found at '{target_path}'. "
        "Configure 'HF_ARTIFACTS_REPO' or 'FUSION_MODEL_URL' in Streamlit Secrets to download automatically."
    )


def ensure_behavioral_scaler(target_path: str = DEFAULT_SCALER_PATH) -> Tuple[bool, str]:
    """
    Ensure ESConv fitted 8-feature StandardScaler is present and verified.
    """
    if verify_scaler(target_path):
        return True, f"Behavioral scaler verified on disk at '{target_path}'."

    # Try downloading from Hugging Face repo
    repo_id = get_model_repo("artifacts")
    token = get_hf_token()
    if repo_id:
        if download_from_hf_hub(repo_id, "behav_scaler_esconv.pkl", target_path, token=token):
            if verify_scaler(target_path):
                return True, f"Downloaded and verified 'behav_scaler_esconv.pkl' from Hugging Face repo '{repo_id}'."

    # Try downloading from direct URL
    url = get_artifact_url("behav_scaler")
    if url:
        if download_file_from_url(url, target_path):
            if verify_scaler(target_path):
                return True, f"Downloaded and verified 'behav_scaler_esconv.pkl' from configured URL '{url}'."

    return False, (
        f"Behavioral scaler not found at '{target_path}'. "
        "Configure 'HF_ARTIFACTS_REPO' or 'BEHAV_SCALER_URL' in Streamlit Secrets to download automatically."
    )
