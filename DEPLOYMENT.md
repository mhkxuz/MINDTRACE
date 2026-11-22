# MINDTRACE — Streamlit Community Cloud Deployment Guide

**Project:** MINDTRACE: Multimodal Mental Health Risk Detection Research Platform  
**Target Environment:** Streamlit Community Cloud (Free Tier)  
**Main Entry Point:** `app/main.py`  
**Target Branch:** `master`  
**License & Scope:** Academic Research Demonstration  

---

## 1. Executive Summary & Architecture Overview

MINDTRACE is a multimodal mental health research platform demonstrating risk detection across conversational text, behavioral interaction dynamics, and facial affect analysis.

### Dual-Encoder Decoupled Architecture
The system maintains strict academic integrity through a decoupled dual-encoder design:
1. **Text Analysis (Page 2):** Fine-tuned 7-class MentalBERT sequence classifier (`models/text/mentalbert_best_export/`, Safetensors weights, 83.30% held-out test accuracy).
2. **Multimodal Distress Fusion (Page 5):** Frozen base MentalBERT encoder (`mental/mental-bert-base-uncased`, 768-d CLS token extraction) combined with 8 standardized behavioral dialogue metrics into a 776-dimensional joint representation for binary distress detection.
3. **Facial Affect Recognition (Page 3):** 3-Block PyTorch CNN (`models/facial/fer_model.pt`, 7 emotion probabilities).
4. **Behavioral Dynamics (Page 4):** Deterministic statistical feature extraction (13 metrics + 8 standardized features via `artifacts/scalers/behav_scaler_esconv.pkl`).

### Why Binary Model Weights Are Not in Git
GitHub enforces a 100 MB file limit, and best practices prohibit committing heavy binary model checkpoints (`.safetensors`, `.pt`, `.pkl`) directly into Git repositories. Accordingly, `.gitignore` excludes large model weights, while code, configurations, templates, and notebooks remain fully tracked.

For Streamlit Community Cloud, MINDTRACE implements a **seamless, on-demand artifact manager** (`src/utils/artifact_manager.py`) that downloads and verifies model artifacts from your designated Hugging Face repository or direct cloud storage upon first invocation, caching them within the container runtime.

---

## 2. Model Artifact Inventory & Hosting Options

The application utilizes four model artifacts:

| Artifact | Local Path | Approximate Size | Hosting Location |
|---|---|---|---|
| **Fine-Tuned MentalBERT Classifier** | `models/text/mentalbert_best_export/` (`model.safetensors`, `config.json`, etc.) | ~418 MB | Hugging Face Model Repo (`HF_MODEL_REPO` or `HF_TEXT_MODEL_REPO`) |
| **FER2013 Facial CNN Checkpoint** | `models/facial/fer_model.pt` | ~4.9 MB | Hugging Face Model Repo or Direct URL (`FER_MODEL_URL`) |
| **Multimodal Fusion Checkpoint** | `models/fusion/fusion_model.pt` | ~106 KB | Hugging Face Model Repo or Direct URL (`FUSION_MODEL_URL`) |
| **ESConv Behavioral Scaler** | `artifacts/scalers/behav_scaler_esconv.pkl` | ~775 Bytes | Hugging Face Model Repo or Direct URL (`BEHAV_SCALER_URL`) |
| **Base MentalBERT Model** | `mental/mental-bert-base-uncased` | ~440 MB | Official Hugging Face Hub (Gated access requires `HF_TOKEN`) |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Prepare Your Hugging Face Model Repository
To provide the model artifacts to Streamlit Community Cloud, host them on Hugging Face (free, fast, and secure).

1. Log in to [Hugging Face](https://huggingface.co/).
2. Click **New Model** (or go to `https://huggingface.co/new`).
3. Set the repository name (e.g. `mindtrace-models`) and visibility (Private or Public).
4. Upload the following files from your local workspace to the root of your Hugging Face model repository:
   - All files from `models/text/mentalbert_best_export/`:
     * `config.json`
     * `model.safetensors`
     * `tokenizer.json`
     * `tokenizer_config.json`
     * `training_metadata.json`
   - `models/facial/fer_model.pt`
   - `models/fusion/fusion_model.pt`
   - `artifacts/scalers/behav_scaler_esconv.pkl`

*(You can upload via the Hugging Face web UI or using the Python script below)*:

```python
# Helper script to upload all artifacts to your Hugging Face repository
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/mindtrace-models"

# Upload fine-tuned MentalBERT export folder
api.upload_folder(
    folder_path="models/text/mentalbert_best_export",
    repo_id=repo_id,
    repo_type="model",
)

# Upload individual model checkpoints and scaler
api.upload_file(path_or_fileobj="models/facial/fer_model.pt", path_in_repo="fer_model.pt", repo_id=repo_id)
api.upload_file(path_or_fileobj="models/fusion/fusion_model.pt", path_in_repo="fusion_model.pt", repo_id=repo_id)
api.upload_file(path_or_fileobj="artifacts/scalers/behav_scaler_esconv.pkl", path_in_repo="behav_scaler_esconv.pkl", repo_id=repo_id)
print("All artifacts uploaded successfully!")
```

---

### Step 2: Obtain Hugging Face Access Token
1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
2. Click **Create new token** with `Read` role.
3. Name it (e.g. `mindtrace-streamlit-cloud`).
4. Copy the token string (starts with `hf_...`).

---

### Step 3: Deploy on Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
2. Click **New app**.
3. Configure the deployment settings:
   - **Repository:** `mhkxuz/MINDTRACE` (or your repository URL)
   - **Branch:** `master`
   - **Main file path:** `app/main.py`
   - **App URL:** Customize if desired (e.g. `mindtrace-research.streamlit.app`)
4. **DO NOT** click Deploy yet; click **Advanced settings** (or expand **Secrets**).

---

### Step 4: Configure Streamlit Secrets
In the **Secrets** section of the Streamlit app setup, paste the following configuration:

```toml
# =========================================================================
# Streamlit Community Cloud Secrets for MINDTRACE
# =========================================================================

# 1. Hugging Face Access Token (Required for gated MentalBERT access)
HF_TOKEN = "hf_your_actual_token_here"

# 2. Your Hugging Face Repository containing the uploaded artifacts
HF_MODEL_REPO = "your-username/mindtrace-models"

# (Optional: separate repos if not using a unified repository)
# HF_TEXT_MODEL_REPO = "your-username/mindtrace-mentalbert"
# HF_ARTIFACTS_REPO = "your-username/mindtrace-artifacts"

# (Optional: direct URLs if hosting artifacts on GitHub Releases or cloud storage)
# FER_MODEL_URL = "https://..."
# FUSION_MODEL_URL = "https://..."
# BEHAV_SCALER_URL = "https://..."
```

5. Click **Save** and then click **Deploy!**

---

## 4. Verification & Testing Across All Six Pages

Once Streamlit Community Cloud builds and starts the application, perform the following verification:

1. **Landing Dashboard (`app/main.py`):**
   - Verify sidebar status pills: Text, Facial, Behavioral, and Fusion should display **Operational** or **Cloud Ready**.
   - Check the Model Inventory & Implementation Audit table.
2. **Page 1: Overview & Research Gap:**
   - Verify that all architecture cards, dataset inventory, and the 776-d vs 783-d comparison tables render cleanly.
3. **Page 2: Text Analysis (MentalBERT 7-Class):**
   - Select a sample statement (e.g. "Sample 1 (High Anxiety)" or "Sample 4 (Suicidal / Critical Alert)").
   - Click **Run MentalBERT Inference**.
   - On first run, the app will download `model.safetensors` from your repository, verify its checksum, and execute inference with uncertainty quantification (entropy and margin).
4. **Page 3: Facial Affect (FER2013 CNN):**
   - Upload a sample facial photo from `data/test_images/` (or any facial JPG/PNG).
   - Click **Classify Facial Emotion**.
   - Verify 7-class emotion probabilities and dominant affect output.
5. **Page 4: Behavioral Analysis:**
   - Input a conversational dialogue transcript.
   - Click **Extract Behavioral Metrics & Profile**.
   - Verify calculation of 13 interaction metrics, dialogue trajectory, and withdrawal risk indicators (pure statistical computation, zero neural net overhead).
6. **Page 5: Multimodal Distress Fusion (Model C):**
   - Provide a situation text and multi-turn dialogue.
   - Click **Execute Multimodal Fusion Pipeline**.
   - Verify that the frozen base MentalBERT encoder extracts the 768-d embedding, the behavioral feature extractor generates 8 standardized metrics, and Model C outputs the binary distress score (`Lower Distress` vs `Higher Distress`).
7. **Page 6: Research Evaluation Dashboard:**
   - Inspect documented classification reports, 5-seed benchmarks, and cross-validation paired t-test results.

---

## 5. Free-Tier Resource Optimization & Troubleshooting

Streamlit Community Cloud free tier provides approximately **1 GB of RAM** and shared CPU. MINDTRACE has been engineered specifically to stay within these boundaries.

### Memory Optimization Best Practices Implemented
- **Lazy Model Loading:** Models are loaded into memory only when their specific page and inference button are triggered. The landing dashboard and overview pages do not load any neural network weights into RAM.
- **Resource Caching (`@st.cache_resource`):** Ensures only a single instance of `MindTracePlatform` and loaded models exists across multiple page views and sessions, preventing memory duplication.
- **No-Grad Inference Mode:** All PyTorch forward passes run under `torch.no_grad()` with eval mode (`.eval()`), eliminating backward graph memory allocation.
- **Compact CNN & Fusion Footprint:** `FacialEmotionCNN` requires only ~5 MB of RAM; `FusionModelWeighted` requires ~100 KB; the behavioral feature extractor uses pure rule/regression mathematics.

### Troubleshooting Common Issues

#### 1. Cold Start Timeout on First Inference
- **Symptom:** The first time you click "Run MentalBERT Inference" or "Execute Multimodal Fusion", the spinner takes 30–60 seconds.
- **Cause:** Streamlit Cloud is streaming the ~418 MB `model.safetensors` or base MentalBERT weights from Hugging Face into runtime storage.
- **Resolution:** This only occurs once per container lifecycle. Subsequent inferences on that page execute in under 1 second.

#### 2. "Model Weights Missing from Disk" Warning
- **Symptom:** An informational banner appears stating weights were not found.
- **Cause:** `HF_MODEL_REPO` or `HF_TOKEN` was not configured in Streamlit Secrets, or the repository name has a typo.
- **Resolution:** Open **App Settings > Secrets** in Streamlit Cloud, verify `HF_MODEL_REPO` and `HF_TOKEN`, and save. The app will reload automatically.

#### 3. Out-of-Memory (OOM) Container Restarts
- **Symptom:** App crashes and displays "Resource Limit Exceeded" or restarts.
- **Cause:** Running heavy text classification and multimodal fusion repeatedly in rapid succession on large batches.
- **Resolution:** The application is tuned for single-prompt and conversational inference. If memory pressure occurs, simply reload the browser tab to reset the container memory cache.

---

## 6. Local Development Verification

Your local Arch Linux development workflow is completely preserved and unaffected by cloud configuration:
- Running `./run.sh` continues to load local checkpoints directly from `models/` without performing any network downloads.
- Run tests at any time with:
  ```bash
  pytest tests/ -v
  ```
  *(29/29 tests validate behavioral metrics, model architectures, dual-encoder offline isolation, and deployment secrets resolution).*
