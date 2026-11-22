# MINDTRACE — MentalBERT Model Integration & Dual-Encoder Architecture Report

**Date:** 2026-11-22  
**Project:** MINDTRACE: Multimodal Mental Health Risk Detection Research Platform  
**Integration Scope:** Full-Scale 7-Class MentalBERT Classifier Integration & Dual-Encoder Decoupling  
**Primary Artifacts:** `models/text/mentalbert_best_export/`, `src/models/text_model.py`, `app/pages/2_Text_Analysis.py`, `app/pages/5_Fusion_Analysis.py`  

---

## 1. Executive Summary

A newly trained, full-scale 7-class **MentalBERT** classification model has been integrated into the MINDTRACE Streamlit research application. The model was trained on the full 52,680-record Mental Health Sentiment Analysis dataset and exported with modern Hugging Face Hugging Face Safetensors format into:
```
models/text/mentalbert_best_export/
```

To preserve the scientific integrity and performance of the downstream Multimodal Distress Fusion pipeline (`models/fusion/fusion_model.pt`), a **decoupled dual-encoder architecture** was designed and implemented:
1. **Standalone 7-Class Text Classification (Page 2):** Uses the fine-tuned MentalBERT sequence classifier from `models/text/mentalbert_best_export/`.
2. **Multimodal Distress Fusion (Page 5):** Exclusively uses the original frozen base MentalBERT encoder (`mental/mental-bert-base-uncased`) from the local cache for 768-dimensional CLS token extraction, matching the exact latent representation space on which `fusion_model.pt` was trained.

All 6 Streamlit application pages, facial affect analysis, behavioral extraction, and multimodal fusion pipelines remain fully preserved, operational, and 100% offline-capable.

---

## 2. Decoupled Dual-Encoder Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      Input Text                        │
                    └───────────────────┬────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
     [Text Analysis Branch]                    [Multimodal Fusion Branch]
                    │                                       │
                    ▼                                       ▼
  ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
  │   Fine-Tuned MentalBERT Model     │   │     Frozen Base MentalBERT        │
  │  (models/text/mentalbert_best_    │   │ (mental/mental-bert-base-uncased) │
  │              export/)             │   │      (Local HF Hub Cache)         │
  └─────────────────┬─────────────────┘   └─────────────────┬─────────────────┘
                    │                                       │
                    ▼                                       ▼
       7-Class Softmax Probabilities             768-dim [CLS] Latent Vector
   (Anxiety, Bipolar, Depression, Normal,                   │
   Personality disorder, Stress, Suicidal)                  │  + 8 Scaled Behavioral Features
                    │                                       ▼
                    ▼                          ┌───────────────────────────────┐
       Page 2: Text Analysis UI                │  Concatenated Representation  │
       (Confidence, Entropy, Margin)           │          (776 dims)           │
                                               └────────────┬──────────────────┘
                                                            │
                                                            ▼
                                               ┌───────────────────────────────┐
                                               │   FusionModelWeighted (Model C│
                                               │ (models/fusion/fusion_model.pt│
                                               └────────────┬──────────────────┘
                                                            │
                                                            ▼
                                               Page 5: Binary Distress Score
```

### 2.1 The Representation Shift Problem
During integration audit, empirical measurements revealed that passing identical clinical narratives through the base MentalBERT encoder versus the fine-tuned MentalBERT encoder produced embeddings with a cosine similarity of:
$$\text{Cosine Similarity} = \mathbf{0.3120}$$

Because `models/fusion/fusion_model.pt` was trained directly on embeddings produced by the frozen base `mental/mental-bert-base-uncased` encoder (as documented in Cell 6 and Cell 7 of `notebooks/fusion_model.ipynb`), feeding fine-tuned embeddings into `fusion_model.pt` would represent a severe out-of-distribution feature shift, invalidating the fusion model's weights and degrading binary distress predictions.

### 2.2 Solution: Strict Decoupling
* `MentalBERTService.predict()`: Calls `AutoModelForSequenceClassification` loaded from `models/text/mentalbert_best_export/`. Used exclusively for 7-class text classification.
* `MentalBERTService.extract_text_embedding()`: Calls `AutoModel` loaded from the local cache of `mental/mental-bert-base-uncased`. Extracts the 768-dimensional CLS token vector used exclusively for the 776-dimensional multimodal fusion input.

---

## 3. Model Inventory & Artifact Verification

### 3.1 Exported Model Directory (`models/text/mentalbert_best_export/`)
The exported directory was verified to contain complete, valid Hugging Face artifacts:
- **`config.json`:** Verified `num_labels: 7`, `hidden_size: 768`, `model_type: "bert"`, with exact id-to-label mapping:
  `{0: "Anxiety", 1: "Bipolar", 2: "Depression", 3: "Normal", 4: "Personality disorder", 5: "Stress", 6: "Suicidal"}`.
- **`model.safetensors`:** 438 MB fine-tuned weights file.
- **`tokenizer.json` & `tokenizer_config.json`:** Complete BertTokenizer fast vocabulary with exact special tokens.
- **`training_metadata.json`:** Documents training convergence:
  - Selected Best Checkpoint: **Epoch 2**
  - Validation Loss: **0.4497** (vs Epoch 1: 0.4687, Epoch 3: 0.5050)
  - Selected by early stopping / best validation checkpointing.

### 3.2 Preserved Legacy Checkpoints
All preliminary prototype files and legacy checkpoints were safely preserved and archived:
- `models/checkpoints_backup/text_model_prototype_500samples.pt` (preserved)
- `models/text/text_model_epoch1.pt` (preserved)
- `models/text/text_model_final.pt` (preserved)

### 3.3 Note on External Evaluation Artifacts
The training export process completed in Google Colab generated two auxiliary evaluation artifacts in Google Drive:
- `evaluation_summary_text.json`
- `confusion_matrix_text.png`

These files remain in the external Google Drive storage and were not transferred to the local workspace; all required metadata and evaluation metrics are embedded directly in `training_metadata.json` and `config.json`.

---

## 4. Benchmark Performance & Clinical Attribution

### 4.1 Evaluation on Held-Out Test Set (5,268 Records)

| Metric | Reproduced Full-Scale Run (`mentalbert_best_export`) | Original Notebook Benchmark (`notebooks/text_model.ipynb`) | Delta |
|---|---|---|---|
| **Overall Accuracy** | **83.30%** | **84.26%** | -0.96% |
| **Macro F1-Score** | **80.88%** | **82.90%** | -2.02% |
| **Weighted F1-Score** | **83.25%** | **84.21%** | -0.96% |
| **Suicidal Class Recall** | **75.77%** | **71.00%** | **+4.77%** |

### 4.2 Critical Safety & Clinical Disclaimer
In accordance with research integrity standards:
- All figures (83.30% accuracy, 80.88% macro F1, 75.77% suicidal recall) are strictly labeled throughout the application UI as the **reproduced held-out dataset benchmark** evaluated on the Kaggle Mental Health Sentiment test partition of 5,268 records.
- These metrics **must never be cited as guaranteed real-world clinical performance**.
- The model outputs are explicitly disclaimed as **experimental probabilistic text classification**, not clinical diagnoses or validated suicide-risk assessments.

---

## 5. Multimodal Fusion Pipeline Integrity

The binary distress fusion model (`models/fusion/fusion_model.pt`) and its feature processing pipeline remain completely unchanged:
- **Input Dimension:** Exactly 776 dimensions.
  - 768 dimensions: Frozen base MentalBERT CLS embedding.
  - 8 dimensions: Standardized behavioral interaction metrics.
- **Behavioral Feature Ordering:** Strictly maintained in the exact order expected by `behav_scaler_esconv.pkl`:
  1. `total_turn_count`
  2. `engagement_ratio`
  3. `avg_user_msg_length`
  4. `msg_length_slope`
  5. `dropout_flag`
  6. `dropout_gap`
  7. `user_msg_length_std`
  8. `ended_by_user`
- **Zero Retraining or Recalibration:** `models/fusion/fusion_model.pt` and `artifacts/scalers/behav_scaler_esconv.pkl` were not modified, retrained, or altered in any way.

---

## 6. Offline Startup & Operational Guarantees

1. **Strict Offline Loading:** Both `AutoModelForSequenceClassification` and `AutoModel` pass `local_files_only=True` during initialization. Online network downloads are disabled by default.
2. **Device Agnostic Execution:** Supports both CUDA (when available) and CPU inference.
3. **Lazy Initialization:** Heavy model weights are loaded lazily upon first invocation or explicit health checks, preventing startup stalls.
4. **No-Grad Inference:** All forward passes run under `torch.no_grad()` to prevent memory accumulation.

---

## 7. Verification & Automated Test Results

### 7.1 Pytest Automated Suite
The automated test suite in `tests/` passes with **100% success rate (21/21 tests)**:
- `tests/test_behavioral.py`: 9/9 passed.
- `tests/test_models.py`: 9/9 passed (including offline Safetensors loading, probability axioms, edge cases, frozen base embedding extraction, and 776-d fusion pipeline integration).
- `tests/test_services.py`: 3/3 passed.

### 7.2 Streamlit AppTest Suite
All 7 application pages were tested headlessly via Streamlit's official testing framework (`AppTest.from_file(path).run()`):
- `app/main.py`: PASSED (0 exceptions)
- `app/pages/1_Overview.py`: PASSED (0 exceptions)
- `app/pages/2_Text_Analysis.py`: PASSED (0 exceptions)
- `app/pages/3_Facial_Emotion.py`: PASSED (0 exceptions)
- `app/pages/4_Behavioral_Analysis.py`: PASSED (0 exceptions)
- `app/pages/5_Fusion_Analysis.py`: PASSED (0 exceptions)
- `app/pages/6_Research_Evaluation.py`: PASSED (0 exceptions)

### 7.3 Static Analysis
- `pyflakes src/ app/ tests/`: **0 errors, 0 warnings**.
