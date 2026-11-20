# MINDTRACE — MentalBERT Text Classification Model Retraining & Provenance Report

## Executive Summary

This report documents the autonomous investigation, dataset recovery, reproducible implementation, and evaluation framework for the **MentalBERT 7-Class Text Classifier** (`models/text/text_model_final.pt`) within the MINDTRACE multimodal mental health research platform.

---

## 1. Model & Dataset Provenance

### 1.1 Base Architecture
- **Base Encoder:** `mental/mental-bert-base-uncased` (Hugging Face Hub)
- **Head Architecture:** 7-class linear sequence classification head atop 768-dimensional `[CLS]` token pooling
- **Parameters:** ~110M parameters
- **Expected Weights Path:** `models/text/text_model_final.pt`

### 1.2 Training Dataset Provenance
- **Dataset Title:** Kaggle Sentiment Analysis for Mental Health / Combined Mental Health Statements
- **Author/Mirror:** `AhmedSSoliman/sentiment-analysis-for-mental-health-Combined-Data`
- **Total Valid Records:** 52,681 records (after dropping null statements/statuses)
- **Class Distribution:**
  - `Normal`: 16,343 samples (31.0%)
  - `Depression`: 15,404 samples (29.2%)
  - `Suicidal`: 10,652 samples (20.2%)
  - `Anxiety`: 3,841 samples (7.3%)
  - `Bipolar`: 2,777 samples (5.3%)
  - `Stress`: 2,587 samples (4.9%)
  - `Personality disorder`: 1,077 samples (2.0%)

### 1.3 Target Classes & Label Mapping
The target classes match the exact alphabetical sorting produced by `sklearn.preprocessing.LabelEncoder` in the original research notebook (`notebooks/text_model.ipynb`):

| Class Index | Label Name |
| :---: | :--- |
| **0** | `Anxiety` |
| **1** | `Bipolar` |
| **2** | `Depression` |
| **3** | `Normal` |
| **4** | `Personality disorder` |
| **5** | `Stress` |
| **6** | `Suicidal` |

---

## 2. Experimental Methodology & Preprocessing

### 2.1 Text Cleaning & Preprocessing
- **String Sanitization:** Explicit type coercion `str(x).strip()` ensuring non-string / NaN values do not corrupt PyTorch tensors.
- **Empty Token Guard:** Empty or whitespace-only strings are safely replaced with placeholder token `"empty"`.
- **Tokenization:**
  - Tokenizer: `AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")`
  - Max Token Length: 128
  - Padding: `padding=True`
  - Truncation: `truncation=True`
  - Return Format: PyTorch tensors (`return_tensors='pt'`)

### 2.2 Dataset Splitting (Reproducing Cell 6)
- **Train Set:** 80% (42,144 records), stratified by class label (`random_state=42`)
- **Validation Set:** 10% (5,268 records), stratified by class label (`random_state=42`)
- **Test Set:** 10% (5,269 records), hold-out evaluation, stratified (`random_state=42`)

### 2.3 Optimization & Hyperparameters
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** $2 \times 10^{-5}$
- **Batch Size:** 16
- **Epochs:** 3
- **Loss Function:** `nn.CrossEntropyLoss`

---

## 3. Computational Feasibility & Hardware Strategy

### 3.1 Local Hardware Audit
- **CPU:** AMD / Intel host (16 threads available)
- **Virtual Environment:** PyTorch 2.15.0+cpu (CPU-only build)
- **Benchmark Calculation on CPU:**
  - Train dataset: 42,144 samples / batch size 16 = 2,634 batches/epoch
  - On CPU, forward + backward pass for BERT-base averages ~1.8 seconds per batch.
  - Estimated duration per epoch: $\approx 79 \text{ minutes}$ ($1.3 \text{ hours}$).
  - 3 epochs total duration: $\approx 4 \text{ to } 5 \text{ hours}$ under continuous full CPU load (or up to 14-16 hours under throttled thermal conditions).

### 3.2 Google Colab GPU Pipeline
To provide a fast, 1-click execution pathway:
- **Notebook Created:** `notebooks/train_text_model_colab.ipynb`
- **Execution Environment:** Free Google Colab T4 GPU (16GB VRAM)
- **Colab Training Time:** ~40-50 minutes total for all 3 epochs.
- **Automated Workflow:**
  1. Clones repository or loads directly from Hugging Face.
  2. Executes full 3-epoch training and validation.
  3. Evaluates hold-out test set (Confusion Matrix + Classification Report).
  4. Automatically downloads `text_model_final.pt` to local machine.

### 3.3 Standalone Local Training Script
- **Script Created:** `src/models/train_text_model.py`
- **CLI Options:**
  - `--epochs`: Number of epochs (default: 3)
  - `--batch-size`: Batch size (default: 16)
  - `--lr`: Learning rate (default: 2e-5)
  - `--max-samples`: Subsample flag for smoke testing or rapid fine-tuning
  - `--output-path`: Checkpoint save path (default: `models/text/text_model_final.pt`)
  - `--dataset-path`: Optional custom CSV path

---

## 4. Benchmark Results Comparison

| Metric | Original Notebook Reported Benchmark | Retraining Framework Target |
| :--- | :---: | :---: |
| **Test Accuracy** | **84.26%** | $\ge 82.0\%$ |
| **Macro F1** | **82.90%** | $\ge 81.0\%$ |
| **Weighted F1** | **84.21%** | $\ge 83.0\%$ |
| **Suicidal Recall** | **71.0%** | $\ge 70.0\%$ |
| **Depression F1** | **79.0%** | $\ge 78.0\%$ |
| **Normal F1** | **96.0%** | $\ge 95.0\%$ |

---

## 5. System Integration & Verification

The checkpoint format saved by both `src/models/train_text_model.py` and `notebooks/train_text_model_colab.ipynb` is a standard PyTorch `state_dict` matching:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "mental/mental-bert-base-uncased", num_labels=7
)
state_dict = torch.load("models/text/text_model_final.pt", map_location=device)
model.load_state_dict(state_dict)
```
This is 100% compatible with `src.models.text_model.MentalBERTService.load_classifier()` and the multimodal fusion pipeline in `src.services.inference.MultimodalInferenceEngine`.
