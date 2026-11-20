# MINDTRACE — Facial Emotion Recognition (FER2013) CNN Retraining Report

## Executive Summary

This report documents the autonomous recovery, data pipeline construction, training, evaluation, and operational verification of the **Facial Emotion Recognition CNN** (`models/facial/fer_model.pt`) within the MINDTRACE multimodal mental health research platform.

---

## 1. Model Architecture & Parameters

### 1.1 Architecture Specification
- **Model Class:** `FacialEmotionCNN` (defined in `src/models/facial_model.py`, reproducing `notebooks/facial_emotion_label.ipynb`)
- **Structure:** Custom 3-block Convolutional Neural Network
  - **Block 1:** `Conv2d(3, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Output: $32 \times 24 \times 24$)
  - **Block 2:** `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Output: $64 \times 12\times 12$)
  - **Block 3:** `Conv2d(64, 128, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Output: $128 \times 6 \times 6$)
  - **Classifier:** `Flatten()` $\rightarrow$ `Linear(4608, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 7)`
- **Total Parameters:** **1,274,951** (exact match with notebook architecture)
- **Model Checkpoint Path:** `models/facial/fer_model.pt`
- **File Size:** **4.9 MB**

### 1.2 Target Emotion Classes
The model outputs raw logits across 7 emotion classes in the exact order defined in `notebooks/facial_emotion_label.ipynb`:

| Index | Emotion Class |
| :---: | :--- |
| **0** | `angry` |
| **1** | `disgust` |
| **2** | `fear` |
| **3** | `happy` |
| **4** | `neutral` |
| **5** | `sad` |
| **6** | `surprise` |

---

## 2. Training Dataset & Image Preprocessing

### 2.1 Dataset Provenance
- **Dataset:** FER2013 (Facial Expression Recognition 2013 Challenge)
- **Source Mirror:** `AutumnQiu/fer2013` (Hugging Face Parquet format)
- **Training Samples:** 28,709 images
- **Test Samples:** 3,589 images
- **Validation Samples:** 3,589 images
- **Resolution:** $48 \times 48$ pixels, 3 channels (RGB converted)

### 2.2 Balanced Class Weighting
To counter FER2013 class imbalances (e.g. disgust having only 436 training samples vs happy having 7,215), balanced inverse frequency weights were computed using `sklearn.utils.class_weight.compute_class_weight`:

| Class Index | Emotion | Balanced Class Weight |
| :---: | :--- | :---: |
| **0** | `angry` | 1.0266 |
| **1** | `disgust` | 9.4066 |
| **2** | `fear` | 1.0010 |
| **3** | `happy` | 0.5684 |
| **4** | `neutral` | 0.8260 |
| **5** | `sad` | 0.8491 |
| **6** | `surprise` | 1.2934 |

### 2.3 Image Transformations
- **Training Pipeline:**
  - Resize to $(48, 48)$
  - Random Horizontal Flip (`p=0.5`)
  - ToTensor (pixel values normalized to $[0, 1]$)
  - Normalize with mean $[0.5, 0.5, 0.5]$ and std $[0.5, 0.5, 0.5]$
- **Inference / Test Pipeline:**
  - Resize to $(48, 48)$
  - ToTensor
  - Normalize with mean $[0.5, 0.5, 0.5]$ and std $[0.5, 0.5, 0.5]$

---

## 3. Training Execution & Progression

- **Optimizer:** `torch.optim.Adam` ($lr=10^{-3}$)
- **Loss Criterion:** `nn.CrossEntropyLoss(weight=class_weights)`
- **Batch Size:** 64
- **Epochs:** 10
- **Total Training Batches:** 449 per epoch ($4,490$ total)
- **Hardware Device:** CPU (16 parallel threads)
- **Total Elapsed Training Time:** **7.88 minutes** (~47 seconds per epoch)

### 3.1 Epoch-by-Epoch Progress

| Epoch | Avg Training Loss | Training Accuracy | Epoch Duration |
| :---: | :---: | :---: | :---: |
| **1 / 10** | 1.8044 | 27.08% | 0.73 min |
| **2 / 10** | 1.5822 | 39.68% | 0.76 min |
| **3 / 10** | 1.4358 | 45.22% | 0.77 min |
| **4 / 10** | 1.3498 | 48.25% | 1.18 min |
| **5 / 10** | 1.2990 | 50.13% | 0.80 min |
| **6 / 10** | 1.2264 | 52.73% | 0.79 min |
| **7 / 10** | 1.1800 | 53.77% | 0.80 min |
| **8 / 10** | 1.1459 | 54.69% | 0.78 min |
| **9 / 10** | 1.1040 | 56.49% | 0.77 min |
| **10 / 10** | **1.0641** | **57.41%** | 0.79 min |

---

## 4. Final Test Set Evaluation Results

Evaluation performed on the full hold-out test set (3,589 images):

| Metric | Retrained Model (Current) | Historical Notebook Benchmark | Variance |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **56.03%** | **58.79%** | $-2.76\%$ |
| **Macro F1** | **52.76%** | **56.47%** | $-3.71\%$ |
| **Weighted F1** | **55.95%** | **59.00%** | $-3.05\%$ |

### 4.1 Per-Class Performance Breakdown

| Emotion Class | Precision | Recall | F1-Score | Support | Historical Notebook F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `angry` | 0.44 | 0.52 | **0.47** | 491 | 0.51 |
| `disgust` | 0.30 | 0.73 | **0.42** | 55 | 0.51 |
| `fear` | 0.42 | 0.30 | **0.35** | 528 | 0.43 |
| `happy` | 0.79 | 0.77 | **0.78** | 879 | 0.80 |
| `neutral` | 0.55 | 0.55 | **0.55** | 626 | 0.55 |
| `sad` | 0.41 | 0.42 | **0.41** | 594 | 0.41 |
| `surprise` | 0.72 | 0.69 | **0.70** | 416 | 0.74 |

---

### 4.3 Reconstructed Original 7,178-Image Test Evaluation

As identified in the research audit, the original `notebooks/facial_emotion_label.ipynb` evaluated against Kaggle's merged `test` folder of **7,178 images** (combining PublicTest: 3,589 images and PrivateTest: 3,589 images). In `AutumnQiu/fer2013`, these are stored as `valid` (3,589) and `test` (3,589).

Evaluating `models/facial/fer_model.pt` across the reconstructed combined test partition (7,178 images) yields:

| Metric | Retrained Model (Current) | Historical Notebook Benchmark | Variance |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **55.42%** | **58.79%** | $-3.37\%$ |
| **Macro F1** | **52.27%** | **56.47%** | $-4.20\%$ |
| **Weighted F1** | **55.51%** | **59.00%** | $-3.49\%$ |

#### Exact Per-Class Breakdown on 7,178 Images:
| Emotion Class | Precision | Recall | F1-Score | Support (Actual) | Original Notebook Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `angry` | 0.43 | 0.51 | **0.46** | 958 | 958 |
| `disgust` | 0.29 | 0.68 | **0.41** | 111 | 111 |
| `fear` | 0.41 | 0.31 | **0.35** | 1,024 | 1,024 |
| `happy` | 0.79 | 0.75 | **0.77** | 1,774 | 1,774 |
| `neutral` | 0.52 | 0.52 | **0.52** | 1,233 | 1,233 |
| `sad` | 0.43 | 0.43 | **0.43** | 1,247 | 1,247 |
| `surprise` | 0.72 | 0.71 | **0.71** | 831 | 831 |
| *Macro Average* | 0.51 | 0.56 | **0.52** | 7,178 | 7,178 |
| *Weighted Average* | 0.56 | 0.55 | **0.56** | 7,178 | 7,178 |
| **Accuracy** | | | **0.5603** | 3,589 | 0.5879 |
| **Macro Avg** | 0.52 | 0.57 | **0.5276** | 3,589 | 0.5647 |
| **Weighted Avg**| 0.57 | 0.56 | **0.5595** | 3,589 | 0.5900 |

---

## 5. Verification & Live System Integration

1. **Weights Saved:** `models/facial/fer_model.pt` (4.9 MB).
2. **Architecture Verified:** `FacialEmotionService` loads model weights cleanly with zero missing or unexpected keys.
3. **Inference Confirmed:** Tested with synthetic and uploaded face images; probabilities sum to $1.0000 \pm 10^{-7}$.
4. **App Integration:** Streamlit Page 3 (`app/pages/3_Facial_Emotion.py`) automatically reflects **Operational** status and live classification buttons are enabled.
