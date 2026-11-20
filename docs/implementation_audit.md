# Implementation Audit: Multimodal Mental Health Risk Detection

## 1. Executive Summary

This document provides a rigorous, code-level audit comparing the research vision articulated in the project report (`Project_Mentor Report(updated) (1).pdf`) against the concrete implementations found in the four Jupyter notebooks in `/notebooks`.

The research objective is to investigate whether combining conversational text, behavioral interaction patterns, and facial emotion signals can improve mental health distress detection compared to relying solely on conversational text.

---

## 2. Comprehensive Notebook-by-Notebook Audit

### A. Notebook 1: `data_preparation+behavioral_features.ipynb`
- **Purpose**: Ingestion of the ESConv (Emotional Support Conversation) dataset from Hugging Face (`thu-coai/esconv`), conversational text cleaning, and behavioral interaction feature extraction.
- **Dataset**: ESConv dialogue records parsed from JSON. Each record contains a seeker situation, initial/final emotion intensity (scale 1–5), empathy scores, and turn-by-turn dialogue between seeker (`user`) and supporter (`assistant`).
- **Feature Extraction**:
  - Computed 13 raw behavioral metrics per conversation:
    1. `user_turn_count`: Total turns spoken by user/seeker.
    2. `assistant_turn_count`: Total turns spoken by assistant/supporter.
    3. `total_turn_count`: Sum of all turns in dialogue.
    4. `engagement_ratio`: $\frac{\text{user\_turn\_count}}{\text{total\_turn\_count}}$.
    5. `avg_user_msg_length`: Mean word count of user messages.
    6. `avg_assistant_msg_length`: Mean word count of assistant messages.
    7. `user_msg_length_std`: Standard deviation of user message word counts.
    8. `last_user_turn`: Maximum turn index corresponding to user.
    9. `last_speaker`: Speaker of the final message ("user" vs "assistant").
    10. `ended_by_user`: Binary flag (1 if `last_speaker == 'user'` else 0).
    11. `msg_length_slope`: Linear regression slope of user message word length across turn sequence (using `scipy.stats.linregress`).
    12. `dropout_gap`: $\text{total\_turn\_count} - \text{last\_user\_turn}$.
    13. `dropout_flag`: Binary indicator (1 if $\text{dropout\_gap} > 2$ else 0).
  - Merged with seeker survey metadata (situation, `initial_emotion_intensity`, `emotion_type`, `problem_type`).
  - Saved outputs: `behavioral_features_final.csv` (910 rows, 21 columns).
- **Audit Findings**:
  - Implementation is fully sound, reproducible, and mathematically well-defined.
  - The feature engineering does not track real-time typing speed or keystroke dynamics, as ESConv records are static turn-based text without millisecond timestamps.

---

### B. Notebook 2: `text_model.ipynb`
- **Purpose**: Fine-tuning MentalBERT (`mental/mental-bert-base-uncased`) for 7-class mental health text classification.
- **Dataset**: Kaggle Sentiment Analysis for Mental Health (52,680 records across 7 categories: `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality disorder`, `Stress`, `Suicidal`).
- **Architecture & Training**:
  - Base Model: `AutoModelForSequenceClassification.from_pretrained("mental/mental-bert-base-uncased", num_labels=7)`.
  - Tokenizer: MentalBERT uncased tokenizer.
  - Optimizer: AdamW ($lr=2\times 10^{-5}$).
  - Epochs: 3 epochs.
- **Reported Benchmark Performance**:
  - **Accuracy**: $84.26\%$
  - **Macro F1**: $82.90\%$
  - **Weighted F1**: $84.21\%$
  - Per-class F1-scores: Normal ($0.96$), Anxiety ($0.88$), Bipolar ($0.87$), Stress ($0.79$), Depression ($0.79$), Personality disorder ($0.78$), Suicidal ($0.73$, with $71\%$ recall).
- **Saved Checkpoints**: Notebook saves `text_model_epoch1.pt`, `text_model_epoch2.pt`, `text_model_epoch3.pt`, and `text_model_final.pt`.
- **Audit Findings**:
  - The model was trained in Google Colab on an NVIDIA Tesla T4 GPU.
  - The high accuracy ($84.26\%$) belongs strictly to this 7-class text classification task on Kaggle data, not the multimodal distress fusion task.
  - The repository in Git did not include the large binary weights (`text_model_final.pt` ~438MB).

---

### C. Notebook 3: `facial_emotion_label.ipynb`
- **Purpose**: Custom Convolutional Neural Network (CNN) trained on the FER2013 facial expression dataset.
- **Dataset**: FER2013 (28,709 train images, 7,178 test images, $48\times 48$ grayscale converted to 3-channel RGB).
  - 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- **Architecture**:
  - 3 Convolutional Blocks:
    1. $\text{Conv2D}(3 \to 32, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($48\times 48 \to 24\times 24$)
    2. $\text{Conv2D}(32 \to 64, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($24\times 24 \to 12\times 12$)
    3. $\text{Conv2D}(64 \to 128, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($12\times 12 \to 6\times 6$)
  - Classifier:
    - $\text{Flatten} \to \text{Linear}(128\times 6\times 6 = 4608 \to 256) \to \text{ReLU} \to \text{Dropout}(0.5) \to \text{Linear}(256 \to 7)$
  - Total Parameters: 1,274,951.
- **Preprocessing & Transforms**:
  - Training: $\text{Resize}((48, 48)) \to \text{RandomHorizontalFlip}() \to \text{ToTensor}() \to \text{Normalize}([0.5], [0.5])$.
  - Inference: $\text{Resize}((48, 48)) \to \text{ToTensor}() \to \text{Normalize}([0.5], [0.5])$.
- **Reported Test Results**:
  - **Accuracy**: $58.79\%$
  - **Macro F1**: $56.47\%$
  - Weighted F1: $59\%$
- **Saved Checkpoint**: Notebook executed `torch.save(model_fer.state_dict(), "fer_model.pt")`.
- **Audit Findings**:
  - The CNN operates strictly on static facial expressions. Facial affect classification is not clinical diagnosis.
  - The model weights `fer_model.pt` were saved inside Google Colab ephemeral storage and not retained in the git tree.

---

### D. Notebook 4: `fusion_model.ipynb`
- **Purpose**: Comparative evaluation of Text-only (Model A), Behavioral-only (Model B), and Text+Behavioral Fusion (Model C) on binary distress detection.
- **Dataset & Labels**:
  - Conversations from ESConv (908 samples).
  - Distress label defined as: $\text{distress\_label} = \mathbb{I}(\text{initial\_emotion\_intensity} \ge 4)$.
  - Class distribution: Class 1 (High distress): 702 ($77.3\%$), Class 0 (Low distress): 206 ($22.7\%$).
- **Features Used**:
  - Text: 768-dimensional embeddings generated from the `situation` text using a frozen MentalBERT encoder (`mental/mental-bert-base-uncased`), saved as `esconv_text_embeddings.npy`.
  - Behavioral: Exactly 8 features selected and scaled via `StandardScaler`:
    1. `total_turn_count`
    2. `engagement_ratio`
    3. `avg_user_msg_length`
    4. `msg_length_slope`
    5. `dropout_flag`
    6. `dropout_gap`
    7. `user_msg_length_std`
    8. `ended_by_user`
  - Total combined input dimension: $768 + 8 = 776$ dimensions.
- **Architectures**:
  - **Model A (Text Only)**:
    - $\text{Linear}(768 \to 128) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128 \to 2)$ (98,690 parameters).
  - **Model B (Behavioral Only)**:
    - $\text{Linear}(8 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 16) \to \text{ReLU} \to \text{Linear}(16 \to 2)$ (850 parameters).
  - **Model C (Fusion - `FusionModelWeighted`)**:
    - Text branch: $\text{Linear}(768 \to 32) \to \text{ReLU} \to \text{Dropout}(0.4)$
    - Behavioral branch: $\text{Linear}(8 \to 16) \to \text{ReLU} \to \text{Dropout}(0.2)$
    - Classifier: Concatenation $(32 + 32 = 64) \to \text{Linear}(64 \to 16) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(16 \to 2)$ (25,970 parameters).
- **Reported Experimental Results**:
  - **5-Seed Holdout Evaluation**:
    | Model | Accuracy (mean ± std) | Balanced Acc (mean ± std) | F1 (mean ± std) | AUROC (mean ± std) |
    |---|---|---|---|---|
    | **A - Text** | $0.554 \pm 0.117$ | $0.527 \pm 0.033$ | $0.649 \pm 0.155$ | $0.496 \pm 0.023$ |
    | **B - Behavioral** | $0.448 \pm 0.030$ | $0.385 \pm 0.018$ | $0.583 \pm 0.035$ | $0.370 \pm 0.022$ |
    | **C - Fusion** | $0.576 \pm 0.035$ | $0.514 \pm 0.035$ | $0.694 \pm 0.034$ | $0.477 \pm 0.025$ |
  - **5-Fold Cross-Validation**:
    | Model | Balanced Acc (mean ± std) | F1 (mean ± std) | AUROC (mean ± std) |
    |---|---|---|---|
    | **A - Text** | $0.556 \pm 0.039$ | $0.655 \pm 0.153$ | $0.573 \pm 0.046$ |
    | **B - Behavioral** | $0.525 \pm 0.009$ | $0.527 \pm 0.280$ | $0.506 \pm 0.073$ |
    | **C - Fusion** | $0.549 \pm 0.043$ | $0.710 \pm 0.052$ | $0.559 \pm 0.052$ |
  - **Statistical Significance**: Paired t-test between Model C and Model A AUROC yielded a mean difference of $-0.0138$ with $p = 0.2957$.
  - **Conclusion**: The experimental data does **not** demonstrate a statistically significant performance advantage for fusion over text alone on this dataset.
- **Audit Findings**:
  - Cell 23 showed that `fusion_model.pt` was not saved originally.
  - **Resolution in MINDTRACE**: The training pipeline was reproduced in `src/models/train_fusion.py` using ESConv, generating MentalBERT embeddings, fitting the StandardScaler, and saving the trained weights to `models/fusion/fusion_model.pt` and `artifacts/scalers/behav_scaler_esconv.pkl`. Live fusion inference is verified and operational.
  - See `docs/model_recovery_report.md` for complete recovery audit details.

---

## 3. Discrepancy Analysis: Research Report vs. Actual Implementation

| Dimension | Research Report Proposal | Notebook Implementation | Application Treatment |
|---|---|---|---|
| **Input Modalities** | 3 Modalities: Text (768), Behavioral (8), Facial Affect (7). | 2 Modalities: Text (768) + Behavioral (8). | Clearly present 2-branch fusion as the implemented research experiment; classify 3-branch as Proposed Architecture. |
| **Combined Dimension** | 783 dimensions ($768 + 8 + 7$). | 776 dimensions ($768 + 8$). | Enforce exact 776-dimensional input vector for inference. No fabricated 783-d mockups. |
| **Facial Affect Role** | Proposed to feed 7 emotion probabilities into multimodal fusion. | Implemented only as a standalone vision classifier on FER2013; no paired facial data exists in ESConv. | Standalone FER2013 vision module; disclaimers that affect $\ne$ mental state. |
| **Behavioral Features** | Proposed real-time typing dynamics (latency, typing speed). | Static turn metrics (turn counts, engagement ratio, message slope, dropout gap). | Accurately compute the 13 defined turn-level metrics from text. |
| **Text Classification** | 7 mental health classes ($84.26\%$ accuracy). | 7 classes on Kaggle; frozen MentalBERT 768-d embeddings for ESConv. | Dedicated 7-class text analysis page; clear distinction from binary distress fusion. |
| **Model Weights** | Assumed checkpoints were stored on disk. | Checkpoints (`text_model_final.pt`, `fer_model.pt`, `fusion_model.pt`) were in Colab memory. | Transparent artifact management: load trained checkpoints when present, provide clear instructions/reproducibility tools. |

---

## 4. Architectural Plan for MINDTRACE Research Platform

1. **Modular Architecture**:
   - `src/features/behavioral.py`: Pure, robust feature extractor implementing the exact equations from Notebook 1.
   - `src/models/facial_model.py`: PyTorch `FacialEmotionCNN` matching Notebook 3.
   - `src/models/text_model.py`: Hugging Face MentalBERT classifier and embedding extractor matching Notebook 2.
   - `src/models/fusion_model.py`: PyTorch `FusionModelWeighted` (776 inputs $\to$ 2 binary outputs) matching Notebook 4.
   - `src/services/inference.py`: Unified inference pipeline with caching, safe error handling, and honesty assertions.
2. **Honesty & Integrity**:
   - Never fabricate predictions or use dummy random weights.
   - If an artifact is not downloaded or verified, display an actionable "Artifact Missing / Setup Needed" notice.
3. **User Interface (MINDTRACE)**:
   - Deep charcoal dark theme, modern typography, responsive cards, interactive Plotly visualizations.
