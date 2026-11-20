# MINDTRACE: Technical Architecture & System Documentation

## 1. System Overview

**MINDTRACE** is an academic research platform designed to investigate multimodal mental health risk detection by uniting three distinct observational channels:
1. **Conversational Text** (*What* is communicated)
2. **Behavioral Interaction Dynamics** (*How* communication occurs over dialogue turns)
3. **Facial Affect Recognition** (*What* facial expressions indicate)

This document provides a technical specification of the software architecture, deep learning models, mathematical formulations, service layers, and frontend components.

---

## 2. Directory Structure

```
multimodal-mental-health/
├── app/                               # Streamlit Multipage Frontend
│   ├── components/
│   │   └── ui.py                      # Reusable UI widgets & Plotly charts
│   ├── pages/
│   │   ├── 1_Overview.py              # Architecture & dataset inventory
│   │   ├── 2_Text_Analysis.py         # MentalBERT 7-class inference
│   │   ├── 3_Facial_Emotion.py        # FER2013 CNN vision analysis
│   │   ├── 4_Behavioral_Analysis.py   # 13 turn metrics & withdrawal profiling
│   │   ├── 5_Fusion_Analysis.py       # 776-d Text + Behavioral fusion
│   │   └── 6_Research_Evaluation.py   # Benchmark reports & statistical tests
│   ├── styles/
│   │   └── custom.css                 # Premium dark research design system
│   └── main.py                        # Landing dashboard & system router
├── artifacts/
│   ├── configs/
│   │   └── model_manifest.json        # Central model registry
│   ├── encoders/                      # Tokenizers & label encoders
│   └── scalers/
│       └── behav_scaler_esconv.pkl    # Standardized behavioral scaler
├── data/
│   ├── samples/                       # Sample dialogue transcripts & CSVs
│   └── test_images/                   # Verification images
├── docs/
│   ├── implementation_audit.md        # Audit of notebooks vs research report
│   ├── model_recovery_report.md       # Autonomous model recovery & audit report
│   ├── research_report.pdf            # Accompanying research report
│   └── technical_documentation.md     # Technical system documentation
├── models/
│   ├── behavioral/
│   ├── facial/
│   │   └── fer_model.pt               # Trained FER2013 CNN weights
│   ├── fusion/
│   │   └── fusion_model.pt            # Trained Model C fusion weights
│   └── text/
│       └── text_model_final.pt        # Fine-tuned MentalBERT weights
├── notebooks/                         # Original Google Colab notebooks
│   ├── data_preparation+behavioral_features.ipynb
│   ├── facial_emotion_label.ipynb
│   ├── fusion_model.ipynb
│   └── text_model.ipynb
├── src/                               # Modular Python Backend
│   ├── features/
│   │   ├── __init__.py
│   │   └── behavioral.py              # Behavioral interaction extractor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── facial_model.py            # Custom 3-block PyTorch CNN
│   │   ├── fusion_model.py            # Text + Behavioral FusionModelWeighted
│   │   ├── text_model.py              # MentalBERT classifier & embedding extractor
│   │   └── train_fusion.py            # Reproducible fusion training pipeline
│   └── services/
│       ├── __init__.py
│       └── inference.py               # Unified MindTracePlatform service
├── tests/                             # Comprehensive automated test suite
│   ├── test_behavioral.py
│   ├── test_models.py
│   └── test_services.py
├── .env.example                       # Environment secrets template
├── .gitignore
├── requirements.txt                   # Dependency specification
├── run.sh                             # Application startup script
└── README.md                          # Project documentation
```

---

## 3. Mathematical & Model Specifications

### 3.1. Behavioral Feature Extraction Engine (`src/features/behavioral.py`)

The behavioral engine processes conversational turns: $\mathcal{T} = \{(s_i, m_i)\}_{i=1}^N$ where $s_i \in \{\text{user}, \text{assistant}\}$ and $m_i$ is the message string.

1. **Turn Counts**:
   $$\text{user\_turn\_count} = \sum_{i=1}^N \mathbb{I}(s_i = \text{user}), \quad \text{assistant\_turn\_count} = \sum_{i=1}^N \mathbb{I}(s_i = \text{assistant}), \quad \text{total\_turn\_count} = N$$

2. **Engagement Ratio**:
   $$\text{engagement\_ratio} = \frac{\text{user\_turn\_count}}{N}$$

3. **Message Lengths & Variability**:
   Let $\mathcal{W}_u = \{|m_i|_{\text{words}} : s_i = \text{user}\}$:
   $$\text{avg\_user\_msg\_length} = \frac{1}{|\mathcal{W}_u|}\sum_{w \in \mathcal{W}_u} w, \quad \text{user\_msg\_length\_std} = \sqrt{\frac{1}{|\mathcal{W}_u|-1}\sum_{w \in \mathcal{W}_u} (w - \bar{w})^2}$$

4. **Message Length Slope ($\beta$)**:
   Computed using ordinary least squares regression over the user turn index sequence:
   $$\beta = \frac{\sum_{k=1}^K (k - \bar{k})(w_k - \bar{w})}{\sum_{k=1}^K (k - \bar{k})^2}$$
   A negative slope indicates progressive verbal reduction / withdrawal.

5. **Dropout Metrics**:
   $$\text{last\_user\_turn} = \max \{i : s_i = \text{user}\}$$
   $$\text{dropout\_gap} = N - \text{last\_user\_turn}$$
   $$\text{dropout\_flag} = \mathbb{I}(\text{dropout\_gap} > 2)$$
   $$\text{ended\_by\_user} = \mathbb{I}(s_N = \text{user})$$

6. **Fusion Feature Vector (8 Dimensions)**:
   Ordered strictly as:
   $$[\text{total\_turn\_count}, \text{engagement\_ratio}, \text{avg\_user\_msg\_length}, \text{msg\_length\_slope}, \text{dropout\_flag}, \text{dropout\_gap}, \text{user\_msg\_length\_std}, \text{ended\_by\_user}]$$

---

### 3.2. Text Classification & Embedding Pipeline (`src/models/text_model.py`)

- **Encoder**: `mental/mental-bert-base-uncased` (BERT architecture with 12 hidden layers, 768 hidden dimensions, 12 attention heads, uncased vocabulary).
- **Classification Head**:
  $$\mathbf{h}_{\text{CLS}} = \text{BERT}(\mathbf{x})[:, 0, :] \in \mathbb{R}^{768}$$
  $$\mathbf{z}_{\text{text}} = \mathbf{W}_c \mathbf{h}_{\text{CLS}} + \mathbf{b}_c \in \mathbb{R}^{7}$$
  $$p(c|\mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=1}^7 \exp(z_j)}$$
- **Uncertainty Quantification**:
  Normalized Shannon Entropy:
  $$\mathcal{H}_{\text{norm}} = -\frac{\sum_{c=1}^7 p(c) \ln(p(c) + \epsilon)}{\ln(7)}$$
  Top-2 Confidence Margin:
  $$\Delta_{\text{margin}} = p_{(1)} - p_{(2)}$$

---

### 3.3. Facial Emotion Vision CNN (`src/models/facial_model.py`)

- **Input**: Image $\mathbf{I} \in \mathbb{R}^{3 \times 48 \times 48}$, normalized with $\mu = [0.5, 0.5, 0.5], \sigma = [0.5, 0.5, 0.5]$.
- **Architecture**:
  - Block 1: $\text{Conv2D}(3 \to 32, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($48 \to 24$)
  - Block 2: $\text{Conv2D}(32 \to 64, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($24 \to 12$)
  - Block 3: $\text{Conv2D}(64 \to 128, k=3, p=1) \to \text{ReLU} \to \text{MaxPool2d}(2)$ ($12 \to 6$)
  - Dense: $\text{Linear}(4608 \to 256) \to \text{ReLU} \to \text{Dropout}(0.5) \to \text{Linear}(256 \to 7)$
  - Total Parameters: **1,274,951**.

---

### 3.4. Multimodal Distress Fusion Model (`src/models/fusion_model.py`)

- **Inputs**:
  - Text: $\mathbf{e}_{\text{text}} \in \mathbb{R}^{768}$ (frozen MentalBERT CLS embedding)
  - Behavioral: $\mathbf{f}_{\text{behav}} = \text{StandardScaler}(\mathbf{x}_{\text{behav}}) \in \mathbb{R}^{8}$
  - Total Input Space: $\mathbb{R}^{776}$
- **Architecture (`FusionModelWeighted`)**:
  - Text Branch: $\mathbf{h}_t = \text{Dropout}_{0.4}(\text{ReLU}(\mathbf{W}_t \mathbf{e}_{\text{text}} + \mathbf{b}_t)) \in \mathbb{R}^{32}$
  - Behavioral Branch: $\mathbf{h}_b = \text{ReLU}(\mathbf{W}_b \mathbf{f}_{\text{behav}} + \mathbf{b}_b) \in \mathbb{R}^{32}$
  - Fusion Representation: $\mathbf{h}_{\text{fused}} = [\mathbf{h}_t \,\|\, \mathbf{h}_b] \in \mathbb{R}^{64}$
  - Classifier Head:
    $$\mathbf{h}_c = \text{Dropout}_{0.4}(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{fused}} + \mathbf{b}_1)) \in \mathbb{R}^{16}$$
    $$\mathbf{z}_{\text{distress}} = \mathbf{W}_2 \mathbf{h}_c + \mathbf{b}_2 \in \mathbb{R}^{2}$$
  - Total Parameters: **25,970**.
- **Objective Function**: Weighted Cross-Entropy Loss to handle the 77.3% / 22.7% class imbalance:
  $$\mathcal{L} = -\sum_{i} w_{y_i} \log p(y_i | \mathbf{x}_i), \quad w_0 \approx 2.2039, \; w_1 \approx 0.6467$$

---

## 4. Security, Privacy, and Research Integrity

1. **Zero Secret Leakage**: API tokens (e.g., `HF_TOKEN`) are strictly consumed from system environment variables or local `.env` files ignored by git.
2. **Ephemeral Memory Processing**: Audio/webcam frames and conversational transcripts submitted through the UI are held purely in volatile process RAM during inference; no persistent conversation logging occurs.
3. **No Fabricated Predictions**: If a model checkpoint is missing from disk, the platform refuses to generate mock or random outputs. It transparently explains which checkpoint is missing and shows the exact extracted input features.
4. **Distinction of Research Findings**: Published results (such as the 84.26% text accuracy) are clearly attributed to their original isolated datasets and never misrepresented as the accuracy of the overall multimodal system.
