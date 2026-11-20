# MINDTRACE: Multimodal Mental Health Risk Detection Research Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)

**MINDTRACE** is an academic research platform investigating whether uniting **conversational text**, **behavioral interaction patterns**, and **facial affect recognition** provides superior discriminative signal for psychological distress detection compared with relying solely on conversational text.

Developed as a demonstration and evaluation platform for university research, MINDTRACE adheres to strict scientific integrity, transparently delineating between **implemented experiments**, **validated model weights**, and **proposed future architectures**.

---

## 📌 Research Overview

### 1. The Core Research Question
Does combining *what* a person says, *how* they communicate over dialogue turns, and *what* their facial expressions indicate improve distress risk detection over text alone?

The research investigates three complementary observational channels:
1. **Conversational Text** (*What* is said): High-dimensional semantic language representations encoded via domain-specific Transformers (`mental/mental-bert-base-uncased`).
2. **Behavioral Interaction Dynamics** (*How* communication occurs): Quantitative dialogue progression metrics, such as turn count balance, message length trajectory slope, assistant-carrying ratio, and unanswered closing dropout gaps.
3. **Facial Affect** (*What* facial expressions indicate): Static 7-emotion expression probabilities classified by a custom 3-block Convolutional Neural Network trained on FER2013.

### 2. Implemented Research vs. Proposed Architecture
A fundamental scientific distinction must be maintained between the experimental work completed in the research notebooks and future proposals:
* **Implemented Fusion Experiment (Notebook 4):** Evaluates a **776-dimensional** multimodal representation combining **768 text dimensions** (MentalBERT) and **8 standardized behavioral interaction features** on the ESConv dataset to predict binary distress ($initial\_emotion\_intensity \ge 4$).
* **Proposed Tri-Modal Fusion (Research Report):** A theoretical **783-dimensional** architecture ($768 \text{ text} + 8 \text{ behavioral} + 7 \text{ facial emotion}$) intended for future datasets featuring synchronized audio-visual dialogue recordings (e.g., DAIC-WOZ, VEATIC). This model is clearly classified as **Proposed Architecture — Not Yet Implemented or Validated**.

---

## 🏗️ Technology Stack

* **Frontend & Visualization:** Streamlit (Custom Multipage Dark UI), Custom CSS, Plotly Interactive Charts.
* **Deep Learning Framework:** PyTorch, Torchvision.
* **Natural Language Processing:** Hugging Face Transformers, MentalBERT (`mental/mental-bert-base-uncased`), Tokenizers.
* **Computer Vision:** OpenCV Headless, Pillow.
* **Scientific Computing & Tabular:** NumPy, Pandas, Scikit-learn, SciPy, Joblib.
* **Testing:** Pytest.
* **Environment:** Python 3.11, Virtual Environment (`.venv`), `python-dotenv`.

---

## 📁 Repository Directory Structure

```
multimodal-mental-health/
├── app/                               # Streamlit Multipage Frontend
│   ├── components/
│   │   └── ui.py                      # Reusable dark UI cards, badges & Plotly charts
│   ├── pages/
│   │   ├── 1_Overview.py              # Architecture, research gap & dataset inventory
│   │   ├── 2_Text_Analysis.py         # MentalBERT 7-category inference & entropy
│   │   ├── 3_Facial_Emotion.py        # FER2013 CNN image & webcam affect classification
│   │   ├── 4_Behavioral_Analysis.py   # 13 turn metrics & withdrawal pattern analysis
│   │   ├── 5_Fusion_Analysis.py       # 776-d Text + Behavioral binary distress fusion
│   │   └── 6_Research_Evaluation.py   # Published classification reports & statistical tests
│   ├── styles/
│   │   └── custom.css                 # Premium dark research design system
│   └── main.py                        # Landing dashboard & system status router
├── artifacts/
│   ├── configs/
│   │   └── model_manifest.json        # Single source of truth model registry
│   ├── encoders/                      # Vocabulary & tokenizers
│   └── scalers/
│       └── behav_scaler_esconv.pkl    # Fitted StandardScaler for 8 behavioral features
├── data/
│   ├── samples/                       # Sample high/low distress transcripts & ESConv CSVs
│   └── test_images/                   # Sample vision inputs
├── docs/
│   ├── implementation_audit.md        # Rigorous comparison: Notebooks vs. Research Report
│   ├── research_report.pdf            # Original mentor project research report
│   └── technical_documentation.md     # In-depth architectural & mathematical specification
├── models/
│   ├── behavioral/
│   ├── facial/
│   │   └── fer_model.pt               # Custom PyTorch CNN weights (FER2013)
│   ├── fusion/
│   │   └── fusion_model.pt            # Model C FusionModelWeighted weights
│   └── text/
│       └── text_model_final.pt        # Fine-tuned 7-class MentalBERT weights
├── notebooks/                         # Original Google Colab research notebooks
│   ├── data_preparation+behavioral_features.ipynb
│   ├── facial_emotion_label.ipynb
│   ├── fusion_model.ipynb
│   └── text_model.ipynb
├── src/                               # Modular Python Backend Services
│   ├── features/
│   │   └── behavioral.py              # Pure 13-metric statistical feature extractor
│   ├── models/
│   │   ├── facial_model.py            # FacialEmotionCNN architecture & service
│   │   ├── fusion_model.py            # FusionModelWeighted & baseline architectures
│   │   ├── text_model.py              # MentalBERT classifier & embedding extractor
│   │   └── train_fusion.py            # Reproducible fusion training pipeline
│   └── services/
│       └── inference.py               # Unified MindTracePlatform orchestration service
├── tests/                             # Comprehensive automated test suite
│   ├── test_behavioral.py
│   ├── test_models.py
│   └── test_services.py
├── .env.example                       # Environment secrets template
├── .gitignore
├── requirements.txt                   # Production dependencies
├── run.sh                             # One-click startup script
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository & Create Environment
```bash
git clone https://github.com/your-username/multimodal-mental-health.git
cd multimodal-mental-health

# Create Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the environment template:
```bash
cp .env.example .env
```
Edit `.env` to configure your Hugging Face authentication token (required for accessing gated MentalBERT repositories):
```ini
HF_TOKEN=your_huggingface_token_here
```
> **Security Notice:** Never commit `.env` or hardcode tokens into source files. Git ignores `.env` automatically.

---

## 🚀 Running the Application

Launch the platform using the provided shell script:
```bash
./run.sh
```
Or directly via Streamlit:
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the complete automated test suite:
```bash
PYTHONPATH=. pytest tests/
```
The test suite validates:
* Mathematical correctness of all 13 behavioral metrics and withdrawal detection.
* Exact parameter counts for `FacialEmotionCNN` (1,274,951), `TextOnlyModel` (98,690), `BehavioralOnlyModel` (850), and `FusionModelWeighted` (25,970).
* 776-dimensional vector concatenation for multimodal fusion.
* Safe and graceful handling of missing model checkpoints without crashing.

---

## 📊 Published Research Benchmarks

| Modality / Experiment | Model Architecture | Dataset / Size | Accuracy | Macro F1 | Key Finding / Limitation |
|---|---|---|---|---|---|
| **Text Classification** | MentalBERT Uncased | Kaggle Sentiment (52,680) | **84.26%** | **82.90%** | Suicidal class recall lowest (71.0%), confused with Depression. |
| **Facial Affect** | 3-Block Custom CNN | FER2013 (35,887) | **58.79%** | **56.47%** | Posed static faces; affect $\ne$ psychiatric diagnosis. |
| **Text-Only Baseline (A)** | 768-d MLP | ESConv (908 convs) | 55.4% | 0.649 | AUROC: 0.496. |
| **Behavioral Baseline (B)**| 8-d MLP | ESConv (908 convs) | 44.8% | 0.583 | AUROC: 0.370. |
| **Multimodal Fusion (C)**  | 776-d FusionModelWeighted | ESConv (908 convs) | **57.6%** | **0.694** | Nominal F1 gain; paired AUROC t-test $p = 0.2957$ (not statistically significant). |

---

## ⚖️ Ethical & Clinical Disclaimers

> [!WARNING]
> **RESEARCH PROTOTYPE ONLY — NOT A CLINICAL DIAGNOSTIC TOOL**
> 
> * **No Medical Advice:** This software is an experimental prototype designed strictly for academic demonstration and machine learning research. It has not been clinically evaluated by medical authorities.
> * **No Definitive Intent Detection:** Predictions generated by these models must never be interpreted as definitive psychiatric assessments, suicidal crisis confirmations, or medical diagnoses.
> * **Facial Affect Caveat:** Surface facial expressions do not establish internal mental health state. A person experiencing depression may present a neutral or smiling face.
> * **Crisis Resources:** If you or someone you know is in crisis, please seek immediate help from certified mental health professionals or free crisis hotlines (e.g., US: dial 988; UK: 111; India: 14416 / KIRAN 1800-599-0019).
