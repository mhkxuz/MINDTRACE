"""
MINDTRACE: Multimodal Mental Health Research Platform
Main Application Landing & Overview Dashboard
"""

import streamlit as st
import pandas as pd
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_disclaimer,
    render_metric_card,
    render_status_badge,
)
from src.services.inference import MindTracePlatform

st.set_page_config(
    page_title="MINDTRACE | Multimodal Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()

# Initialize platform service
@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()
status = platform.get_system_status()

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 **MINDTRACE**")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#9CA3AF; margin-bottom:1rem;">
            Multimodal Mental Health Research Demonstration Platform
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### ⚡ **System Status**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.caption("Text (MentalBERT)")
        txt_status = "ready" if status["text_model"]["checkpoint_exists"] else "missing"
        txt_label = "Operational" if status["text_model"]["checkpoint_exists"] else "No Weights"
        st.markdown(render_status_badge(txt_status, txt_label), unsafe_allow_html=True)
    with col_s2:
        st.caption("Facial (FER CNN)")
        fac_status = "ready" if status["facial_model"]["checkpoint_exists"] else "missing"
        fac_label = "Operational" if status["facial_model"]["checkpoint_exists"] else "No Weights"
        st.markdown(render_status_badge(fac_status, fac_label), unsafe_allow_html=True)

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.caption("Behavioral")
        st.markdown(render_status_badge("ready", "Operational"), unsafe_allow_html=True)
    with col_s4:
        st.caption("Fusion (C)")
        fus_status = "ready" if status["fusion_model"]["checkpoint_exists"] else "missing"
        fus_label = "Operational" if status["fusion_model"]["checkpoint_exists"] else "No Weights"
        st.markdown(render_status_badge(fus_status, fus_label), unsafe_allow_html=True)

    st.divider()
    st.caption("Navigation: Use the sidebar pages to interact with individual modalities.")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#6B7280; margin-top:1.5rem;">
            Strict Academic Research Prototype<br>
            University Evaluation Demo
        </div>
        """,
        unsafe_allow_html=True,
    )

# Hero Header
render_header(
    title="MINDTRACE",
    subtitle="Multimodal Mental Health Risk Detection Using Behavioral Interaction Features, Conversational Text, and Real-Time Facial Affect Analysis",
)

render_disclaimer()

# Research Objective & Architecture Section
st.markdown("### 🔬 Research Objective & Theoretical Framework")

col_intro1, col_intro2 = st.columns([3, 2])

with col_intro1:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#818CF8; margin-top:0;">Core Research Question</h4>
            <p style="color:#CBD5E1; font-size:0.95rem; line-height:1.6;">
                Does combining <strong>conversational text</strong>, <strong>behavioral interaction patterns</strong>, and <strong>facial emotion features</strong> provide superior discriminative signal for mental distress detection compared to relying solely on conversational text?
            </p>
            <div style="margin-top:1rem; color:#94A3B8; font-size:0.88rem;">
                Traditional NLP approaches inspect only <em>what</em> a patient types. In real-world crisis dialogue, how a user communicates—their response dynamics, shrinking message lengths, and withdrawal patterns—may carry critical diagnostic clues.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_intro2:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#06B6D4; margin-top:0;">Tri-Modal Framework</h4>
            <ul style="color:#CBD5E1; font-size:0.88rem; line-height:1.7; padding-left:1.2rem;">
                <li><strong>Modality 1: Conversational Text</strong><br><span style="color:#9CA3AF;">MentalBERT 768-d language representations</span></li>
                <li><strong>Modality 2: Behavioral Interaction</strong><br><span style="color:#9CA3AF;">8 engineered turn & withdrawal metrics</span></li>
                <li><strong>Modality 3: Facial Affect (FER2013)</strong><br><span style="color:#9CA3AF;">7-class static facial expression probabilities</span></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Published Benchmark Performance Cards
st.markdown("### 📊 Documented Experimental Benchmarks")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    render_metric_card(
        label="MentalBERT Text Acc",
        value="84.26%",
        subtext="Macro F1: 82.90% (52.6K samples)",
    )

with col_m2:
    render_metric_card(
        label="FER2013 CNN Acc",
        value="58.79%",
        subtext="Macro F1: 56.47% (7.1K test images)",
    )

with col_m3:
    render_metric_card(
        label="Fusion (Model C) F1",
        value="0.694",
        subtext="5-Seed Holdout (ESConv distress)",
    )

with col_m4:
    render_metric_card(
        label="Fusion p-value (AUROC)",
        value="p = 0.2957",
        subtext="Not statistically significant vs Text",
    )

st.markdown("---")

# Implementation Status Table
st.markdown("### 🛠️ Model Inventory & Implementation Audit")

audit_data = [
    {
        "Modality / Model": "Text Classifier (MentalBERT)",
        "Architecture": "mental-bert-base-uncased (7 classes)",
        "Training Dataset": "Kaggle Sentiment Analysis (52.6K)",
        "Notebook": "text_model.ipynb",
        "Implementation": "Implemented",
        "Trained": "Yes (Notebook)",
        "Operational": "Operational" if status["text_model"]["checkpoint_exists"] else "Unavailable (No Weights)",
        "Checkpoint on Disk": "Found" if status["text_model"]["checkpoint_exists"] else "Missing",
        "Operational Notes": "Fine-tuned 3 epochs; 84.26% test acc.",
    },
    {
        "Modality / Model": "Facial Affect (FER CNN)",
        "Architecture": "3-Block CNN (48x48 -> 7 classes)",
        "Training Dataset": "FER2013 (28.7K train / 7.1K test)",
        "Notebook": "facial_emotion_label.ipynb",
        "Implementation": "Implemented",
        "Trained": "Yes (Notebook)",
        "Operational": "Operational" if status["facial_model"]["checkpoint_exists"] else "Unavailable (No Weights)",
        "Checkpoint on Disk": "Found" if status["facial_model"]["checkpoint_exists"] else "Missing",
        "Operational Notes": "Trained 10 epochs; 58.79% test acc.",
    },
    {
        "Modality / Model": "Behavioral Extractor",
        "Architecture": "Rule-based & Linear Regression",
        "Training Dataset": "ESConv dialogue records",
        "Notebook": "data_preparation+behavioral_features.ipynb",
        "Implementation": "Implemented",
        "Trained": "N/A (Rule Engine)",
        "Operational": "Operational",
        "Checkpoint on Disk": "Functional (Scaler Loaded)",
        "Operational Notes": "Computes 13 metrics + 8 standardized fusion features.",
    },
    {
        "Modality / Model": "Fusion Model C (Text + Behav)",
        "Architecture": "FusionModelWeighted (776 dims -> 2)",
        "Training Dataset": "ESConv (908 conversations)",
        "Notebook": "fusion_model.ipynb",
        "Implementation": "Implemented",
        "Trained": "Yes (Trained & Saved)",
        "Operational": "Operational" if status["fusion_model"]["checkpoint_exists"] else "Unavailable",
        "Checkpoint on Disk": "Found" if status["fusion_model"]["checkpoint_exists"] else "Missing",
        "Operational Notes": "5-seed holdout & 5-fold CV evaluated; trained checkpoint verified.",
    },
    {
        "Modality / Model": "Proposed Tri-Modal Fusion",
        "Architecture": "Conceptual 783-d (Text 768 + Behav 8 + Facial 7)",
        "Training Dataset": "Unpaired (Proposed Future Work)",
        "Notebook": "None (Report Proposal Only)",
        "Implementation": "Proposed Architecture",
        "Trained": "No",
        "Operational": "Not Operational (Proposed)",
        "Checkpoint on Disk": "Not Implemented",
        "Operational Notes": "Clearly marked as proposed research extension.",
    },
]

st.markdown(
    """
    <div style="font-size:0.85rem; color:#9CA3AF; margin-bottom:0.75rem;">
        The table below provides strict transparency regarding which components were trained, evaluated, and saved versus theoretical proposals.
        Statuses distinguish: <strong>Implemented</strong> (code exists), <strong>Trained</strong> (model weights exist), <strong>Operational</strong> (can execute live inference), and <strong>Proposed</strong> (future research architecture).
    </div>
    """,
    unsafe_allow_html=True,
)

st.table(pd.DataFrame(audit_data))

# Navigation Cards
st.markdown("### 🧭 Application Modules")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#6366F1; margin-top:0;">1. Text Analysis</h4>
            <p style="color:#94A3B8; font-size:0.85rem;">
                MentalBERT 7-class text inference. Classifies mental health categories with entropy uncertainty quantification.
            </p>
            <a href="/Text_Analysis" target="_self" style="color:#818CF8; font-weight:600; text-decoration:none; font-size:0.85rem;">Open Module →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#06B6D4; margin-top:0;">2. Facial Affect</h4>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Custom 3-block CNN for FER2013 facial emotion inference via file upload or optional live webcam capture.
            </p>
            <a href="/Facial_Emotion" target="_self" style="color:#22D3EE; font-weight:600; text-decoration:none; font-size:0.85rem;">Open Module →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#10B981; margin-top:0;">3. Behavioral Analysis</h4>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Analyzes dialogue transcripts or CSVs, extracting 13 interaction metrics, slope regression, and withdrawal indicators.
            </p>
            <a href="/Behavioral_Analysis" target="_self" style="color:#34D399; font-weight:600; text-decoration:none; font-size:0.85rem;">Open Module →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

c4, c5, _ = st.columns(3)

with c4:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#F59E0B; margin-top:0;">4. Multimodal Fusion</h4>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Evaluates the 776-dimensional binary distress classifier combining MentalBERT embeddings with standardized behavioral dynamics.
            </p>
            <a href="/Fusion_Analysis" target="_self" style="color:#FBBF24; font-weight:600; text-decoration:none; font-size:0.85rem;">Open Module →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#EC4899; margin-top:0;">5. Research & Evaluation</h4>
            <p style="color:#94A3B8; font-size:0.85rem;">
                Comprehensive experimental results, confusion matrices, 5-seed metrics, CV cross-validation, and p-value statistical significance.
            </p>
            <a href="/Research_Evaluation" target="_self" style="color:#F472B6; font-weight:600; text-decoration:none; font-size:0.85rem;">Open Module →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
