"""
MINDTRACE: Overview & Research Architecture
Page 1: Comprehensive system overview, dataset inventory, and research gap.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_disclaimer,
)
from src.services.inference import MindTracePlatform

st.set_page_config(
    page_title="Overview | MINDTRACE",
    page_icon="🔬",
    layout="wide",
)

load_custom_css()

@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()
status = platform.get_system_status()

render_header(
    title="Research Overview & Platform Architecture",
    subtitle="MINDTRACE: Multimodal Mental Health Risk Detection",
)

render_disclaimer()

# Section 1: The Research Gap
st.markdown("### 🎯 1. The Research Gap")

st.markdown(
    """
    <div class="research-card">
        <h4 style="color:#818CF8; margin-top:0;">What Exists vs. What is Missing</h4>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1.5rem; margin-top:1rem;">
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border-left:3px solid #3B82F6;">
                <strong style="color:#60A5FA;">Existing Research Approaches:</strong>
                <p style="color:#94A3B8; font-size:0.88rem; margin-top:0.4rem;">
                    • Text sentiment and emotion classification from static social media posts.<br>
                    • Speech prosody and acoustic pitch features in clinical interview recordings.<br>
                    • Isolated facial action unit (AU) extraction under laboratory settings.
                </p>
            </div>
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border-left:3px solid #EC4899;">
                <strong style="color:#F472B6;">The MINDTRACE Research Focus:</strong>
                <p style="color:#94A3B8; font-size:0.88rem; margin-top:0.4rem;">
                    • Investigating whether <em>dialogue interaction dynamics</em> (shrinking messages, withdrawal gaps, engagement slope) provide incremental discriminative power when fused with NLP embeddings.<br>
                    • Combining conversational text, interaction behavior, and facial emotion signals into a unified risk detection paradigm.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 2: Architecture
st.markdown("### 🧩 2. Tri-Modal Architecture Framework")

col_a1, col_a2, col_a3 = st.columns(3)

with col_a1:
    st.markdown(
        """
        <div class="research-card" style="height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="color:#6366F1; margin:0;">Modality 1: Text</h4>
                <span class="status-pill pill-green">Implemented</span>
            </div>
            <div style="margin:0.8rem 0; font-size:1.2rem; font-weight:700; color:#F8FAFC;">MentalBERT</div>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.6;">
                <strong>Encoder:</strong> <code>mental-bert-base-uncased</code><br>
                <strong>Pretraining:</strong> 52,680 Kaggle sentiment posts<br>
                <strong>Vector Output:</strong> 768-dimensional CLS embedding<br>
                <strong>Task:</strong> Captures <em>WHAT</em> the person says.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_a2:
    st.markdown(
        """
        <div class="research-card" style="height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="color:#10B981; margin:0;">Modality 2: Behavioral</h4>
                <span class="status-pill pill-green">Implemented</span>
            </div>
            <div style="margin:0.8rem 0; font-size:1.2rem; font-weight:700; color:#F8FAFC;">Interaction Dynamics</div>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.6;">
                <strong>Engine:</strong> Deterministic statistical extraction<br>
                <strong>Source:</strong> ESConv multi-turn dialogues<br>
                <strong>Vector Output:</strong> 8 standardized numerical features<br>
                <strong>Task:</strong> Captures <em>HOW</em> the person communicates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_a3:
    st.markdown(
        """
        <div class="research-card" style="height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="color:#06B6D4; margin:0;">Modality 3: Facial Affect</h4>
                <span class="status-pill pill-amber">Standalone CNN</span>
            </div>
            <div style="margin:0.8rem 0; font-size:1.2rem; font-weight:700; color:#F8FAFC;">FER2013 CNN</div>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.6;">
                <strong>Architecture:</strong> Custom 3-block PyTorch CNN<br>
                <strong>Dataset:</strong> 35,887 48x48 facial images<br>
                <strong>Vector Output:</strong> 7 emotion probability scores<br>
                <strong>Task:</strong> Captures <em>WHAT</em> facial expression indicates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Section 3: Implemented Fusion vs Proposed Fusion
st.markdown("### ⚖️ 3. Implemented Fusion vs. Proposed Architecture")

st.markdown(
    """
    <div class="research-card">
        <table style="width:100%; border-collapse:collapse; color:#CBD5E1; font-size:0.88rem;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1); text-align:left; color:#9CA3AF;">
                    <th style="padding:0.75rem;">Feature</th>
                    <th style="padding:0.75rem;">Currently Implemented Experiment (Notebook 4)</th>
                    <th style="padding:0.75rem;">Mentor Report Proposed Architecture</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.75rem; font-weight:600;">Active Modalities</td>
                    <td style="padding:0.75rem; color:#34D399;">2 Modalities: Text + Behavioral Interaction</td>
                    <td style="padding:0.75rem; color:#A78BFA;">3 Modalities: Text + Behavioral + Facial Affect</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.75rem; font-weight:600;">Input Dimensions</td>
                    <td style="padding:0.75rem; font-family:monospace; color:#38BDF8;">776 Dimensions (768 Text + 8 Behavioral)</td>
                    <td style="padding:0.75rem; font-family:monospace; color:#A78BFA;">783 Dimensions (768 Text + 8 Behav + 7 Facial)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.75rem; font-weight:600;">Classification Target</td>
                    <td style="padding:0.75rem;">Binary Distress (ESConv initial intensity ≥ 4)</td>
                    <td style="padding:0.75rem;">Multimodal Distress / Crisis Escalation</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.75rem; font-weight:600;">Experimental Findings</td>
                    <td style="padding:0.75rem;">Fusion F1: 0.694 vs Text F1: 0.649. Paired AUROC t-test p=0.2957 (not statistically significant).</td>
                    <td style="padding:0.75rem; color:#F59E0B;">Theoretical hypothesis; paired video-dialogue dataset not yet trained.</td>
                </tr>
                <tr>
                    <td style="padding:0.75rem; font-weight:600;">Application Classification</td>
                    <td style="padding:0.75rem;"><span class="status-pill pill-green">Implemented Experiment</span></td>
                    <td style="padding:0.75rem;"><span class="status-pill pill-purple">Proposed Architecture</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 4: Datasets
st.markdown("### 📚 4. Datasets Used in the Research")

c_d1, c_d2, c_d3 = st.columns(3)

with c_d1:
    st.markdown(
        """
        <div class="research-card">
            <h5 style="color:#818CF8; margin-top:0;">ESConv (Emotional Support)</h5>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.5;">
                • <strong>Source:</strong> Hugging Face (<code>thu-coai/esconv</code>)<br>
                • <strong>Samples:</strong> 908 cleaned conversations<br>
                • <strong>Metadata:</strong> Situation description, turn-by-turn dialogue, seeker empathy rating, initial & final emotion intensity (1–5).<br>
                • <strong>Role:</strong> Extraction of 13 interaction metrics and binary distress ground-truth labels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_d2:
    st.markdown(
        """
        <div class="research-card">
            <h5 style="color:#38BDF8; margin-top:0;">Sentiment for Mental Health</h5>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.5;">
                • <strong>Source:</strong> Kaggle<br>
                • <strong>Samples:</strong> 52,680 text records<br>
                • <strong>Categories:</strong> Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal.<br>
                • <strong>Role:</strong> Domain fine-tuning of MentalBERT; text classifier benchmark.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_d3:
    st.markdown(
        """
        <div class="research-card">
            <h5 style="color:#34D399; margin-top:0;">FER2013 Facial Dataset</h5>
            <p style="color:#94A3B8; font-size:0.85rem; line-height:1.5;">
                • <strong>Source:</strong> Kaggle<br>
                • <strong>Samples:</strong> 28,709 train, 7,178 test images<br>
                • <strong>Classes:</strong> Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.<br>
                • <strong>Role:</strong> Training custom 3-block CNN for static facial expression affect classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
