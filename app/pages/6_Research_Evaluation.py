"""
MINDTRACE: Research & Evaluation Dashboard
Page 6: Comprehensive Academic Evaluation, Benchmark Verification, and Statistical Analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_disclaimer,
)

st.set_page_config(
    page_title="Research Evaluation | MINDTRACE",
    page_icon="📊",
    layout="wide",
)

load_custom_css()

render_header(
    title="Comprehensive Research & Evaluation Dashboard",
    subtitle="Rigorous Benchmark Analysis, Cross-Validation Results, and Scientific Findings",
)

render_disclaimer()

tab_benchmarks, tab_cv, tab_training, tab_limits = st.tabs([
    "📈 Modality Benchmarks & Classification Reports",
    "🔄 5-Seed & Cross-Validation Fusion Analysis",
    "📉 Training Dynamics & Convergence",
    "⚠️ Research Limitations & Proposed 3-Branch Work",
])

with tab_benchmarks:
    st.markdown("### 1. Documented Classification Reports (Original Experiments)")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📝 MentalBERT 7-Class Text Classification")
        st.caption("Trained on 52,680 Kaggle sentiment records for 3 epochs.")

        text_rep = [
            {"Category": "Anxiety", "Precision": 0.90, "Recall": 0.86, "F1": 0.88, "Support": 384},
            {"Category": "Bipolar", "Precision": 0.88, "Recall": 0.85, "F1": 0.87, "Support": 278},
            {"Category": "Depression", "Precision": 0.78, "Recall": 0.81, "F1": 0.79, "Support": 1540},
            {"Category": "Normal", "Precision": 0.95, "Recall": 0.97, "F1": 0.96, "Support": 1635},
            {"Category": "Personality disorder", "Precision": 0.85, "Recall": 0.72, "F1": 0.78, "Support": 107},
            {"Category": "Stress", "Precision": 0.79, "Recall": 0.80, "F1": 0.79, "Support": 259},
            {"Category": "Suicidal", "Precision": 0.75, "Recall": 0.71, "F1": 0.73, "Support": 1065},
        ]
        st.table(pd.DataFrame(text_rep))
        st.info("Overall Test Accuracy: **84.26%** | Macro F1: **82.90%** | Weighted F1: **84.21%**")

    with col_t2:
        st.markdown("#### 🎭 FER2013 7-Emotion Facial Affect Classification")
        st.caption("Trained on 28,709 images for 10 epochs. Evaluated on 7,178 test images.")

        fer_rep = [
            {"Emotion": "Angry", "Precision": 0.46, "Recall": 0.57, "F1": 0.51, "Support": 958},
            {"Emotion": "Disgust", "Precision": 0.42, "Recall": 0.65, "F1": 0.51, "Support": 111},
            {"Emotion": "Fear", "Precision": 0.41, "Recall": 0.45, "F1": 0.43, "Support": 1024},
            {"Emotion": "Happy", "Precision": 0.83, "Recall": 0.77, "F1": 0.80, "Support": 1774},
            {"Emotion": "Neutral", "Precision": 0.54, "Recall": 0.56, "F1": 0.55, "Support": 1233},
            {"Emotion": "Sad", "Precision": 0.49, "Recall": 0.35, "F1": 0.41, "Support": 1247},
            {"Emotion": "Surprise", "Precision": 0.72, "Recall": 0.77, "F1": 0.74, "Support": 831},
        ]
        st.table(pd.DataFrame(fer_rep))
        st.info("Overall Test Accuracy: **58.79%** | Macro F1: **56.47%** | Weighted Avg F1: **59.00%**")

with tab_cv:
    st.markdown("### 2. Multi-Seed & 5-Fold Cross-Validation Fusion Comparison")

    col_cv1, col_cv2 = st.columns([3, 2])

    with col_cv1:
        st.markdown("##### 5-Seed Holdout Evaluation (Mean ± Std)")
        seed_data = [
            {"Model": "Model A - Text Only", "Accuracy": "55.4% ± 11.7%", "Balanced Acc": "52.7% ± 3.3%", "F1-Score": "0.649 ± 0.155", "AUROC": "0.496 ± 0.023"},
            {"Model": "Model B - Behavioral Only", "Accuracy": "44.8% ± 3.0%", "Balanced Acc": "38.5% ± 1.8%", "F1-Score": "0.583 ± 0.035", "AUROC": "0.370 ± 0.022"},
            {"Model": "Model C - Fusion (Text+Behav)", "Accuracy": "57.6% ± 3.5%", "Balanced Acc": "51.4% ± 3.5%", "F1-Score": "0.694 ± 0.034", "AUROC": "0.477 ± 0.025"},
        ]
        st.table(pd.DataFrame(seed_data))

        st.markdown("##### 5-Fold Cross-Validation (Mean ± Std)")
        cv_data = [
            {"Model": "Model A - Text Only", "CV Balanced Acc": "0.556 ± 0.039", "CV F1-Score": "0.655 ± 0.153", "CV AUROC": "0.573 ± 0.046"},
            {"Model": "Model B - Behavioral Only", "CV Balanced Acc": "0.525 ± 0.009", "CV F1-Score": "0.527 ± 0.280", "CV AUROC": "0.506 ± 0.073"},
            {"Model": "Model C - Fusion (Text+Behav)", "CV Balanced Acc": "0.549 ± 0.043", "CV F1-Score": "0.710 ± 0.052", "CV AUROC": "0.559 ± 0.052"},
        ]
        st.table(pd.DataFrame(cv_data))

    with col_cv2:
        st.markdown("##### Comparative Performance Chart")
        models = ["Model A (Text)", "Model B (Behav)", "Model C (Fusion)"]
        f1_vals = [0.649, 0.583, 0.694]
        auroc_vals = [0.496, 0.370, 0.477]

        fig_comp = go.Figure(data=[
            go.Bar(name="F1-Score", x=models, y=f1_vals, marker_color="#6366F1"),
            go.Bar(name="AUROC", x=models, y=auroc_vals, marker_color="#06B6D4"),
        ])
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 0.9]),
            margin=dict(l=10, r=10, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown(
        """
        <div class="research-disclaimer" style="border-left-color: #EC4899;">
            <strong>STATISTICAL SIGNIFICANCE ANALYSIS:</strong>
            A paired t-test comparing Model C (Fusion) against Model A (Text Only) on cross-validation AUROC scores yielded:
            <code>Mean Diff = -0.0138</code> with <code>p = 0.2957</code>.
            Because <em>p > 0.05</em>, we fail to reject the null hypothesis. The empirical results show that the addition of the 8 behavioral features did not yield a statistically significant improvement in discriminative ability over text embeddings alone on this ESConv task.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_training:
    st.markdown("### 3. Documented Training Dynamics Across Epochs")

    col_curv1, col_curv2 = st.columns(2)

    with col_curv1:
        st.markdown("#### MentalBERT Fine-Tuning Convergence (Notebook 2)")
        epochs_text = [1, 2, 3]
        train_acc_text = [0.789, 0.861, 0.904]
        val_acc_text = [0.824, 0.838, 0.840]

        fig_text = go.Figure()
        fig_text.add_trace(go.Scatter(x=epochs_text, y=train_acc_text, mode="lines+markers", name="Train Accuracy", line=dict(color="#8B5CF6", width=2)))
        fig_text.add_trace(go.Scatter(x=epochs_text, y=val_acc_text, mode="lines+markers", name="Val Accuracy", line=dict(color="#38BDF8", width=2, dash="dash")))
        fig_text.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(title="Epoch", dtick=1, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Accuracy", gridcolor="rgba(255,255,255,0.06)", range=[0.7, 0.95]),
            height=280,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_text, use_container_width=True)
        st.caption("Training converged cleanly by epoch 3 before overfitting commenced.")

    with col_curv2:
        st.markdown("#### FER2013 CNN Training Progression (Notebook 3)")
        epochs_fer = list(range(1, 11))
        train_acc_fer = [0.607, 0.625, 0.641, 0.665, 0.680, 0.697, 0.710, 0.725, 0.738, 0.750]

        fig_fer = go.Figure()
        fig_fer.add_trace(go.Scatter(x=epochs_fer, y=train_acc_fer, mode="lines+markers", name="Train Accuracy", line=dict(color="#10B981", width=2)))
        fig_fer.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(title="Epoch", dtick=1, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Train Accuracy", gridcolor="rgba(255,255,255,0.06)"),
            height=280,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_fer, use_container_width=True)
        st.caption("Trained with Adam (lr=1e-3), reaching 58.79% generalization accuracy on the test partition.")

with tab_limits:
    st.markdown("### 4. Methodological Limitations & Proposed Tri-Modal Work")

    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#F43F5E; margin-top:0;">Documented Research Limitations</h4>
            <ol style="color:#CBD5E1; font-size:0.88rem; line-height:1.7; padding-left:1.2rem;">
                <li><strong>Unpaired Modalities:</strong> The FER2013 dataset contains isolated static faces, whereas ESConv consists solely of text dialogue. There was no synchronized audio-visual dialogue dataset linking facial affect directly to conversational turns in the existing experiments.</li>
                <li><strong>Absence of Millisecond Temporal Logging:</strong> Behavioral features in ESConv are derived from turn counts and message word length regressions. Real-time typing dynamics (keystroke latency, inter-turn pauses) were not recorded in the source benchmark.</li>
                <li><strong>Severe Class Imbalance in ESConv Distress:</strong> The binary cutoff (<code>initial_emotion_intensity ≥ 4</code>) produced a 77.3% high distress / 22.7% low distress split, creating skew that necessitated weighted cross-entropy loss.</li>
                <li><strong>Suicidal Detection Recall Gap:</strong> MentalBERT achieved only 71.0% recall on the Suicidal class in the 7-class text benchmark, showing significant lexical overlap with Depression.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="research-card">
            <h4 style="color:#A78BFA; margin-top:0;">Proposed 3-Branch Architecture (Future Direction)</h4>
            <p style="color:#94A3B8; font-size:0.88rem;">
                The project report envisions a 783-dimensional unified architecture:
                <code>768 Text (MentalBERT) + 8 Behavioral Dynamics + 7 Facial Emotion Probabilities = 783 Dimensions</code>.
                Training this architecture requires multimodal datasets with synchronized video recordings (such as <strong>DAIC-WOZ</strong> or <strong>VEATIC</strong>).
                Until paired data is collected and formally trained, this 3-branch model remains a proposed theoretical concept and is not presented as an existing completed model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
