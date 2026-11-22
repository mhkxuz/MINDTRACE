"""
MINDTRACE: Text Analysis Module
Page 2: MentalBERT 7-Category Text Classification Interface
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_disclaimer,
    render_metric_card,
    plot_probability_distribution,
)
from src.services.inference import MindTracePlatform
from src.models.text_model import TEXT_BENCHMARK_METRICS

st.set_page_config(
    page_title="Text Analysis | MINDTRACE",
    page_icon="📝",
    layout="wide",
)

load_custom_css()

@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()
status = platform.get_system_status()["text_model"]

render_header(
    title="Conversational Text Analysis (MentalBERT)",
    subtitle="Fine-Tuned Domain Transformer for Mental Health Text Categorization",
)

render_disclaimer()

# Benchmark & Model Spec Bar
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    render_metric_card("Base Architecture", "MentalBERT", "mental-bert-base-uncased")
with col_b2:
    render_metric_card("Reproduced Accuracy", "83.30%", "Held-out test split (5,268 records)")
with col_b3:
    render_metric_card("Reproduced Macro F1", "80.88%", "Suicidal Recall: 75.77% | W-F1: 83.25%")
with col_b4:
    ckpt_text = "Operational (Local)" if status["checkpoint_exists"] else "Weights Missing"
    render_metric_card("Model Status", ckpt_text, status["checkpoint_path"])

st.markdown("---")

st.info(
    "💡 **Dual-Encoder Architecture Notice:**  \n"
    "This module performs seven-class mental-health text categorization using the fine-tuned MentalBERT sequence classifier "
    "(Epoch 2 best checkpoint) loaded locally from `models/text/mentalbert_best_export/` without external network dependencies.  \n"
    "To prevent representation shift and preserve exact feature compatibility, the **Multimodal Distress Fusion Module (Page 5)** "
    "exclusively utilizes the **frozen base MentalBERT encoder** (`mental/mental-bert-base-uncased`) for 768-d CLS embedding extraction.  \n"
    "*Scientific Qualification: Model confidence reflects statistical classification probability over dataset-defined categories. "
    "Outputs do not constitute clinical diagnoses or validated suicide-risk assessments.*"
)

# Main Interface Tabs
tab_single, tab_conv, tab_bench = st.tabs([
    "💬 Single Statement Analysis",
    "🗨️ Conversation Dialogue Analysis",
    "📊 Published Benchmark & Weak-Class Analysis",
])

with tab_single:
    st.markdown("#### Enter Text Sample")
    st.caption("Input a patient message, clinical statement, or conversational thought.")

    sample_texts = {
        "Custom Input": "",
        "Sample 1 (High Anxiety)": "I cannot stop my heart from pounding whenever I have to present at work. I feel completely paralyzed and terrified.",
        "Sample 2 (Depression/Withdrawal)": "I don't enjoy anything anymore. Getting out of bed feels pointless and I have no energy left for anyone.",
        "Sample 3 (Normal / Stress Management)": "I had a busy workday with back-to-back meetings, but I am going for a walk to decompress and recharge.",
        "Sample 4 (Suicidal / Critical Alert)": "I just want the pain to stop forever. Everyone in my life would be much better off without me here.",
    }

    selected_sample = st.selectbox("Load sample statement:", list(sample_texts.keys()))
    default_text = sample_texts[selected_sample]

    user_text = st.text_area(
        "Statement Text:",
        value=default_text,
        height=110,
        placeholder="Type or paste conversational text here...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_inference = st.button("Run MentalBERT Inference", key="btn_single_text")

    if run_inference:
        if not user_text.strip():
            st.warning("Please provide non-empty text input before running analysis.")
        else:
            with st.spinner("Analyzing text with MentalBERT..."):
                res = platform.analyze_text(user_text)

            if not res.get("success"):
                st.error(f"⚠️ {res.get('error')}")
                st.markdown(
                    f"""
                    <div class="research-disclaimer" style="border-left-color: #F59E0B;">
                        <strong>Checkpoint Setup Notice:</strong> The fine-tuned weights file <code>{status['checkpoint_path']}</code> was not found on disk.
                        In accordance with research integrity, random weights or mock predictions are strictly prohibited.
                        To enable live inference, place <code>text_model_final.pt</code> into <code>models/text/</code> or provide valid Hugging Face credentials.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                pred_class = res["predicted_class"]
                conf = res["confidence"]
                entropy = res.get("entropy", 0.0)
                margin = res.get("margin", 0.0)

                st.success(f"**Predicted Category:** `{pred_class}` (Confidence: {conf*100:.1f}%)")

                col_res1, col_res2 = st.columns([3, 2])
                with col_res1:
                    fig = plot_probability_distribution(
                        res["probabilities"],
                        title="MentalBERT 7-Class Probability Distribution",
                        highlight_label=pred_class,
                    )
                    st.plotly_chart(fig, width='stretch')

                with col_res2:
                    st.markdown("##### 🔍 Uncertainty Quantification")
                    st.markdown(
                        f"""
                        <div class="research-card">
                            <div style="margin-bottom:0.75rem;">
                                <span style="font-size:0.8rem; color:#9CA3AF;">PREDICTION MARGIN</span><br>
                                <strong style="font-size:1.2rem; color:#38BDF8;">{margin:.3f}</strong>
                                <div style="font-size:0.75rem; color:#64748B;">Difference between top-1 and top-2 class probability.</div>
                            </div>
                            <div>
                                <span style="font-size:0.8rem; color:#9CA3AF;">NORMALIZED ENTROPY</span><br>
                                <strong style="font-size:1.2rem; color:{'#F43F5E' if entropy > 0.6 else '#10B981'};">{entropy:.3f}</strong>
                                <div style="font-size:0.75rem; color:#64748B;">Higher values indicate greater classification uncertainty.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with tab_conv:
    st.markdown("#### Conversation-Level Text Analysis")
    st.caption("Classify multi-turn dialogue with turn-level decomposition and aggregation.")

    sample_dialogue_default = """User: I've been feeling so alone lately and work is suffocating me.
Assistant: That sounds very heavy. How long have you felt this way?
User: A few weeks now. I don't feel like talking to my friends anymore.
Assistant: Withdrawal is often a sign of overwhelming exhaustion. Please take gentle care of yourself."""

    conv_text = st.text_area(
        "Dialogue Transcript (User / Assistant turns):",
        value=sample_dialogue_default,
        height=140,
    )

    agg_mode = st.radio(
        "Text Aggregation Mode:",
        ["user_only", "all"],
        format_func=lambda x: "User Turns Only (Recommended - Seeker focus)" if x == "user_only" else "All Dialogue Turns (Whole conversation)",
        horizontal=True,
    )

    if st.button("Analyze Conversation Text", key="btn_conv_text"):
        if not conv_text.strip():
            st.warning("Please provide conversation transcript text.")
        else:
            with st.spinner("Processing conversation text..."):
                res_conv = platform.analyze_text_conversation(conv_text, aggregate_mode=agg_mode)

            if not res_conv.get("success"):
                st.error(f"⚠️ {res_conv.get('aggregate_prediction', {}).get('error', 'Inference could not be completed.')}")
                st.info(f"Checkpoint required at: `{status['checkpoint_path']}`.")
            else:
                agg_pred = res_conv["aggregate_prediction"]
                st.success(f"**Overall Conversation Category:** `{agg_pred['predicted_class']}` (Confidence: {agg_pred['confidence']*100:.1f}%)")

                c1, c2 = st.columns([3, 2])
                with c1:
                    fig = plot_probability_distribution(
                        agg_pred["probabilities"],
                        title="Aggregated Conversation 7-Class Distribution",
                        highlight_label=agg_pred["predicted_class"],
                    )
                    st.plotly_chart(fig, width='stretch')
                with c2:
                    st.markdown("##### Turn Breakdown")
                    if res_conv.get("turn_predictions"):
                        for tp in res_conv["turn_predictions"]:
                            st.markdown(f"**Turn {tp['turn_index']}**: `{tp['predicted_class']}` ({tp['confidence']*100:.1f}%)")
                            st.caption(f'"{tp["message"]}"')
                    else:
                        st.caption("No individual user turn predictions available.")

with tab_bench:
    st.markdown("#### 🔬 Benchmark Performance & Evaluation Breakdown")
    st.markdown(
        """
        The fine-tuned MentalBERT model (`models/text/mentalbert_best_export`) was evaluated on the 
        held-out test split (5,268 records) of the Kaggle Sentiment Analysis for Mental Health dataset.
        Below are the verified held-out test evaluation results compared against historical reference metrics:
        """
    )

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    with c_b1:
        st.metric("Test Accuracy", "83.30%", "Historical: 84.26%")
    with c_b2:
        st.metric("Macro F1", "80.88%", "Historical: 82.90%")
    with c_b3:
        st.metric("Weighted F1", "83.25%", "Historical: 84.21%")
    with c_b4:
        st.metric("Suicidal Recall", "75.77%", "Historical: 71.00%")

    st.markdown("##### Per-Class Performance on Held-Out Test Set")
    bench_df = pd.DataFrame([
        {"Category": k, "Precision": f"{v['precision']*100:.1f}%", "Recall": f"{v['recall']*100:.1f}%", "F1-Score": f"{v['f1']*100:.1f}%"}
        for k, v in TEXT_BENCHMARK_METRICS["class_metrics"].items()
    ])
    st.table(bench_df)

    st.markdown(
        """
        <div class="clinical-warning">
            <strong>RESEARCH & CLINICAL DISCLAIMER:</strong>
            These metrics reflect statistical classification performance across dataset-defined categories 
            on the Kaggle held-out test partition. The outputs represent seven-class mental-health text categorization 
            and must not be interpreted as validated clinical diagnoses or suicidal risk assessments. 
            Confidence values represent model probability over the dataset distribution, not real-world risk probability.
        </div>
        """,
        unsafe_allow_html=True,
    )
