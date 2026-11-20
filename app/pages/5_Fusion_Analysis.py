"""
MINDTRACE: Multimodal Fusion Analysis Module
Page 5: Text + Behavioral Binary Distress Fusion (Model C)
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
from src.models.fusion_model import (
    FIVE_SEED_BENCHMARK,
    FIVE_FOLD_CV_BENCHMARK,
)
from src.features.behavioral import FUSION_BEHAVIORAL_COLS

st.set_page_config(
    page_title="Fusion Analysis | MINDTRACE",
    page_icon="🧬",
    layout="wide",
)

load_custom_css()

@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()
status = platform.get_system_status()["fusion_model"]

render_header(
    title="Multimodal Fusion: Text + Behavioral Distress Detection",
    subtitle="Model C: 776-Dimensional Joint Representation (MentalBERT 768-d + Behavioral 8-d)",
)

render_disclaimer()

# Benchmark Overview Bar
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    render_metric_card("Input Representation", "776 Dimensions", "768 Text + 8 Behavioral")
with col_f2:
    render_metric_card("Fusion F1 Score", "0.694", "5-Seed Mean (vs Text 0.649)")
with col_f3:
    render_metric_card("CV AUROC Difference", "p = 0.2957", "Paired t-test (Not Sig)")
with col_f4:
    ckpt_text = "Verified on Disk" if status["checkpoint_exists"] else "Checkpoint Missing"
    render_metric_card("Fusion Checkpoint", ckpt_text, status["checkpoint_path"])

st.markdown("---")

tab_infer, tab_comp, tab_arch = st.tabs([
    "🧪 Run Multimodal Distress Inference",
    "⚖️ Controlled Model Comparison (A vs B vs C)",
    "📐 Architecture & Statistical Significance",
])

with tab_infer:
    st.markdown("#### Test Multi-Branch Distress Inference")
    st.caption("Input conversational dialogue and situation context to extract both modalities and evaluate joint distress classification.")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("##### 1. Situation Text (For MentalBERT Embedding)")
        sample_sit = "I have been struggling with debilitating insomnia and panic attacks after losing my job. I feel overwhelmed and unable to manage daily responsibilities."
        sit_input = st.text_area(
            "Situation Narrative / Seeker Context:",
            value=sample_sit,
            height=120,
        )

    with col_in2:
        st.markdown("##### 2. Dialogue Turns (For Behavioral Dynamics)")
        sample_diag = """User: I cannot sleep or concentrate on anything.
Assistant: I am deeply sorry. Have you spoken with a doctor?
User: no... terrified of going outside.
Assistant: Taking small steps is okay. I am right here listening.
User: thank you.
Assistant: We can take this one breath at a time."""
        diag_input = st.text_area(
            "Conversation Dialogue (User / Assistant):",
            value=sample_diag,
            height=120,
        )

    if st.button("Execute Multimodal Fusion Pipeline", key="btn_run_fusion"):
        if not diag_input.strip():
            st.warning("Please provide dialogue turns for behavioral feature extraction.")
        else:
            with st.spinner("Extracting MentalBERT embeddings, scaling behavioral features, and running fusion..."):
                res = platform.run_fusion_analysis(diag_input, situation_text=sit_input)

            if not res.get("success"):
                st.error(f"⚠️ {res.get('error') or res.get('fusion_prediction', {}).get('error', 'Fusion inference could not be completed.')}")
                st.markdown(
                    f"""
                    <div class="research-disclaimer" style="border-left-color: #F59E0B;">
                        <strong>Checkpoint Transparency Notice:</strong> The trained fusion model weights file <code>{status['checkpoint_path']}</code> was not found on disk.
                        In accordance with research integrity, randomly initialized weights are not used to generate mock predictions.
                        To execute live fusion inference, train or save <code>fusion_model.pt</code> into <code>models/fusion/</code>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show extracted features even when model checkpoint is missing!
                st.markdown("##### 🔍 Extracted Features from Input (Fully Computed):")
                c_ext1, c_ext2 = st.columns(2)
                with c_ext1:
                    st.write("**Extracted 8 Behavioral Features:**")
                    st.dataframe(pd.DataFrame([{
                        "Feature": col,
                        "Value": res["behavioral_features"].get(col, 0.0)
                    } for col in FUSION_BEHAVIORAL_COLS]), width='stretch')
                with c_ext2:
                    st.write("**Text Representation:**")
                    st.write(f"MentalBERT Target Text: *\"{res.get('target_text')}\"*")
                    st.write("Target Text Embedding: 768-dimensional float32 vector.")
            else:
                f_pred = res["fusion_prediction"]
                pred_label = f_pred["prediction"]
                conf = f_pred["confidence"]
                p_distress = f_pred["distress_probability"]

                st.success(f"**Predicted Distress Level:** `{pred_label.upper()}` (Confidence: {conf*100:.1f}%)")

                c_res1, c_res2 = st.columns([3, 2])
                with c_res1:
                    fig = plot_probability_distribution(
                        f_pred["probabilities"],
                        title="Binary Distress Probability Distribution",
                        highlight_label=pred_label,
                    )
                    st.plotly_chart(fig, width='stretch')

                with c_res2:
                    st.markdown("##### 🧬 Input Dimensions Verified")
                    st.markdown(
                        f"""
                        <div class="research-card">
                            <ul style="color:#CBD5E1; font-size:0.85rem; line-height:1.7; padding-left:1.2rem;">
                                <li><strong>Text Embedding:</strong> 768 dimensions (MentalBERT)</li>
                                <li><strong>Behavioral Features:</strong> 8 standardized numerical features</li>
                                <li><strong>Concatenated Representation:</strong> 776 dimensions</li>
                                <li><strong>Probability of High Distress:</strong> <span style="color:#F43F5E; font-weight:700;">{p_distress*100:.1f}%</span></li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with tab_comp:
    st.markdown("#### ⚖️ Controlled Experimental Comparison (Notebook 4)")
    st.markdown(
        """
        In <code>fusion_model.ipynb</code>, three models were trained and evaluated on identical holdout splits across 5 random seeds:
        - **Model A (Text Only):** MentalBERT 768-d embedding -> Linear(768, 128) -> Dropout(0.3) -> Linear(128, 2)
        - **Model B (Behavioral Only):** 8 features -> Linear(8, 32) -> Linear(32, 16) -> Linear(16, 2)
        - **Model C (Fusion - `FusionModelWeighted`):** 768 text + 8 behavioral -> 776 input dims
        """
    )

    rows = []
    for model_name, m in FIVE_SEED_BENCHMARK.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['acc_mean']*100:.1f}% ± {m['acc_std']*100:.1f}%",
            "Balanced Acc": f"{m['bal_acc_mean']*100:.1f}% ± {m['bal_acc_std']*100:.1f}%",
            "F1-Score": f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
            "Precision": f"{m['precision_mean']:.3f} ± {m['precision_std']:.3f}",
            "Recall": f"{m['recall_mean']:.3f} ± {m['recall_std']:.3f}",
            "AUROC": f"{m['auroc_mean']:.3f} ± {m['auroc_std']:.3f}",
        })
    st.table(pd.DataFrame(rows))

with tab_arch:
    st.markdown("#### 🔬 5-Fold Cross-Validation & Statistical Significance")

    cv_rows = []
    for model_name, m in FIVE_FOLD_CV_BENCHMARK.items():
        if model_name == "paired_ttest_C_vs_A":
            continue
        cv_rows.append({
            "Model": model_name,
            "CV Balanced Acc": f"{m['bal_acc_mean']:.3f} ± {m['bal_acc_std']:.3f}",
            "CV F1-Score": f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
            "CV AUROC": f"{m['auroc_mean']:.3f} ± {m['auroc_std']:.3f}",
        })
    st.table(pd.DataFrame(cv_rows))

    st.markdown(
        """
        <div class="research-disclaimer" style="border-left-color: #6366F1;">
            <h5 style="color:#818CF8; margin-top:0;">Paired t-test Statistical Evaluation</h5>
            A paired t-test was conducted between Model C (Fusion) and Model A (Text Only) AUROC scores across the 5 cross-validation folds:
            <ul style="margin-top:0.5rem; line-height:1.6;">
                <li><strong>Mean AUROC Difference (C - A):</strong> -0.0138</li>
                <li><strong>p-value:</strong> <code>0.2957</code></li>
                <li><strong>Conclusion:</strong> The difference is <em>not statistically significant</em> at α = 0.05.</li>
            </ul>
            <strong>Scientific Takeaway:</strong> While fusion achieved higher nominal F1 (0.710 vs 0.655), the AUROC metric demonstrates that combining behavioral interaction features with MentalBERT did not confer a statistically significant discrimination advantage on this ESConv dataset split.
        </div>
        """,
        unsafe_allow_html=True,
    )
