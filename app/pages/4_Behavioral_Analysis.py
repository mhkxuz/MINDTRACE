"""
MINDTRACE: Behavioral Interaction Analysis Module
Page 4: Conversational Dynamics, Turn Metrics, and Withdrawal Pattern Profiling
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_metric_card,
    plot_turn_length_trend,
)
from src.services.inference import MindTracePlatform
from src.features.behavioral import FUSION_BEHAVIORAL_COLS

st.set_page_config(
    page_title="Behavioral Analysis | MINDTRACE",
    page_icon="📈",
    layout="wide",
)

load_custom_css()

@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()

render_header(
    title="Behavioral Interaction & Dialogue Dynamics",
    subtitle="Extraction of 13 Turn-Level Behavioral Metrics & Communication Withdrawal Patterns",
)

# Mandatory scientific disclaimer
st.markdown(
    """
    <div class="research-disclaimer">
        <strong>METHODOLOGICAL NOTICE:</strong>
        The behavioral metrics extracted below are computed strictly from static conversation turn structure and message word lengths.
        They reflect <strong>interaction dynamics</strong> (turn counts, length trajectory, assistant carrying ratio, and unanswered closing turns).
        They do <strong>not</strong> measure millisecond keystroke dynamics, typing latency, or physiological biomarkers, and do <strong>not</strong> represent clinical indicators or psychiatric diagnoses.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_transcript, tab_csv, tab_defs = st.tabs([
    "💬 Interactive Dialogue Transcript",
    "📄 Batch CSV Dataset Analysis",
    "📖 Mathematical Definitions & Formulas",
])

with tab_transcript:
    st.markdown("#### Analyze Conversational Transcript")

    sample_choices = {
        "Sample A: Gradual Withdrawal (High Dropout Gap)": "data/samples/sample_dialogue_high_distress.txt",
        "Sample B: Sustained Engagement (Normal Ending)": "data/samples/sample_dialogue_low_distress.txt",
        "Manual Input": None,
    }

    selected_choice = st.selectbox("Select conversation sample or enter your own:", list(sample_choices.keys()))

    if sample_choices[selected_choice] and os.path.exists(sample_choices[selected_choice]):
        with open(sample_choices[selected_choice]) as f:
            default_transcript = f.read()
    else:
        default_transcript = """User: Hello, I have had a difficult couple of days.
Assistant: I am here with you. Would you like to share what happened?
User: Just feeling completely exhausted and overwhelmed with expectations.
Assistant: That sounds very heavy. Are you getting enough sleep?
User: barely.
Assistant: Chronic sleep deprivation makes everything feel magnified. Please be gentle with yourself.
User: trying.
Assistant: Even small moments of rest matter. Take all the time you need.
Assistant: We can pause here if you need a break."""

    transcript_input = st.text_area(
        "Conversation Turns (User / Assistant):",
        value=default_transcript,
        height=180,
    )

    if st.button("Extract Behavioral Features", key="btn_run_behav_transcript"):
        if not transcript_input.strip():
            st.warning("Please provide a non-empty conversation transcript.")
        else:
            with st.spinner("Extracting behavioral features..."):
                res = platform.analyze_behavioral_transcript(transcript_input)

            feats = res["features"]
            turns_data = feats.get("turns_data", [])

            # High-level metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Turns", feats["total_turn_count"], f"User: {feats['user_turn_count']} | Asst: {feats['assistant_turn_count']}")
            with c2:
                render_metric_card("Engagement Ratio", f"{feats['engagement_ratio']:.2f}", "User turns / Total turns")
            with c3:
                render_metric_card("Message Length Slope", f"{feats['msg_length_slope']:+.2f}", "Regression over turn sequence")
            with c4:
                render_metric_card("Dropout Gap", feats["dropout_gap"], f"Flag: {feats['dropout_flag']} ({'WITHDRAWAL' if feats['dropout_flag'] == 1 else 'NORMAL'})")

            st.markdown("---")

            # Chart and Withdrawal Analysis
            col_chart, col_withd = st.columns([3, 2])

            with col_chart:
                fig = plot_turn_length_trend(turns_data)
                st.plotly_chart(fig, width='stretch')

            with col_withd:
                st.markdown("##### 🚨 Withdrawal Pattern Profiling")

                is_withdrawing = feats["msg_length_slope"] < -1.0
                is_dropped_out = feats["dropout_flag"] == 1
                ended_user = feats["ended_by_user"] == 1

                st.markdown(
                    f"""
                    <div class="research-card">
                        <div style="margin-bottom:0.8rem;">
                            <strong>Message Length Trajectory:</strong><br>
                            <span style="color:{'#F43F5E' if is_withdrawing else '#34D399'}; font-weight:600;">
                                {'Shrinking rapidly (Negative Slope)' if is_withdrawing else 'Stable / Expanding'}
                            </span>
                            <div style="font-size:0.75rem; color:#64748B;">Slope = {feats['msg_length_slope']:.3f} words/turn</div>
                        </div>
                        <div style="margin-bottom:0.8rem;">
                            <strong>Dropout Signal:</strong><br>
                            <span style="color:{'#F43F5E' if is_dropped_out else '#34D399'}; font-weight:600;">
                                {'Dropout Flag = 1 (Gap > 2 turns)' if is_dropped_out else 'No Dropout (Gap ≤ 2)'}
                            </span>
                            <div style="font-size:0.75rem; color:#64748B;">{feats['dropout_gap']} assistant turns unanswered at end.</div>
                        </div>
                        <div>
                            <strong>Final Speaker:</strong><br>
                            <span style="color:#38BDF8; font-weight:600;">
                                {feats['last_speaker'].upper()} ({'User Closed' if ended_user else 'Assistant Left Hanging'})
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 8-Feature Vector Table for Fusion
            st.markdown("##### 🔢 Standardized 8-Feature Input Vector for Multimodal Fusion")
            vector_df = pd.DataFrame([{
                "Feature Name": col,
                "Value": feats.get(col, 0.0),
                "Fusion Role": "Branch 2 Numerical Input"
            } for col in FUSION_BEHAVIORAL_COLS])
            st.table(vector_df)

with tab_csv:
    st.markdown("#### Batch CSV Dialogue Dataset Analysis")
    st.caption("Upload an ESConv-style CSV containing columns: `conversation_id`, `turn_number`, `speaker`, `message`.")

    sample_csv_path = "data/samples/sample_esconv_format.csv"
    if os.path.exists(sample_csv_path):
        st.download_button(
            "Download Sample CSV Format",
            data=open(sample_csv_path).read(),
            file_name="sample_esconv_format.csv",
            mime="text/csv",
        )

    uploaded_csv = st.file_uploader("Upload CSV file:", type=["csv"], key="csv_behav_uploader")

    if uploaded_csv is not None:
        try:
            df_in = pd.read_csv(uploaded_csv)
            st.write(f"Loaded CSV with {len(df_in)} rows.")
            st.dataframe(df_in.head(6), width='stretch')

            if st.button("Process All Conversations in CSV"):
                with st.spinner("Extracting behavioral metrics across all conversations..."):
                    summary_df, _ = platform.analyze_behavioral_dataframe(df_in)

                st.success(f"Successfully processed {len(summary_df)} conversations!")
                st.dataframe(summary_df, width='stretch')

                # Summary statistics
                st.markdown("##### Aggregate Statistical Summary")
                st.dataframe(summary_df[FUSION_BEHAVIORAL_COLS].describe(), width='stretch')
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

with tab_defs:
    st.markdown("#### 📖 Mathematical Definitions & Formulas (Notebook 1)")
    st.markdown(
        """
        The following 13 metrics are computed from turn sequences as formulated in <code>data_preparation+behavioral_features.ipynb</code>:
        """
    )

    formulas = [
        {"Metric": "user_turn_count", "Formula": "Count(turns where speaker == 'user')", "Interpretation": "Volume of user contributions."},
        {"Metric": "assistant_turn_count", "Formula": "Count(turns where speaker == 'assistant')", "Interpretation": "Volume of supporter contributions."},
        {"Metric": "total_turn_count", "Formula": "user_turn_count + assistant_turn_count", "Interpretation": "Total dialogue length."},
        {"Metric": "engagement_ratio", "Formula": "user_turn_count / total_turn_count", "Interpretation": "Degree of user participation (0 to 1)."},
        {"Metric": "avg_user_msg_length", "Formula": "Mean(word_count(user_messages))", "Interpretation": "Average verbal length of seeker."},
        {"Metric": "avg_assistant_msg_length", "Formula": "Mean(word_count(assistant_messages))", "Interpretation": "Supporter message depth."},
        {"Metric": "user_msg_length_std", "Formula": "StdDev(word_count(user_messages))", "Interpretation": "Consistency vs erratic verbal length."},
        {"Metric": "last_user_turn", "Formula": "Max(turn_number where speaker == 'user')", "Interpretation": "Final point of user contact."},
        {"Metric": "last_speaker", "Formula": "Speaker of last turn in dialogue", "Interpretation": "Identifies who closed conversation."},
        {"Metric": "ended_by_user", "Formula": "1 if last_speaker == 'user' else 0", "Interpretation": "1 = normal closure, 0 = assistant hanging."},
        {"Metric": "msg_length_slope", "Formula": "linregress(turn_seq, word_counts).slope", "Interpretation": "Negative slope signifies shrinking communication."},
        {"Metric": "dropout_gap", "Formula": "total_turn_count - last_user_turn", "Interpretation": "Unanswered assistant turns at dialogue conclusion."},
        {"Metric": "dropout_flag", "Formula": "1 if dropout_gap > 2 else 0", "Interpretation": "Binary flag for likely user disengagement/dropout."},
    ]
    st.table(pd.DataFrame(formulas))
