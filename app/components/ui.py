"""
Reusable UI and visualization components for MINDTRACE.
Provides dark-themed research cards, badges, and interactive Plotly charts.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def load_custom_css():
    """Inject custom dark theme CSS."""
    with open("app/styles/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "Multimodal Mental Health Research Platform"):
    """Render top hero header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="brand-title">{title}</div>
            <div class="brand-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Render mandatory ethics and research disclaimer."""
    st.markdown(
        """
        <div class="research-disclaimer">
            <strong>RESEARCH PROTOTYPE NOTICE:</strong> This application is a multimodal research demonstration developed for academic investigation. 
            It is <strong>not</strong> a clinical diagnostic tool and does <strong>not</strong> provide psychiatric assessments or medical evaluations. 
            Facial emotion and conversational metrics do not constitute psychiatric diagnoses. If you or someone you know is in distress, please contact professional healthcare providers or crisis hotlines.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, subtext: Optional[str] = None):
    """Render a styled metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status_type: str, text: str) -> str:
    """Return HTML for a status pill."""
    type_map = {
        "implemented": "pill-green",
        "ready": "pill-green",
        "experimental": "pill-amber",
        "missing": "pill-red",
        "proposed": "pill-purple",
        "active": "pill-blue",
    }
    css_class = type_map.get(status_type.lower(), "pill-blue")
    return f'<span class="status-pill {css_class}">{text}</span>'


def plot_probability_distribution(
    prob_dict: Dict[str, float],
    title: str = "Class Probability Distribution",
    highlight_label: Optional[str] = None,
    color_scale: str = "Viridis",
) -> go.Figure:
    """Create an interactive, dark-themed horizontal or vertical bar chart for class probabilities."""
    labels = list(prob_dict.keys())
    scores = [float(prob_dict[k]) for k in labels]

    colors = []
    for l in labels:
        if l == highlight_label:
            colors.append("#6366F1")  # vibrant indigo
        else:
            colors.append("#1E293B")  # subtle slate

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, "#1E293B"], [0.5, "#3B82F6"], [1.0, "#6366F1"]],
                line=dict(color="#818CF8", width=1),
            ),
            text=[f"{s*100:.1f}%" for s in scores],
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=12),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F8FAFC", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, max(max(scores) * 1.25, 0.1)],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickformat=".0%",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            tickfont=dict(color="#E2E8F0", size=12),
            autorange="reversed",
        ),
        margin=dict(l=10, r=30, t=40, b=20),
        height=320,
    )
    return fig


def plot_turn_length_trend(turns_data: List[Dict[str, Any]]) -> go.Figure:
    """Plot message word count over conversation turn sequence."""
    if not turns_data:
        return go.Figure()

    df = pd.DataFrame(turns_data)
    user_df = df[df["speaker"] == "user"]
    asst_df = df[df["speaker"] == "assistant"]

    fig = go.Figure()

    if not user_df.empty:
        fig.add_trace(
            go.Scatter(
                x=user_df["turn_number"],
                y=user_df["word_count"],
                mode="lines+markers",
                name="User Turn Words",
                line=dict(color="#06B6D4", width=2.5),
                marker=dict(size=8, color="#06B6D4"),
            )
        )

    if not asst_df.empty:
        fig.add_trace(
            go.Scatter(
                x=asst_df["turn_number"],
                y=asst_df["word_count"],
                mode="lines+markers",
                name="Assistant Words",
                line=dict(color="#8B5CF6", width=1.5, dash="dot"),
                marker=dict(size=6, color="#8B5CF6"),
            )
        )

    fig.update_layout(
        title=dict(text="Message Length Dynamics across Turns", font=dict(color="#F8FAFC", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Turn Sequence", font=dict(color="#94A3B8", size=12)),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8"),
            dtick=1,
        ),
        yaxis=dict(
            title=dict(text="Word Count", font=dict(color="#94A3B8", size=12)),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8"),
        ),
        legend=dict(font=dict(color="#E2E8F0")),
        margin=dict(l=10, r=20, t=40, b=20),
        height=320,
    )
    return fig


def render_benchmark_table(benchmark_data: Dict[str, Any], title: str = "Published Experimental Benchmark"):
    """Render a styled benchmark card with documented notebook metrics."""
    st.markdown(f"#### 📊 {title}")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:0.75rem;">
            Reported from controlled experiments in the research notebooks.
        </div>
        """,
        unsafe_allow_html=True,
    )
