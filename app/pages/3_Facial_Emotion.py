"""
MINDTRACE: Facial Emotion Analysis Module
Page 3: FER2013 Custom CNN Facial Affect Classification Interface
"""

import streamlit as st
from PIL import Image
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.components.ui import (
    load_custom_css,
    render_header,
    render_metric_card,
    plot_probability_distribution,
)
from src.services.inference import MindTracePlatform
from src.models.facial_model import FER_BENCHMARK_METRICS

st.set_page_config(
    page_title="Facial Emotion | MINDTRACE",
    page_icon="🎭",
    layout="wide",
)

load_custom_css()

@st.cache_resource
def get_platform():
    return MindTracePlatform()

platform = get_platform()
status = platform.get_system_status()["facial_model"]

render_header(
    title="Facial Emotion Recognition (FER2013 CNN)",
    subtitle="Custom 3-Block Deep Convolutional Network for Static Facial Expression Classification",
)

# Mandatory Clinical Disclaimer
st.markdown(
    """
    <div class="clinical-warning">
        <strong>IMPORTANT SCIENTIFIC LIMITATION:</strong>
        Facial expression classification is <strong>not</strong> equivalent to assessing an individual's internal mental or emotional state.
        A person experiencing severe psychological distress may display a neutral or smiling expression, while momentary grimacing or frustration does not indicate a psychiatric disorder.
        This model strictly evaluates surface facial affect patterns trained on FER2013.
    </div>
    """,
    unsafe_allow_html=True,
)

# Benchmark Summary Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    render_metric_card("CNN Architecture", "3 Conv Blocks", "32 -> 64 -> 128 Filters")
with col_m2:
    render_metric_card("Input Resolution", "48 x 48 px", "3 Channels (RGB)")
with col_m3:
    render_metric_card("Test Accuracy", "58.79%", "Macro F1: 56.47% (FER2013)")
with col_m4:
    ckpt_text = "Verified on Disk" if status["checkpoint_exists"] else "Checkpoint Missing"
    render_metric_card("Model Checkpoint", ckpt_text, status["checkpoint_path"])

st.markdown("---")

tab_upload, tab_cam, tab_metrics = st.tabs([
    "📁 Image Upload Analysis",
    "📷 Live Webcam Capture",
    "📊 Model Architecture & Benchmark Details",
])

with tab_upload:
    st.markdown("#### Upload Facial Image")
    st.caption("Upload a front-facing facial photo (JPG, PNG, JPEG). The image will be resized to 48x48 and normalized.")

    uploaded_file = st.file_uploader(
        "Choose an image file:",
        type=["png", "jpg", "jpeg"],
        key="fer_file_uploader",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_pred = st.columns([1, 2])

        with col_img:
            st.image(image, caption="Uploaded Image", width='stretch')
            st.caption(f"Original dimensions: {image.size[0]} x {image.size[1]} pixels")

        with col_pred:
            if st.button("Classify Facial Emotion", key="btn_run_upload_fer"):
                with st.spinner("Processing image through FacialEmotionCNN..."):
                    res = platform.analyze_facial_image(image)

                if not res.get("success"):
                    st.error(f"⚠️ {res.get('error')}")
                    st.markdown(
                        f"""
                        <div class="research-disclaimer" style="border-left-color: #F59E0B;">
                            <strong>Model Checkpoint Notice:</strong> Checkpoint file <code>{status['checkpoint_path']}</code> was not found on disk.
                            In accordance with research integrity, no dummy or random predictions are displayed.
                            Please place <code>fer_model.pt</code> into <code>models/facial/</code> to execute live image classification.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    dom = res["dominant_emotion"]
                    conf = res["confidence"]
                    st.success(f"**Dominant Expression:** `{dom.upper()}` (Probability: {conf*100:.1f}%)")

                    fig = plot_probability_distribution(
                        res["probabilities"],
                        title="FER2013 7-Emotion Softmax Distribution",
                        highlight_label=dom,
                    )
                    st.plotly_chart(fig, width='stretch')

with tab_cam:
    st.markdown("#### Optional Live Webcam Analysis")
    st.caption("Capture a still frame from your webcam to evaluate real-time affect expression.")

    camera_image = st.camera_input("Capture facial frame for emotion classification", key="webcam_cam_input")

    if camera_image is not None:
        cam_pil = Image.open(camera_image)
        c_cam1, c_cam2 = st.columns([1, 2])
        with c_cam1:
            st.image(cam_pil, caption="Captured Frame", width='stretch')
        with c_cam2:
            if st.button("Classify Webcam Frame", key="btn_run_webcam_fer"):
                with st.spinner("Analyzing captured frame..."):
                    res_cam = platform.analyze_facial_image(cam_pil)

                if not res_cam.get("success"):
                    st.error(f"⚠️ {res_cam.get('error')}")
                    st.info(f"Trained checkpoint required at: `{status['checkpoint_path']}`.")
                else:
                    dom_cam = res_cam["dominant_emotion"]
                    st.success(f"**Dominant Expression:** `{dom_cam.upper()}` ({res_cam['confidence']*100:.1f}%)")
                    fig_cam = plot_probability_distribution(
                        res_cam["probabilities"],
                        title="Webcam Frame Emotion Probabilities",
                        highlight_label=dom_cam,
                    )
                    st.plotly_chart(fig_cam, width='stretch')

with tab_metrics:
    st.markdown("#### 🔬 Published FER2013 Benchmark Metrics (Notebook 3)")
    st.markdown(
        """
        The custom PyTorch CNN was trained for 10 epochs on the FER2013 dataset. 
        Test evaluation across 7,178 test images yielded the following per-class metrics:
        """
    )

    f1_df = pd.DataFrame([
        {"Emotion": k.capitalize(), "Test F1-Score": f"{v*100:.1f}%"}
        for k, v in FER_BENCHMARK_METRICS["class_f1"].items()
    ])
    st.table(f1_df)

    st.markdown(
        """
        <div class="research-card">
            <h5 style="color:#818CF8; margin-top:0;">Architecture Specifications</h5>
            <ul style="color:#CBD5E1; font-size:0.85rem; line-height:1.6;">
                <li><strong>Conv Block 1:</strong> Conv2D(3 -> 32, k=3, p=1) + ReLU + MaxPool2D(2) [48x48 -> 24x24]</li>
                <li><strong>Conv Block 2:</strong> Conv2D(32 -> 64, k=3, p=1) + ReLU + MaxPool2D(2) [24x24 -> 12x12]</li>
                <li><strong>Conv Block 3:</strong> Conv2D(64 -> 128, k=3, p=1) + ReLU + MaxPool2D(2) [12x12 -> 6x6]</li>
                <li><strong>Classifier:</strong> Flatten -> Linear(4608 -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> 7)</li>
                <li><strong>Total Parameters:</strong> 1,274,951</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
