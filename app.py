import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import DEFAULT_MODEL_PATH
from src.predictor import load_model_safe, predict_image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap


st.set_page_config(
    page_title="Computer Vision Quality Inspection",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #475569;
        font-size: 16px;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #0F172A;
        margin-top: 10px;
    }
    .info-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Computer Vision Product Quality Inspection System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Deep learning system for detecting defective and non-defective casting product images.</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Configuration")

    model_path = st.text_input(
        "Model Path",
        value=str(DEFAULT_MODEL_PATH),
    )

    threshold = st.slider(
        "OK Classification Threshold",
        0.10,
        0.90,
        0.50,
        0.05,
    )

    show_gradcam = st.toggle(
        "Show Grad-CAM Explanation",
        value=True,
    )

    st.caption(
        "Use the trained model file from the models folder for image prediction."
    )


model, model_error = load_model_safe(model_path)

pages = [
    "Overview",
    "Image Prediction",
    "Model Details",
    "Evaluation Results",
]

page = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


if page == "Overview":
    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        This application supports automated visual quality inspection for casting products.
        It uses a deep learning model to classify uploaded product images as defective or non-defective.
        The system is designed as a decision-support tool for manufacturing quality teams.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Use Case", "Quality Inspection")
    col2.metric("Task", "Binary Classification")
    col3.metric("Model Type", "CNN / MobileNetV2")
    col4.metric("Explainability", "Grad-CAM")

    st.subheader("Application Workflow")

    st.code(
        """
Image Dataset
↓
Image Preprocessing
↓
Data Augmentation
↓
CNN / MobileNetV2 Training
↓
Model Evaluation
↓
Image Upload
↓
Defect Prediction
↓
Grad-CAM Visual Explanation
        """
    )

    st.subheader("Business Value")

    st.write(
        """
        - Helps identify defective casting products from images.
        - Reduces manual inspection effort for initial screening.
        - Supports faster quality control decision-making.
        - Provides prediction confidence for each uploaded image.
        - Uses Grad-CAM to highlight image regions influencing the model prediction.
        """
    )


elif page == "Image Prediction":
    st.markdown(
        '<div class="section-title">Image Prediction</div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.warning("Trained model not found or could not be loaded.")

        st.info(
            "Train the model first using the command below, then return to this page."
        )

        st.code(
            "python train.py --dataset_dir data/casting_data/casting_data --model_type mobilenet --epochs 10"
        )

        if model_error:
            st.code(model_error)

    else:
        uploaded_image = st.file_uploader(
            "Upload a casting product image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)

            result = predict_image(
                model=model,
                image=image,
                threshold=threshold,
            )

            with right_col:
                st.subheader("Prediction Result")

                if result["predicted_class"] == "def_front":
                    st.error(f"Prediction: {result['prediction_label']}")
                    recommendation = (
                        "Recommended action: Send this product for manual inspection "
                        "or rejection review."
                    )
                else:
                    st.success(f"Prediction: {result['prediction_label']}")
                    recommendation = (
                        "Recommended action: Product appears acceptable based on "
                        "current model confidence."
                    )

                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                st.metric(
                    "Probability Defective",
                    f"{result['probability_defective'] * 100:.2f}%",
                )
                st.metric(
                    "Probability OK",
                    f"{result['probability_ok'] * 100:.2f}%",
                )

                st.info(recommendation)

            if show_gradcam:
                st.subheader("Grad-CAM Visual Explanation")

                heatmap, error = make_gradcam_heatmap(
                    model=model,
                    image=image,
                )

                if heatmap is None:
                    st.warning(error)
                else:
                    overlay = overlay_heatmap(image, heatmap)
                    st.image(
                        overlay,
                        caption="Grad-CAM heatmap showing model attention area",
                        use_container_width=True,
                    )


elif page == "Model Details":
    st.markdown(
        '<div class="section-title">Model Details</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        This section explains the model configuration used for quality inspection.
        It is useful for validating the system before using it for prediction.
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Input Image Size", "224 × 224")
    c2.metric("Output Classes", "2")
    c3.metric("Prediction Type", "Binary")

    st.subheader("Classes")

    class_df = pd.DataFrame(
        [
            {
                "Class Folder": "def_front",
                "Display Label": "Defective",
                "Meaning": "Casting product image contains visible defect",
            },
            {
                "Class Folder": "ok_front",
                "Display Label": "Non-defective / OK",
                "Meaning": "Casting product image appears acceptable",
            },
        ]
    )

    st.dataframe(class_df, use_container_width=True)

    st.subheader("Model Architecture")

    st.write(
        """
        The project supports two model options:
        """
    )

    architecture_df = pd.DataFrame(
        [
            {
                "Model": "Baseline CNN",
                "Purpose": "Simple custom convolutional neural network for image classification",
            },
            {
                "Model": "MobileNetV2 Transfer Learning",
                "Purpose": "Pretrained ImageNet model adapted for casting defect classification",
            },
        ]
    )

    st.dataframe(architecture_df, use_container_width=True)

    st.subheader("Training Command")

    st.code(
        "python train.py --dataset_dir data/casting_data/casting_data --model_type mobilenet --epochs 10"
    )

    st.subheader("Evaluation Command")

    st.code(
        "python evaluate.py --dataset_dir data/casting_data/casting_data --model_path models/quality_inspection_model.keras"
    )

    st.subheader("Model File Status")

    model_file = Path(model_path)

    if model_file.exists():
        st.success(f"Model file found: {model_file}")
    else:
        st.warning(f"Model file not found: {model_file}")


elif page == "Evaluation Results":
    st.markdown(
        '<div class="section-title">Evaluation Results</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        This section displays model evaluation results generated after running `evaluate.py`.
        """
    )

    metrics_path = Path("outputs/evaluation_metrics.json")
    report_path = Path("outputs/evaluation_classification_report.csv")
    confusion_matrix_path = Path("outputs/confusion_matrix.png")

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        st.subheader("Model Metrics")

        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df, use_container_width=True)

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        m2.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        m3.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
        m4.metric("F1-score", f"{metrics.get('F1-score', 0):.4f}")
        m5.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}")

    else:
        st.warning("Evaluation metrics file not found.")
        st.code(
            "python evaluate.py --dataset_dir data/casting_data/casting_data --model_path models/quality_inspection_model.keras"
        )

    if report_path.exists():
        st.subheader("Classification Report")
        report_df = pd.read_csv(report_path)
        st.dataframe(report_df, use_container_width=True)

    if confusion_matrix_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(confusion_matrix_path), use_container_width=False)

    st.subheader("Interpretation")

    st.write(
        """
        The evaluation metrics help measure how well the model separates defective and non-defective product images.
        Precision is important when we want to avoid incorrectly marking good products as defective.
        Recall is important when we want to catch as many defective products as possible.
        ROC-AUC shows how well the model separates both classes across different thresholds.
        """
    )