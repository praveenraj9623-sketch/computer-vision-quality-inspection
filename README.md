# Computer Vision Product Quality Inspection System

This project detects defective and non-defective casting product images using deep learning.

It is designed for a manufacturing quality inspection use case and demonstrates a complete computer vision workflow using TensorFlow, Keras, CNN, transfer learning, model evaluation, Grad-CAM explainability, and Streamlit deployment.

---

## Business Problem

In manufacturing, manual quality inspection can be slow and inconsistent. Defective casting products may be missed due to human fatigue or visual similarity between good and defective parts.

This project helps automate initial defect screening by classifying product images as:

- Defective
- Non-defective / OK

The model output can support quality teams by flagging products that need human review.

---

## Dataset

Recommended dataset:

**Casting Product Image Data for Quality Inspection**

Kaggle:
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

Expected classes:

- `def_front` = defective product image
- `ok_front` = non-defective product image

Expected folder structure:

```text
data/
└── casting_data/
    └── casting_data/
        ├── train/
        │   ├── def_front/
        │   └── ok_front/
        └── test/
            ├── def_front/
            └── ok_front/
```

---

## Features

- Image upload and real-time defect prediction
- CNN model training
- MobileNetV2 transfer learning
- Data augmentation
- Model evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix
- Grad-CAM explainability
- Business recommendation based on prediction confidence
- Streamlit dashboard
- Safe fallback messages if model or dataset is missing

---

## Project Structure

```text
cv_quality_inspection_system/
│
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── gradcam.py
│   ├── model_builder.py
│   ├── predictor.py
│   ├── reporting.py
│   └── visualization.py
│
├── data/
├── models/
├── outputs/
├── screenshots/
└── notebooks/
```

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
python -m streamlit run app.py
```

If no trained model is available, the app will still open and show setup guidance.

---

## Train Model

After downloading the Kaggle dataset and placing it inside the `data/` folder:

```bash
python train.py --dataset_dir data/casting_data/casting_data --model_type mobilenet --epochs 10
```

For a lighter baseline CNN:

```bash
python train.py --dataset_dir data/casting_data/casting_data --model_type cnn --epochs 10
```

The trained model will be saved to:

```text
models/quality_inspection_model.keras
```

---

## Evaluate Model

```bash
python evaluate.py --dataset_dir data/casting_data/casting_data --model_path models/quality_inspection_model.keras
```

Evaluation outputs include Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.

---

## JD Skill Coverage

This project covers:

- Computer Vision
- Deep Learning
- CNN
- Neural Networks
- TensorFlow
- Keras
- Image Classification
- Model Training
- Model Testing
- Precision, Recall, F1-score, ROC-AUC
- Explainable AI using Grad-CAM
- Streamlit AI application
- Manufacturing / industrial quality inspection use case

---

## Resume Description

Built a computer vision product quality inspection system to classify casting product images as defective or non-defective. Implemented image preprocessing, data augmentation, CNN baseline model, MobileNetV2 transfer learning, model evaluation using accuracy, precision, recall, F1-score and ROC-AUC, Grad-CAM explainability, and a Streamlit prediction dashboard for real-time quality inspection support.

---

## Limitations

- The project works best with clear product images similar to the training dataset.
- The model is a decision-support tool, not a complete replacement for human quality inspectors.
- Real production deployment would need camera integration, monitoring, retraining pipeline, threshold calibration, and human review workflow.


---

## Screenshots to Add

After training the model locally, add screenshots to the `screenshots/` folder:

- `prediction_result.png`
- `gradcam_heatmap.png`
- `model_setup.png`
- `evaluation_metrics.png`

These screenshots make the GitHub project easier to understand during recruiter screening.
"# computer-vision-quality-inspection" 
