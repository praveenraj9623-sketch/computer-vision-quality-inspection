# Computer Vision Product Quality Inspection System

Live App: https://computer-vision-quality-inspection-qkv9autqvpnjyoz5tzn8n3.streamlit.app/

A computer vision-based quality inspection system that classifies casting product images as defective or non-defective using deep learning. The project uses TensorFlow/Keras, MobileNetV2 transfer learning, model evaluation metrics, Grad-CAM explainability, and a Streamlit web application for real-time image prediction.

---

## Business Problem

In manufacturing, visual quality inspection is often performed manually. This can be time-consuming and may lead to inconsistent inspection results due to human fatigue or visual similarity between defective and non-defective products.

This project helps automate the initial inspection process by classifying casting product images into:

- Defective
- Non-defective / OK

The system is designed as a decision-support tool for manufacturing quality teams.

---

## Live Demo

Streamlit Application:

https://computer-vision-quality-inspection-qkv9autqvpnjyoz5tzn8n3.streamlit.app/

---

## Key Features

- Upload casting product images
- Predict whether the product is defective or non-defective
- Show model confidence score
- Display probability of defective and OK classes
- Generate Grad-CAM visual explanation
- Show model details and class information
- Display evaluation metrics
- Display confusion matrix
- Streamlit-based interactive UI

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | CNN, MobileNetV2 Transfer Learning |
| Computer Vision | Image preprocessing, Grad-CAM |
| Data Handling | NumPy, Pandas, Pillow |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## Dataset

Dataset used:

**Casting Product Image Data for Quality Inspection**

Kaggle dataset link:

https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

The dataset contains casting product images divided into two classes:

```text
def_front  → Defective product images
ok_front   → Non-defective product images
```

Expected dataset folder structure:

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

Note: The dataset is not included in this repository because of file size. Download it from Kaggle and place it inside the `data/` folder.

---

## Project Structure

```text
computer-vision-quality-inspection/
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
├── models/
│   └── quality_inspection_model.keras
│
├── outputs/
│   ├── evaluation_metrics.json
│   ├── evaluation_classification_report.csv
│   └── confusion_matrix.png
│
├── screenshots/
└── data/
```

---

## Model Used

The main model used in this project is **MobileNetV2 Transfer Learning**.

MobileNetV2 was selected because:

- It is lightweight compared to large CNN architectures
- It is suitable for image classification tasks
- It uses pretrained ImageNet weights
- It can learn strong visual features with less training time
- It is practical for deployment in a Streamlit application

The project also includes a baseline CNN option for comparison and experimentation.

---

## Model Training

To train the model locally:

```bash
python train.py --dataset_dir data/casting_data/casting_data --model_type mobilenet --epochs 10
```

To train the baseline CNN model:

```bash
python train.py --dataset_dir data/casting_data/casting_data --model_type cnn --epochs 10
```

The trained model is saved as:

```text
models/quality_inspection_model.keras
```

---

## Model Evaluation

To evaluate the trained model:

```bash
python evaluate.py --dataset_dir data/casting_data/casting_data --model_path models/quality_inspection_model.keras
```

Evaluation outputs are saved inside the `outputs/` folder.

---

## Model Performance

The trained MobileNetV2 model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 0.9888 |
| Precision | 0.9885 |
| Recall | 0.9809 |
| F1-score | 0.9847 |
| ROC-AUC | 0.9991 |

These results show that the model performs strongly in separating defective and non-defective casting product images.

---

## Evaluation Metrics Explained

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Out of predicted defective/OK cases, how many were correct |
| Recall | How many actual cases were correctly identified |
| F1-score | Balanced score between precision and recall |
| ROC-AUC | Measures how well the model separates both classes |

---

## Grad-CAM Explainability

Grad-CAM is used to visually explain the model prediction.

It highlights the image regions that influenced the model’s decision. This helps users understand whether the model is focusing on meaningful product areas before making a prediction.

This is important because computer vision systems should not only make predictions but also provide some level of interpretability.

---

## Streamlit Application Pages

### 1. Overview

Explains the business problem, workflow, use case, model type, and application value.

### 2. Image Prediction

Allows users to upload a casting product image and receive:

- Prediction label
- Confidence score
- Probability defective
- Probability OK
- Recommended action
- Grad-CAM heatmap

### 3. Model Details

Displays:

- Input image size
- Number of output classes
- Class labels
- Model architecture details
- Training command
- Evaluation command
- Model file status

### 4. Evaluation Results

Displays:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- Metric interpretation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/praveenraj9623-sketch/computer-vision-quality-inspection.git
cd computer-vision-quality-inspection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Mac / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## How to Use the App

1. Open the Streamlit application
2. Go to the **Image Prediction** page
3. Upload a casting product image
4. View the prediction result
5. Check confidence score and probability values
6. View the Grad-CAM explanation
7. Use the result as decision support for quality inspection

---

## Deployment

This project is deployed using Streamlit Cloud.

Live app:

https://computer-vision-quality-inspection-qkv9autqvpnjyoz5tzn8n3.streamlit.app/

Deployment requirements:

```text
streamlit
tensorflow
numpy
pandas
pillow
scikit-learn
matplotlib
plotly
opencv-python-headless
```

The trained model file must be available in:

```text
models/quality_inspection_model.keras
```

---

## Skills Demonstrated

- Computer Vision
- Deep Learning
- TensorFlow
- Keras
- CNN
- MobileNetV2 Transfer Learning
- Binary Image Classification
- Image Preprocessing
- Data Augmentation
- Model Training
- Model Evaluation
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Grad-CAM Explainability
- Streamlit App Development
- Streamlit Cloud Deployment
- Manufacturing Quality Inspection Use Case

---

## Limitations

- The model works best on images similar to the training dataset.
- It is intended as a decision-support tool, not a complete replacement for human inspection.
- Performance may reduce on images with different lighting, angles, backgrounds, or product types.
- Real production deployment would require camera integration, threshold calibration, monitoring, retraining, and human review workflow.

---

## Future Improvements

- Add more product categories
- Add real-time camera input
- Add batch image prediction
- Add model monitoring dashboard
- Add threshold tuning for quality control teams
- Add defect localization with object detection or segmentation
- Deploy using Docker or cloud infrastructure
- Add user authentication and prediction history

---

## Resume Description

Built a computer vision quality inspection system using TensorFlow/Keras and MobileNetV2 transfer learning to classify casting product images as defective or non-defective. Implemented image preprocessing, data augmentation, model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC, Grad-CAM explainability, and a Streamlit web application for real-time prediction and visual model interpretation.

---

## Author

**Praveen Raj A**

LinkedIn:  
https://www.linkedin.com/in/praveen-raj-a-b05abb2a3/

GitHub:  
https://github.com/praveenraj9623-sketch
