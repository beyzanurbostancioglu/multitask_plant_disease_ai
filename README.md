# 🌿 Multi-Task Plant Disease Detection & Severity Analysis

This project is a multi-task deep learning and computer vision framework designed to simultaneously detect **Plant Species**, **Disease Type**, and **Disease Severity** directly from leaf images.

Built on top of the AI Challenger 2018 dataset, the system combines deep learning predictions with HSV color space filtering and GrabCut segmentation to provide a robust hybrid severity analysis.

---

## 📊 Model Performance & Metrics

Evaluated on the validation dataset, the model achieved high accuracy metrics across all targets:

* **Plant Species Accuracy (`acc_plant`):** 99.46%
* **Disease Diagnosis Accuracy (`acc_disease`):** 93.68%
* **Severity Detection Accuracy (`acc_severity`):** 86.03%
* **PDSA (Plant + Disease + Severity Exact Match):** 81.21%
* **Macro F1-Score (Disease):** 92.81%

---

## 🛠️ Technical Architecture & Methods

- **Model Backbone:** Pretrained `EfficientNet-B0` (PyTorch & `timm`)
- **Multi-Task Architecture:** Simultaneous 3-head prediction in a single forward pass
- **Loss Functions:** Focal Loss & Cross Entropy Loss
- **Computer Vision & Segmentation:** HSV Color Space Filtering, GrabCut Segmentation Algorithm
- **Web Interface:** Gradio Web UI (`app.py`)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone (https://github.com/beyzanurbostancioglu/multitask_plant_disease_ai.git)
cd multitask_plant_disease_ai
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Note: Please ensure PyTorch is installed with appropriate CUDA support for GPU acceleration.

### 4. Launch the Web Interface
```bash
python app.py
```
Once started, access the application locally at http://127.0.0.1:7860.

---

## 🖥️ Web UI & Features
Upon uploading a leaf image:

Calculates the predicted Plant Type, Disease, and Severity Level with confidence scores.

Lists the top 3 most probable diseases.

Highlights infected regions with a red contour mask using computer vision algorithms and provides pixel-based lesion area calculations.


