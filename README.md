<div align="center">

# 🦷 AI Dental X-Ray Analyzer

### Explainable AI-Powered Panoramic Dental Radiograph Analysis

*Automated multi-condition detection, GradCAM interpretability, FDI charting, PDF reporting & treatment cost estimation — powered by YOLOv9*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv9](https://img.shields.io/badge/YOLOv9-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#-license)

[![Accuracy](https://img.shields.io/badge/Accuracy-97.3%25-brightgreen?style=flat-square)]()
[![mAP@50](https://img.shields.io/badge/mAP%4050-97.1%25-brightgreen?style=flat-square)]()
[![Precision](https://img.shields.io/badge/Precision-96.8%25-brightgreen?style=flat-square)]()
[![Recall](https://img.shields.io/badge/Recall-95.9%25-brightgreen?style=flat-square)]()
[![Inference](https://img.shields.io/badge/Inference-18.4ms%20%7C%2054%20FPS-blue?style=flat-square)]()

<a href="https://youtu.be/QV3yqZAQpqA">🎥 Watch Demo</a> •
<a href="#-features">✨ Features</a> •
<a href="#-architecture">🏗️ Architecture</a> •
<a href="#-installation">⚙️ Installation</a> •
<a href="#-results">📊 Results</a> •
<a href="#-authors">👥 Authors</a>

</div>

---

## 📖 Overview

**AI Dental X-Ray Analyzer** is an end-to-end diagnostic pipeline that automatically analyzes **panoramic dental radiographs**, detecting **31 distinct dental conditions** in a single pass — from caries and bone loss to root fractures and impacted teeth.

Unlike black-box detection systems, this project bakes **explainability** directly into the workflow using **Grad-CAM**, so every AI-generated finding comes with a visual justification clinicians can actually inspect — plus an FDI-compliant tooth chart, a structured diagnostic report, and an automated treatment cost estimate, all delivered through a clean **Gradio** interface.

> ⚠️ **Disclaimer:** This is a research/academic project intended as an *assistive* tool. It is **not** a replacement for professional dental diagnosis.

<div align="center">
<img src="screenshots/fig3_annotated_gradcam.jpg" alt="Annotated X-ray with Grad-CAM heatmap" width="850">
</div>

---

## 🎥 Demo

<div align="center">

[![Watch the demo](https://img.shields.io/badge/▶️_Watch_Full_Demo-on_YouTube-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/QV3yqZAQpqA)

*Click above to see the full walkthrough — upload, detection, Grad-CAM heatmaps, and PDF report generation in action.*

</div>

---

## ✨ Features

| | |
|---|---|
| 🎯 **Multi-Condition Detection** | Detects **31 dental pathology classes** simultaneously in a single panoramic X-ray |
| 🔍 **Explainable AI (Grad-CAM)** | Per-class saliency heatmaps show *exactly* where the model is looking |
| 🦷 **FDI Tooth Charting** | Auto-maps detected conditions onto a standard FDI-compliant tooth chart |
| 📄 **Automated PDF Reports** | One-click structured diagnostic report with severity-gated findings |
| 💰 **Treatment Cost Estimator** | Rule-based cost breakdown for each detected condition |
| 🖥️ **Gradio Web Interface** | Clean, accessible UI — no technical expertise required to use it |
| ⚡ **Real-Time Inference** | ~18.4 ms/image (≈54 FPS) on an RTX 3080 |

---

## 🏗️ Architecture

The system is built as a modular processing pipeline:

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│  Image Ingestion │───▶│  Preprocessing    │───▶│  YOLOv9 Backbone   │
│   (Gradio UI)     │    │ Resize · Normalize│    │  Feature Extraction│
│                    │    │      CLAHE        │    │                    │
└─────────────────┘    └──────────────────┘    └─────────┬──────────┘
                                                            │
                     ┌──────────────────────────────────────┘
                     ▼
        ┌─────────────────────────┐        ┌──────────────────────┐
        │ Multi-class Detection    │───────▶│   Grad-CAM (XAI)      │
        │ + Bounding Box Regression│        │  Saliency Heatmaps    │
        └────────────┬─────────────┘        └───────────┬──────────┘
                       │                                   │
                       ▼                                   ▼
        ┌─────────────────────────────────────────────────────────┐
        │      Structured Report · FDI Chart · Cost Estimation      │
        └─────────────────────────────────────────────────────────┘
```

| Module | Input | Output | Tech Stack |
|---|---|---|---|
| Preprocessing | Raw panoramic X-ray | Normalized 640×640 tensor | OpenCV, NumPy |
| YOLOv9 Inference | 640×640 tensor | Bounding boxes + class scores | PyTorch, Ultralytics |
| Postprocessing | Raw detections | Filtered & annotated results | Python, OpenCV |
| Grad-CAM (XAI) | Feature maps + gradients | Saliency heatmaps | PyTorch Hooks |
| Report Engine | Detection results | Diagnostic report + FDI chart | Python, Gradio |
| Cost Estimator | Detected conditions | Treatment cost breakdown | Python rule engine |

### Why YOLOv9?

YOLOv9 introduces **Programmable Gradient Information (PGI)** and **GELAN** (Generalized Efficient Layer Aggregation Networks), which solve the information-bottleneck problem in deep backbones — critical for detecting small, overlapping pathologies like root fractures in high-resolution radiographs.

---

## 📊 Results

### Overall Performance (31 classes, held-out test set)

| Metric | Score |
|---|---|
| Accuracy | **97.3%** |
| Precision | **96.8%** |
| Recall | **95.9%** |
| F1-Score | **96.3%** |
| mAP@50 | **97.1%** |
| Inference Time | **18.4 ms** (~54 FPS on RTX 3080) |

### Comparison Against Baselines

| Method | Accuracy | Precision | Recall | mAP@50 | Inference |
|---|---|---|---|---|---|
| Manual Diagnosis (Dentists) | 78–85% | 76.0% | 72.5% | N/A | Minutes |
| HOG + SVM | 71.2% | 68.4% | 65.7% | 62.3% | ~340 ms |
| ResNet-50 + Sliding Window | 83.6% | 81.9% | 80.3% | 82.1% | ~210 ms |
| YOLOv5 (Baseline) | 91.4% | 90.2% | 89.1% | 91.8% | ~28 ms |
| YOLOv8 (Prior Work) | 94.7% | 93.8% | 92.6% | 94.5% | ~22 ms |
| **YOLOv9 (Ours)** | **97.3%** | **96.8%** | **95.9%** | **97.1%** | **~18 ms** |

### Top-Performing Classes

| Condition | Precision | Recall | F1-Score | AP@50 |
|---|---|---|---|---|
| Missing Tooth | 98.2% | 97.6% | 97.9% | 98.5% |
| Crown/Restoration | 98.1% | 97.3% | 97.7% | 98.2% |
| Dental Caries | 97.4% | 96.8% | 97.1% | 97.9% |
| Impacted Tooth | 97.7% | 96.5% | 97.1% | 97.8% |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-dental-xray-analyzer.git
cd ai-dental-xray-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.1
- Ultralytics (YOLOv9)
- OpenCV 4.8
- Gradio 4.x
- NumPy, PyYAML, Matplotlib

---

## 🚀 Usage

```bash
python app.py
```

Then open the local Gradio URL in your browser:

1. **Upload** a panoramic dental X-ray image
2. Adjust the **detection confidence threshold** if needed
3. Click **Analyze X-ray**
4. View the annotated radiograph, Grad-CAM heatmaps, FDI tooth chart, and diagnostic report
5. **Export/Download** the auto-generated PDF report

---

## 🧠 Explainability (Grad-CAM)

Every detection is paired with a **Gradient-weighted Class Activation Map** that highlights the exact regions influencing the model's decision:

- 🔴 **Warm colors (red/yellow)** → high-relevance regions
- 🔵 **Cool colors (blue)** → low-relevance regions

Clinically, this means interproximal caries detections localize around the crown/enamel-dentin junction, while periapical pathology detections localize around the root apex — validating that the model has learned clinically meaningful patterns rather than spurious correlations.

---

## 🖼️ Screenshots

<details open>
<summary><b>Web Interface</b></summary>
<br>

| Upload Panel | Analysis Controls |
|---|---|
| ![UI Upload](screenshots/fig1a_ui_upload.jpg) | ![UI Controls](screenshots/fig1b_ui_controls.png) |

</details>

<details open>
<summary><b>Detection & Explainability</b></summary>
<br>

**Annotated X-ray + Grad-CAM Heatmap**
![Annotated X-ray and GradCAM](screenshots/fig3_annotated_gradcam.jpg)

**FDI Tooth Chart**
![FDI Tooth Chart](screenshots/fig2_fdi_chart.jpg)

</details>

<details open>
<summary><b>Diagnostic Report & Export</b></summary>
<br>

| Diagnostic Report | PDF Export |
|---|---|
| ![Diagnostic Report](screenshots/fig4a_diagnostic_report.jpg) | ![Report Export](screenshots/fig4b_report_export.png) |

</details>

<details>
<summary><b>Evaluation Dashboards & Charts</b> (click to expand)</summary>
<br>

![Evaluation Dashboard](screenshots/fig5_evaluation_dashboard.jpg)
![Confusion Matrix](screenshots/fig6_confusion_matrix.png)
![AP@50 per Class](screenshots/fig7_ap50_per_class.png)
![Precision vs Recall](screenshots/fig8_precision_recall.jpg)
![Accuracy by Category](screenshots/fig9_accuracy_by_category.png)

</details>

---

## 🗺️ Roadmap

- [ ] Expand dataset diversity for better generalization
- [ ] Real-time and mobile-based analysis support
- [ ] Integration with hospital management systems
- [ ] Semi-supervised pretraining on unlabeled radiograph collections
- [ ] Structured prediction heads with anatomical priors

---

## 👥 Authors

| Name | Role |
|---|---|
| **V I Hemashree** | AI/ML Development, System Architecture, Explainable AI Integration |
| **Ponn Oviyaa S** | Co-Author |

*School of Computer Science and Engineering, Vellore Institute of Technology, Chennai*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this project useful, consider giving it a star!**

</div>
