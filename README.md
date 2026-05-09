# Face Mask Detection System

A real-time face mask detection system built with deep learning and computer vision. This was our 4th semester group project — we trained three CNN models, compared them head-to-head, and wrapped everything in a live webcam web app.

---

## What This Project Does

Point your webcam at someone. The app tells you in real time whether they're wearing a mask or not — with a confidence score and a bounding box drawn around their face. Simple idea, solid execution.

Under the hood:
- **OpenCV Haar Cascade** finds faces in the frame
- **EfficientNetB0** (our best model) classifies each face as `Mask` or `No Mask`
- A **Flask server** streams the annotated video to your browser via MJPEG

---

## Team & Contributions

| Member | Role | What They Did |
|---|---|---|
| **Soumya** | Backend / Web | Built the entire Flask application — camera manager, async inference engine, MJPEG streaming, CPU optimisations |
| **Subham** | Data & EDA | Exploratory data analysis, class distribution analysis, image preprocessing pipeline, data generators |
| **Avijit** | ML Engineer | Trained all three CNN models (EfficientNetB0, MobileNetV2, ResNet50) with transfer learning and class weighting |
| **Sreyan** | Evaluation | Comparative model evaluation — accuracy, precision, recall, F1, ROC-AUC, PR curves, confusion matrices |

---

## Results

All three models were evaluated on a held-out test set of **1,726 images** (863 per class).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **EfficientNetB0** | **100%** | **1.000** | **1.000** | **1.000** | **1.000** |
| MobileNetV2 | 99.59% | 0.992 | 1.000 | 0.996 | 1.000 |
| ResNet50 | 99.54% | 0.991 | 1.000 | 0.995 | ~1.000 |

**EfficientNetB0 was selected** as the production model for the live app — perfect test-set performance with the smallest weight file (~49 MB vs ResNet50's 283 MB).

---

## Project Structure

```
4thSemproject/
│
├── app.py                        # Flask web server (Soumya)
├── step_eda.py                   # EDA pipeline (Subham)
├── requirements.txt
│
├── avijit_task/                  # ML training pipeline (Avijit)
│   ├── step6_preprocessing_pipeline.py
│   ├── step7_image_data_generators.py
│   ├── step10_validation_split_generators.py
│   ├── step11_class_weights.py
│   ├── step12_train_model.py
│   ├── step13_evaluate_model.py
│   ├── mask_model_EfficientNetB0.h5   # ~49 MB (Git LFS)
│   ├── mask_model_MobileNetV2.h5      # ~27 MB (Git LFS)
│   └── mask_model_ResNet50.h5         # ~283 MB (Git LFS)
│
├── sreyan/                       # Evaluation & charts (Sreyan)
│   ├── member3_evaluation.py
│   ├── generate_assets.py
│   ├── generate_extra_charts.py
│   ├── roc_curve_comparison.png
│   ├── pr_curve_comparison.png
│   ├── radar_chart.png
│   ├── metrics_heatmap.png
│   └── all_models_evaluation_metrics.json
│
├── reports/                      # Auto-generated JSON reports
│   ├── eda_01_class_distribution.png
│   ├── eda_02_sample_grid.png
│   └── (step reports for every pipeline step)
│
├── templates/
│   └── index.html                # Web UI
├── static/                       # CSS assets
└── data/
    ├── train/
    └── test/
```

---

## Running It Locally

### 1. Clone the repo

```bash
git clone https://github.com/subham007ai/Face-Mask-detection.git
cd Face-Mask-detection
```

> **Note:** Model weights are stored with Git LFS. Make sure you have `git-lfs` installed — run `git lfs pull` after cloning if the `.h5` files come out as pointer files.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `flask>=3.0`, `opencv-python>=4.8`, `numpy>=1.24`, `tensorflow>=2.15`, `keras>=3.0`

### 3. Start the server

```bash
python app.py
```

Open **http://localhost:5001** in your browser. Click the start button, allow webcam access, and you're live.

---

## How the Model Was Trained

**Transfer learning** — all three backbones were loaded with pretrained ImageNet weights and the base was frozen. Only a custom head was trained:

```
Base (frozen ImageNet weights)
  → GlobalAveragePooling2D
  → Dropout(0.5)
  → Dense(1, activation='sigmoid')
```

- **Input size:** 224×224 RGB
- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Binary crossentropy
- **Epochs:** 25
- **Batch size:** 32
- **Class weights:** Applied to handle any dataset imbalance

---

## Flask App — Technical Notes

The app was built with CPU performance as a priority since not everyone has a GPU. Key design decisions (all by Soumya):

- **Async inference engine** — ML runs in a background thread. The camera loop never blocks waiting for `model.predict()`.
- **Frame skipping** — Inference runs every 3rd frame; cached bounding boxes are reused in between.
- **Half-res face detection** — Haar Cascade runs on a 50% scaled-down copy of the frame, then boxes are scaled back up.
- **Aspect ratio filtering** — Rejects detections that aren't shaped like a face (catches false positives from hands, etc.).
- **Confidence thresholding** — Predictions below 65% confidence are silently ignored.

---

## EDA Highlights

Subham's exploratory analysis covered:
- Class distribution (balanced: ~50/50 Mask vs No Mask)
- Sample grid preview of training images
- Image resolution distribution
- Per-channel pixel statistics (mean, std)
- Blur score analysis using Laplacian variance (threshold: 65)

All charts saved to `reports/`.

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model training and inference
- **OpenCV** — face detection and video capture
- **Flask** — web server and MJPEG streaming
- **scikit-learn** — evaluation metrics
- **Matplotlib** — charts and visualisations

---

## Notes

- The model weights are tracked with **Git LFS**. If you clone this and the `.h5` files are tiny text files, run `git lfs pull`.
- The app runs on CPU — it's slower than GPU inference but works on any machine.
- EfficientNetB0 is the default model loaded by the app. You can swap it for MobileNetV2 by editing `MODEL_PATH` in `app.py`.

---

*4th Semester Project — Group submission.*
