# How to Run the Face Mask Detection App

This project uses a unified architecture where **Flask serves both the backend ML logic and the frontend UI** at the same time. You do not need to run a separate frontend server.

---

## 1. Prerequisites

- Python **3.9 – 3.13** installed
- Git (with Git LFS for image dataset)
- A webcam

---

## 2. Clone the Repository

```bash
git clone https://github.com/subham007ai/Face-Mask-detection.git
cd Face-Mask-detection
```

> **Note:** This repo uses **Git LFS** for the image dataset. If images appear as pointer files, run:
> ```bash
> git lfs pull
> ```

---

## 3. Place the Trained Model

The trained model file (`mask_model_EfficientNetB0.h5`) is **not stored in this repository** due to its size (~47 MB after compression). You must obtain it separately from the project team and place it here:

```
avijit_task/mask_model_EfficientNetB0.h5
```

> The app will print a clear error message and exit gracefully if the file is missing.

---

## 4. Create & Activate a Virtual Environment

**On Windows:**
```cmd
python -m venv venv_new
.\venv_new\Scripts\activate
```

**On macOS / Linux:**
```bash
python -m venv venv_new
source venv_new/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.14+ users:** If TensorFlow fails to install, try:
> ```bash
> pip install tf-nightly tf-keras
> ```

---

## 6. Run the Application

```bash
python app.py
```

You should see:
```
[INFO] Loading model from avijit_task/mask_model_EfficientNetB0.h5 …
[INFO] Model loaded successfully.

  Face Mask Detection Server  [CPU-Optimised]
  Open  http://localhost:5001  in your browser.
```

---

## 7. Use the App

1. Open **http://localhost:5001** in your browser (Chrome / Edge recommended)
2. Click **"Start Detection"**
3. Allow **camera permissions** when prompted
4. The app will detect faces in real-time and classify them as **Mask** or **No Mask**

---

## Performance Notes

The app runs on **CPU** (TensorFlow ≥ 2.11 does not support native Windows GPU). Several optimisations are built in to keep it smooth:

| Optimisation | Detail |
|---|---|
| Frame skipping | ML inference runs every 3rd frame |
| Async inference thread | Camera stream never blocks on `predict()` |
| Haar detection at 0.5× scale | Faster face detection |
| Reduced capture resolution | 480 × 360 px |
| Strict false-positive filtering | `minNeighbors=9`, aspect ratio check, 65% confidence threshold |

> **GPU users (RTX / GTX):** For GPU acceleration on Windows, install Python 3.11 and then:
> ```bash
> pip install tensorflow-cpu==2.13.0 tensorflow-directml-plugin
> ```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Address already in use` | Change port at bottom of `app.py`: `app.run(..., port=5002)` |
| Camera not turning on | Close Zoom / Teams / any app using the webcam |
| Hand detected as Mask | Already fixed — strict filtering is active. Ensure good lighting |
| Model load error | Confirm `mask_model_EfficientNetB0.h5` is in `avijit_task/` |
| TensorFlow import error | Run `pip install tensorflow>=2.15 keras>=3.0` |

---

## Running the Evaluation Pipeline (Team Members)

To regenerate real ROC / PR curves and metrics from the trained models:

```bash
python sreyan/member3_evaluation.py
```

Outputs saved to `sreyan/`:
- `roc_curve_comparison.png`
- `pr_curve_comparison.png`
- `all_models_evaluation_metrics.json`

---

## Project Team

| Member | Contribution |
|---|---|
| **Subham** | Data pipeline, EDA, preprocessing, project setup, web app |
| **Avijit** | Steps 6–13: preprocessing → training → evaluation pipeline |
| **Sreyan** | Model evaluation, ROC/PR curve generation |

---

*4th Semester Academic Project — Face Mask Detection using Deep Learning*
