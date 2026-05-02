#  Face Mask Detection

A binary image classification pipeline that detects whether a person is wearing a face mask or not. Built with **TensorFlow/Keras** and a rigorously cleaned dataset.

---

## 📁 Project Structure

```
Face-Mask-Detection/
├── data/
│   ├── train/
│   │   ├── WithMask/          # 3,450 images
│   │   └── WithoutMask/       # 3,450 images
│   └── test/
│       ├── WithMask/          # 863 images
│       └── WithoutMask/       # 863 images
├── reports/                   # Machine-readable audit logs from each pipeline step
│   ├── cleanup_step2_report.json
│   ├── cleanup_step3_report.json
│   ├── cleanup_step4_report.json
│   ├── step5_split_report.json
│   ├── step6_pipeline_report.json
│   ├── step6_preview.png
│   ├── step7_generator_report.json
│   ├── step9_final_confirmation.json
│   ├── step10_validation_split_report.json
│   └── step11_class_weights_report.json
├── step6_preprocessing_pipeline.py
├── step7_image_data_generators.py
├── step10_validation_split_generators.py
├── step11_class_weights.py
├── PIPELINE.md                # Detailed step-by-step cleanup & pipeline log
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/subham007ai/Face-Mask-detection.git
cd Face-Mask-detection
```

> **Note:** This repo uses **Git LFS** for the image dataset. Make sure you have [Git LFS](https://git-lfs.com/) installed:
> ```bash
> git lfs install
> ```
> LFS files are pulled automatically on clone. If images appear as pointer files, run:
> ```bash
> git lfs pull
> ```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Dataset Summary

| Split      | WithMask | WithoutMask | Total  |
|------------|----------|-------------|--------|
| Train      | 3,450    | 3,450       | 6,900  |
| Test       | 863      | 863         | 1,726  |
| **Total**  | **4,313**| **4,313**   | **8,626** |

- **Perfectly balanced** across both classes
- Images resized to `224 × 224` at runtime
- 15% of training data is held out as a validation split during training

---

## 🔧 Pipeline Scripts

Each script is self-contained and can be run independently. They produce JSON reports in `reports/` for auditability.

| Script | Purpose |
|--------|---------|
| `step6_preprocessing_pipeline.py` | Framework-agnostic image preprocessing with augmentation |
| `step7_image_data_generators.py` | Keras `ImageDataGenerator` setup for train/test |
| `step10_validation_split_generators.py` | Adds a 15% validation split from training data |
| `step11_class_weights.py` | Computes balanced class weights for `model.fit()` |

### Quick Usage

```python
from step10_validation_split_generators import build_generators
from step11_class_weights import get_class_weights

# Build train, validation, and test generators
train_data, val_data, test_data = build_generators()

# Get class weights (auto-computed from training data)
class_weights = get_class_weights()

# Use in model training
model.fit(
    train_data,
    validation_data=val_data,
    class_weight=class_weights,
    epochs=25
)
```

---

##  Data Cleaning Pipeline (Completed)

The raw dataset went through an 11-step cleaning and preparation pipeline. Full details are in [`PIPELINE.md`](PIPELINE.md).

| Step | Action | Outcome |
|------|--------|---------|
| 1 | Dataset inspection | 11,792 raw images verified |
| 2 | Remove corrupted + duplicate images | 212 duplicates removed |
| 3 | Quality filtering (blur, resolution, face detection) | 1,445 low-quality images removed |
| 4 | Class balancing via downsampling | Perfect 50/50 balance achieved |
| 5 | Stratified 80/20 train/test split | No data leakage (SHA-256 verified) |
| 6 | Image standardization + augmentation pipeline | 224×224, 1./255 rescaling |
| 7 | Keras ImageDataGenerator wiring | Batch size 32, binary mode |
| 8 | Generator sanity check | All batch shapes verified |
| 9 | Final confirmation | All checks passed ✅ |
| 10 | Validation split (15% from train) | 5,866 train / 1,034 val / 1,726 test |
| 11 | Class weight computation | Balanced weights ready for training |

---

##  Next Steps

- [ ] Build and train the CNN model architecture
- [ ] Evaluate on test set with confusion matrix and classification report
- [ ] Export trained model for inference
- [ ] Build a real-time detection demo with webcam

---

##  Team

- **Subham** — Data pipeline, preprocessing, and project setup

---

## 📄 License

This project is for academic purposes (4th Semester Project).
