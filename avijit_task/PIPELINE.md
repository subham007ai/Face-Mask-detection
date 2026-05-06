# 🔬 Data Pipeline Log

> This document tracks every step of the dataset inspection, cleaning, and preparation pipeline.  
> Each step is auditable via the corresponding JSON report in `reports/`.

---

## Step 1: Dataset Inspection

- Verified the expected split and class structure:
  - `Train/WithMask`
  - `Train/WithoutMask`
  - `Validation/WithMask`
  - `Validation/WithoutMask`
  - `Test/WithMask`
  - `Test/WithoutMask`
- Counted images before cleaning:
  - `Train`: `WithMask` 5000, `WithoutMask` 5000
  - `Validation`: `WithMask` 400, `WithoutMask` 400
  - `Test`: `WithMask` 483, `WithoutMask` 509
  - Total: 11,792 images
- Randomly inspected 40 images across splits and classes.
- Findings from inspection:
  - Labels looked consistent with the folder names.
  - Several images were blurry, low-resolution, or strongly compressed.
  - No corrupted files were found in the inspection sweep.

## Step 2: Data Cleaning

### 2.1 Remove Corrupted Images

- Ran a full image verification sweep using Pillow.
- Outcome:
  - Corrupted files found: 0
  - Files removed for corruption: 0

### 2.2 Remove Duplicate Images

- Used perceptual hashing with a conservative Hamming-distance threshold of 4.
- Cleaning policy:
  - Prefer `Train` over `Validation` over `Test`
  - Prefer non-augmented files over `Augmented_*` files
  - Prefer higher-resolution and larger-image files when duplicates tie
- Outcome:
  - Duplicate files removed: 212
  - Cross-split duplicates removed: 52
  - Most duplicate removals came from `WithMask`, which suggests many repeated or near-repeated augmented samples

### Post-Cleaning Counts

- `Train`: `WithMask` 4842, `WithoutMask` 4998
- `Validation`: `WithMask` 378, `WithoutMask` 400
- `Test`: `WithMask` 453, `WithoutMask` 509
- Total after cleaning: 11,580 images

> 📎 Detailed report: [`reports/cleanup_step2_report.json`](reports/cleanup_step2_report.json)

## Step 3: Quality Filtering

- Used a conservative quality filter with three rules:
  - Remove images with no detected face using OpenCV YuNet
  - Remove images whose shortest side is below 64 pixels
  - Remove images with Laplacian blur variance below 65
- Outcome:
  - Images removed: 1,445
  - Blurry images removed: 118
  - Extremely low-resolution images removed: 367
  - No-face images removed: 1,149
- Final counts after step 3:
  - `Train`: `WithMask` 3692, `WithoutMask` 4922
  - `Validation`: `WithMask` 267, `WithoutMask` 397
  - `Test`: `WithMask` 354, `WithoutMask` 503
- Total after step 3: 10,135 images

> 📎 Detailed report: [`reports/cleanup_step3_report.json`](reports/cleanup_step3_report.json)

## Step 4: Dataset Size Control

- Kept the full `WithMask` class because it was the smaller class after cleaning.
- Downsampled `WithoutMask` within each split to match the `WithMask` count in that split.
- Selection rule for `WithoutMask`:
  - Prefer better face detection confidence
  - Prefer larger visible face area
  - Prefer higher blur variance
  - Prefer larger image size
- Outcome:
  - Images removed: 1,509
  - Final total: 8,626 images
  - Final class balance: 4,313 `WithMask` and 4,313 `WithoutMask`
- Final split counts:
  - `Train`: `WithMask` 3692, `WithoutMask` 3692
  - `Validation`: `WithMask` 267, `WithoutMask` 267
  - `Test`: `WithMask` 354, `WithoutMask` 354

> 📎 Detailed report: [`reports/cleanup_step4_report.json`](reports/cleanup_step4_report.json)

## Step 5: Final Shuffle and 80/20 Split

- Shuffled the cleaned 8,626-image dataset before splitting.
- Used a strict stratified 80/20 split by class to preserve balance.
- Output structure:
  - `data/train/WithMask`
  - `data/train/WithoutMask`
  - `data/test/WithMask`
  - `data/test/WithoutMask`
- Outcome:
  - `train`: 6,900 images
  - `test`: 1,726 images
  - `train` class balance: 3,450 `WithMask`, 3,450 `WithoutMask`
  - `test` class balance: 863 `WithMask`, 863 `WithoutMask`
- Verification:
  - No SHA-256 overlap between train and test
  - The source cleaned dataset was copied, not moved, so the original cleaned files remain untouched

> 📎 Detailed report: [`reports/step5_split_report.json`](reports/step5_split_report.json)

## Step 6: Image Standardization and Augmentation

- Added a reusable preprocessing pipeline in `step6_preprocessing_pipeline.py`.
- Standardization rules:
  - Resize all images to `224x224`
  - Rescale pixel values with `1./255`
- Training data augmentation:
  - Random rotation in the range `-20` to `20` degrees
  - Random zoom in the range `0.85` to `1.15`
  - Horizontal flip with 50% probability
- Test data handling:
  - Resize to `224x224`
  - Rescale with `1./255`
  - No augmentation applied
- Verification:
  - Training preview confirmed augmentation is applied only to training samples
  - Test preview confirmed resizing and rescaling only

> 📎 Report: [`reports/step6_pipeline_report.json`](reports/step6_pipeline_report.json)  
> 📎 Preview: [`reports/step6_preview.png`](reports/step6_preview.png)

## Step 7: Keras ImageDataGenerator Setup

- Added `step7_image_data_generators.py` to wire the final split into Keras-style generators.
- Configuration:
  - Batch size: `32`
  - Class mode: `binary`
  - Target size: `224x224`
- Training generator:
  - Rescale with `1./255`
  - Rotation range: `20`
  - Zoom range: `0.15`
  - Horizontal flip: enabled
- Test generator:
  - Rescale with `1./255`
  - No augmentation
- Verification:
  - Training generator loads `6900` samples
  - Test generator loads `1726` samples
  - Batch shape for both generators is `32 x 224 x 224 x 3`
  - Label shape for both generators is `32`

> 📎 Detailed report: [`reports/step7_generator_report.json`](reports/step7_generator_report.json)

## Step 8: Generator Sanity Check

- Ran the generator verification through `next(train_data)` and `next(test_data)`.
- Verified:
  - Batch shape is correct for both generators: `32 x 224 x 224 x 3`
  - Labels are binary and aligned with the class indices
  - No runtime errors occurred
- This confirms the preprocessing and generator pipeline is ready for model training.

## Step 9: Final Confirmation

| Check                              | Status |
|------------------------------------|--------|
| No corrupted images                | ✅     |
| Duplicate images removed           | ✅     |
| Dataset balanced                   | ✅     |
| Train/test split correct           | ✅     |
| Test set untouched after split     | ✅     |
| No augmentation applied to test    | ✅     |
| Image size correct (224×224)       | ✅     |
| Data generators working            | ✅     |
| Batch loads without error          | ✅     |

> 📎 Detailed report: [`reports/step9_final_confirmation.json`](reports/step9_final_confirmation.json)

## Step 10: Validation Split for Training

- Added `step10_validation_split_generators.py` to introduce a validation split without touching the test set.
- Split rule:
  - Validation is drawn only from `data/train`
  - Validation split ratio is `0.15`
  - Test data remains untouched until final evaluation
- Generator behavior:
  - Training data uses rescaling + rotation + zoom + horizontal flip
  - Validation data uses rescaling only
  - Test data uses rescaling only
  - Batch size remains `32`
  - Class mode remains `binary`
- Verified generator counts:
  - Training samples: `5866`
  - Validation samples: `1034`
  - Test samples: `1726`
- Verified batch shapes:
  - Training: `32 x 224 x 224 x 3`
  - Validation: `32 x 224 x 224 x 3`
  - Test: `32 x 224 x 224 x 3`

> 📎 Detailed report: [`reports/step10_validation_split_report.json`](reports/step10_validation_split_report.json)

## Step 11: Class Weight Support

- Added `step11_class_weights.py` to compute balanced class weights for model training.
- Purpose:
  - Handle any residual class imbalance and improve model robustness
  - Weights are computed using `sklearn.utils.class_weight.compute_class_weight` with `class_weight='balanced'`
- Computed weights from training data:
  - `WithMask` (class 0): weight `1.0`
  - `WithoutMask` (class 1): weight `1.0`
  - Training data is perfectly balanced at `2933` samples per class, so both weights are equal
- Usage in model training:
  ```python
  from step11_class_weights import get_class_weights

  class_weights = get_class_weights()
  model.fit(..., class_weight=class_weights)
  ```
- The weights auto-adjust if the dataset composition changes in a future re-run.

> 📎 Detailed report: [`reports/step11_class_weights_report.json`](reports/step11_class_weights_report.json)

## EDA: Exploratory Data Analysis

- Added `step_eda.py` to produce a full visual and statistical exploration of the final prepared dataset (`data/train` and `data/test`).
- Analyses performed:
  1. **Class distribution** – bar charts showing per-class image counts for train and test splits (perfectly balanced at 3,450 / 3,450 train, 863 / 863 test).
  2. **Sample image grid** – 5 representative thumbnails per class per split to confirm visual quality and label consistency.
  3. **Resolution analysis** – overlapping width/height histograms across classes. Key finding: `WithMask` images cluster near 224px (many already at target size), while `WithoutMask` images cluster around 106px (smaller native resolution – these are cropped face patches).
  4. **Pixel channel stats** – grouped bar chart of per-channel (R/G/B) mean intensity. `WithoutMask` images show notably higher R-channel mean (0.605) vs `WithMask` (0.544), reflecting more exposed skin tone. Both classes show similar G and B distributions.
  5. **Blur score distribution** – Laplacian variance histograms and box plots. `WithoutMask` images are sharper on average (mean ~817 vs ~598 for `WithMask`), consistent with them being tightly cropped face crops vs wider scene images with masks.
- Outputs:
  - `reports/eda_01_class_distribution.png`
  - `reports/eda_02_sample_grid.png`
  - `reports/eda_03_resolution.png`
  - `reports/eda_04_channel_stats.png`
  - `reports/eda_05_blur.png`
- Elapsed: 44.3 seconds for 6,900 train + 1,726 test images.

> 📎 Detailed report: [`reports/step_eda_report.json`](reports/step_eda_report.json)
