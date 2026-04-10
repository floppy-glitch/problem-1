# Problem Set 01 – Chest X-Ray Pneumonia Classification (CNN)

## Overview

This project builds a **Convolutional Neural Network (CNN)** to classify paediatric chest X-ray images as either **NORMAL** or **PNEUMONIA**. The dataset contains 5,863 JPEG images split across `train/`, `val/`, and `test/` directories.

---

## Dataset

| Split | Images |
|-------|--------|
| Train | 5,216  |
| Val   | 16     |
| Test  | 624    |

- **Source**: Chest X-Ray Images (Pneumonia) – Kaggle / Guangzhou Women and Children's Medical Center  
- **Classes**: `NORMAL` (label 0) · `PNEUMONIA` (label 1)  
- **Note**: The dataset is imbalanced — pneumonia images significantly outnumber normal ones in training.

---

## Approach & Methodology

### 1. Pre-processing & Data Augmentation

Since the training set is limited, **ImageDataGenerator** is used with:

- Random rotation (±15°)
- Width / height shifts (10%)
- Shear & zoom (10–15%)
- Horizontal flipping
- Pixel rescaling to [0, 1]

Validation and test images are only rescaled — no augmentation — to give an honest performance estimate.

### 2. Handling Class Imbalance

**Inverse-frequency class weights** are computed and passed to `model.fit()`:

```
weight_c = total_samples / (2 × count_c)
```

This penalises the model more heavily for misclassifying the minority class (NORMAL), improving recall for both classes.

### 3. CNN Architecture

Four convolutional blocks with increasing filter depth (32 → 64 → 128 → 256), each followed by Batch Normalisation and Max-Pooling. Dropout is applied progressively (0.25 → 0.50) to combat over-fitting.

```
Input (150×150×3)
  │
  ├── Block 1: Conv2D(32) × 2 → BN → MaxPool → Dropout(0.25)
  ├── Block 2: Conv2D(64) × 2 → BN → MaxPool → Dropout(0.25)
  ├── Block 3: Conv2D(128) × 2 → BN → MaxPool → Dropout(0.35)
  ├── Block 4: Conv2D(256)     → BN → MaxPool → Dropout(0.35)
  │
  └── GlobalAveragePooling2D
        → Dense(256, ReLU) → BN → Dropout(0.5)
        → Dense(128, ReLU) → Dropout(0.4)
        → Dense(1, Sigmoid)
```

**GlobalAveragePooling2D** replaces a large Flatten layer, reducing parameters and over-fitting risk.

### 4. Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Max epochs | 30 |
| Input size | 150 × 150 |

**Callbacks used:**

- `EarlyStopping` – monitors `val_auc`, patience = 6, restores best weights
- `ReduceLROnPlateau` – halves LR when `val_loss` stagnates (patience = 3)
- `ModelCheckpoint` – saves the best model by `val_auc`

### 5. Evaluation Metrics

Because this is a medical task, **recall (sensitivity)** is critical — missing a pneumonia case is far more costly than a false alarm.

Metrics tracked:
- Accuracy
- AUC-ROC
- Recall (Sensitivity)
- Precision
- Confusion Matrix
- Full Classification Report

---

## Key Findings

- Pneumonia cases show characteristic opacification / infiltrates; NORMAL lungs are clear, making visual separation feasible with a moderate-depth CNN.
- Class weighting meaningfully improves recall on the minority (NORMAL) class.
- Batch Normalisation stabilises and accelerates training, reducing the need for very deep networks.
- Early stopping prevents over-fitting on the small validation set.
- The model achieves strong AUC on the test set, indicating good discriminative ability even with limited labelled data.

---

## How to Run

```bash
# 1. Install dependencies
pip install tensorflow scikit-learn matplotlib seaborn

# 2. Download & extract the dataset so the structure is:
#    chest_xray/
#      train/NORMAL/   train/PNEUMONIA/
#      val/NORMAL/     val/PNEUMONIA/
#      test/NORMAL/    test/PNEUMONIA/

# 3. Run the script
python cnn_pneumonia_classifier.py
```

Outputs produced:
- `best_cnn_model.keras` – best checkpoint by val AUC
- `cnn_pneumonia_final.keras` – final model after training
- `training_history.png` – accuracy / loss / AUC curves
- `confusion_matrix.png` – test-set confusion matrix

---

## File Structure

```
problem_set_01/
├── cnn_pneumonia_classifier.py   # Full training & evaluation pipeline
└── README.md                     # This file
```

---

## Dependencies

```
tensorflow >= 2.12
scikit-learn
matplotlib
seaborn
numpy
```
