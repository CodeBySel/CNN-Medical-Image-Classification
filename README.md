# 🫁 Chest X-Ray Pneumonia Detection — Deep Learning Pipeline

> **Binary classification of chest X-ray images (NORMAL vs PNEUMONIA) using custom CNNs and ResNet50 transfer learning.**  
> Course project for *Introduction to Artificial Intelligence L2* | Selin Dyortkardeshler | 

---

## 📊 Final Results at a Glance

| Model | Accuracy | F1 Score | Pneumonia Recall | ROC-AUC |
|---|---|---|---|---|
| Custom CNN — Variant A (baseline) | 73.08% | 73.48% | 66.41% | 0.784 |
| Custom CNN — Variant B (improved) | 84.62% | 85.43% | 83.59% | 0.925 |
| **ResNet50 Transfer Learning** | **92.15%** | **93.82%** | **95.38%** | **0.974** |

> **Best model:** ResNet50 Transfer Learning at default threshold (t=0.5)  
> **False negatives on test set:** only 18 out of 390 pneumonia cases missed  
> **Overall improvement from baseline to final:** +28.5% accuracy, 86% fewer false negatives

---

## 🗂️ Project Structure

```
chest-xray-pneumonia-detection/
│
├── src/
│   ├── data_preparation.py         # Report 1 — Dataset loading, EDA, augmentation pipeline
│   ├── cnn_training_variantA.py    # Report 2 — Baseline CNN (Variant A)
│   ├── cnn_training_variantB.py    # Report 2 — Improved CNN (Variant B, L2 + Dropout 0.6)
│   └── resnet50_training.py        # Report 3 — ResNet50 two-phase transfer learning
│
├── results/
│   └── test_metrics.json           # Saved test set metrics (Variant B)
│
├── reports/
│   ├── Report1_DataPreparation.pdf
│   ├── Report2_BaselineCNN.pdf
│   ├── Report3_TransferLearning.pdf
│   └── Report4_ComparisonAndConclusion.pdf
│
└── README.md
```

---

## 🧠 Approach

This project was developed iteratively across four reports, each building on the findings of the previous one.

### Report 1 — Data Preparation
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 pediatric X-rays
- **Class imbalance** identified: 74.3% PNEUMONIA vs 25.7% NORMAL in training set
- Addressed via **balanced class weights** (NORMAL ≈ 1.94, PNEUMONIA ≈ 0.67)
- Augmentation pipeline: rotation ±15°, horizontal flip, zoom ±10%, brightness [0.8–1.2]
- Images resized to **224×224** for ResNet50 compatibility

<img width="2336" height="1183" alt="fig2_sample_images" src="https://github.com/user-attachments/assets/0447452b-cbb1-47ef-88f2-eb94d98079a1" />


### Report 2 — Custom CNN (Variants A & B)
- Architecture: 3 convolutional blocks (32→64→128 filters) + GlobalAvgPooling + Dense head
- **110,785 trainable parameters** — efficient, runs on consumer hardware
- **Key finding:** Kaggle's original 16-image validation set caused catastrophic training instability
- Variant B fix: 20% stratified validation split → +21% accuracy **without any architecture change**
- Additional regularization: Dropout 0.6 + L2 λ=1×10⁻⁴ reduced train/test gap from 30 → 7 points

<img width="2084" height="763" alt="fig1_training_curves" src="https://github.com/user-attachments/assets/c4a94f08-ec54-48bc-953f-282116977ba0" />


### Report 3 — ResNet50 Transfer Learning
- Backbone: **ResNet50** pre-trained on ImageNet (1.2M images)
- **Two-phase training strategy:**
  - Phase 1 (Feature Extraction): backbone frozen, head trained for 20 epochs at LR=0.001
  - Phase 2 (Fine-tuning): top 30 layers unfrozen, LR=0.00001
- Achieves ROC-AUC = **0.974** — near radiologist-level discrimination


### Report 4 — Comparison & Conclusions
- ResNet50 outperforms Variant B on every primary metric
- Variant B retains advantages in **efficiency** (200× fewer parameters) and **probability calibration**
- Clinical deployment recommendation: ResNet50 at t=0.5 for primary screening
  
<img width="1334" height="732" alt="comparison_chart" src="https://github.com/user-attachments/assets/dc14633f-f5f4-4dbb-a4bf-ec8b8651028f" />

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install tensorflow scikit-learn matplotlib seaborn pillow
```

### Dataset
Download the dataset from Kaggle and extract it to `./chest_xray/`:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

### Run the pipeline
```bash
# Step 1: Data preparation & visualization
python src/data_preparation.py

# Step 2a: Train baseline CNN (Variant A)
python src/cnn_training_variantA.py

# Step 2b: Train improved CNN (Variant B)
python src/cnn_training_variantB.py

# Step 3: Train ResNet50 transfer learning model
python src/resnet50_training.py
```

---

## 📈 Key Metrics — ResNet50 (Best Model)

| Threshold | Accuracy | Precision | Pneumonia Recall | Normal Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| t = 0.5 (default) | 92.15% | 92.31% | 95.38% | 86.75% | 93.82% | 0.974 |
| t = 0.971 (optimal) | 90.38% | 97.41% | 86.92% | 96.15% | 91.87% | 0.974 |

**Confusion Matrix at t=0.5 (test set, 624 images):**
```
              Predicted
              NORMAL   PNEUMONIA
True NORMAL    203        31
     PNEUMONIA  18       372
```

---

## 🔬 Key Findings

1. **Data pipeline quality > model architecture** — Fixing the 16-image validation set improved accuracy by 21 points with zero architectural changes.
2. **Transfer learning is highly effective** for small medical datasets — ResNet50 pre-trained on natural images transfers well to X-rays because low-level visual features (edges, textures) are shared across domains.
3. **Threshold selection matters clinically** — The same model can be optimized for either minimizing missed diagnoses (t=0.5, recall=95.4%) or minimizing false alarms (t=0.971, precision=97.4%).
4. **Class imbalance** is substantially resolved by transfer learning (recall gap: 77.86% → 8.63%).

---

## ⚠️ Limitations

- Dataset covers only pediatric patients (ages 1–5) from a single center in China — generalizability is unvalidated
- ResNet50 exhibits poor probability calibration (optimal threshold 0.971) — Platt scaling recommended before clinical use
- No external validation set — performance on different imaging equipment is unknown

---

## 📚 References

1. Mooney, P. (2018). [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Kaggle.
2. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases through Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

---

## 👩‍💻 Author

**Selin Dyortkardeshler** |
Course: Introduction to Artificial Intelligence L2 | Instructor: Agnieszka Duraj  
University of Vizja, Warsaw | 2026
