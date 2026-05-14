"""
Report 2 - Baseline CNN Model
AI 2026 Project 1 - CNN Medical Image Classification
Selin Dyortkardeshler - 54817

Run this from the folder that contains 'chest_xray/' directory:
    python cnn_training.py
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 60)
print("  Report 2 — Baseline CNN Training")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU found  : {[g.name for g in gpus] if gpus else 'None (CPU mode)'}")
print("=" * 60)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("chest_xray")
TRAIN_DIR  = BASE_DIR / "train"
VAL_DIR    = BASE_DIR / "val"
TEST_DIR   = BASE_DIR / "test"

IMG_SIZE   = (224, 224)
BATCH      = 32
EPOCHS     = 30
LR         = 1e-3
SEED       = 42
OUT_DIR    = Path("report2_figures")
OUT_DIR.mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── DATA GENERATORS ───────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
eval_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', shuffle=True, seed=SEED)

val_ds = eval_gen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', shuffle=False)

test_ds = eval_gen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', shuffle=False)

# Class weights
labels = train_ds.classes
cw_vals = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw_vals))
print(f"\n  Class indices : {train_ds.class_indices}")
print(f"  Class weights : {class_weights}")
print(f"  Train samples : {train_ds.samples}")
print(f"  Val samples   : {val_ds.samples}")
print(f"  Test samples  : {test_ds.samples}\n")

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────
def build_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # ── Block 1 ──
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 224 → 112

        # ── Block 2 ──
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 112 → 56

        # ── Block 3 ──
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 56 → 28

        # ── Head ──
        layers.GlobalAveragePooling2D(),    # 28×28×128 → 128
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="Baseline_CNN")
    return model

model = build_cnn()
model.summary()

total_params = model.count_params()
print(f"\n  Total parameters: {total_params:,}\n")

# ── COMPILE ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.ModelCheckpoint(
        'best_cnn_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=7,
        restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1),
    callbacks.CSVLogger('training_log.csv')
]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting training...")
print("=" * 60)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=cb_list,
    verbose=1
)

print("\n  Training complete!")

# Save history for later use
with open('training_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals]
               for k, vals in history.history.items()}, f)

# ── EVALUATE ON TEST SET ──────────────────────────────────────────────────────
print("\n  Evaluating on test set...")
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds, verbose=0)
print(f"  Test accuracy  : {test_acc:.4f}")
print(f"  Test precision : {test_prec:.4f}")
print(f"  Test recall    : {test_rec:.4f}")
print(f"  Test AUC       : {test_auc:.4f}")

# Predictions
test_ds.reset()
y_pred_prob = model.predict(test_ds, verbose=0).ravel()
y_pred      = (y_pred_prob >= 0.5).astype(int)
y_true      = test_ds.classes

# F1 from classification report
report_dict = classification_report(y_true, y_pred,
                                     target_names=['NORMAL','PNEUMONIA'],
                                     output_dict=True)
f1 = report_dict['weighted avg']['f1-score']
print(f"  Test F1-score  : {f1:.4f}")
print(f"  Test loss      : {test_loss:.4f}")

# ── FIGURES ───────────────────────────────────────────────────────────────────
hist = history.history
epochs_ran = range(1, len(hist['loss']) + 1)
STYLE = {'axes.spines.top': False, 'axes.spines.right': False}
plt.rcParams.update(STYLE)

# ── Fig 1: Loss & Accuracy curves ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_ran, hist['loss'],     color='#4C72B0', lw=2, label='Train loss')
ax1.plot(epochs_ran, hist['val_loss'], color='#DD8452', lw=2, ls='--', label='Val loss')
ax1.set_title('Training vs. Validation Loss', fontweight='bold', fontsize=13)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Binary Cross-Entropy Loss')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs_ran, hist['accuracy'],     color='#4C72B0', lw=2, label='Train accuracy')
ax2.plot(epochs_ran, hist['val_accuracy'], color='#DD8452', lw=2, ls='--', label='Val accuracy')
ax2.set_title('Training vs. Validation Accuracy', fontweight='bold', fontsize=13)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('Baseline CNN — Training Curves', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_training_curves.png")

# ── Fig 2: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL','PNEUMONIA'],
            yticklabels=['NORMAL','PNEUMONIA'],
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Confusion Matrix — Test Set', fontweight='bold', fontsize=13)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.12,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        ha='center', transform=ax.transAxes, fontsize=10, color='#444444')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_confusion_matrix.png")

# ── Fig 3: ROC Curve ──────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc     = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#4C72B0', lw=2.5,
        label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], color='gray', lw=1.5, ls='--', label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.07, color='#4C72B0')
ax.set_title('ROC Curve — Baseline CNN', fontweight='bold', fontsize=13)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_roc_curve.png")

# ── Fig 4: Precision-Recall Curve ────────────────────────────────────────────
prec_c, rec_c, _ = precision_recall_curve(y_true, y_pred_prob)
pr_auc = auc(rec_c, prec_c)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_c, prec_c, color='#DD8452', lw=2.5,
        label=f'PR curve (AUC = {pr_auc:.3f})')
ax.axhline(y=y_true.mean(), color='gray', lw=1.5, ls='--', label='Baseline (prevalence)')
ax.fill_between(rec_c, prec_c, alpha=0.07, color='#DD8452')
ax.set_title('Precision–Recall Curve — Baseline CNN', fontweight='bold', fontsize=13)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_pr_curve.png")

# ── Fig 5: Per-class metrics bar chart ───────────────────────────────────────
classes_lbl = ['NORMAL', 'PNEUMONIA']
metrics_data = {
    'Precision': [report_dict['NORMAL']['precision'],
                  report_dict['PNEUMONIA']['precision']],
    'Recall':    [report_dict['NORMAL']['recall'],
                  report_dict['PNEUMONIA']['recall']],
    'F1-score':  [report_dict['NORMAL']['f1-score'],
                  report_dict['PNEUMONIA']['f1-score']],
}
x = np.arange(len(classes_lbl))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#4C72B0', '#55A868', '#DD8452']
for i, (metric, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x + i*width - width, vals, width, label=metric,
                  color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(classes_lbl, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class Evaluation Metrics — Test Set', fontweight='bold', fontsize=13)
ax.set_ylabel('Score')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_metrics_per_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_metrics_per_class.png")

# ── Fig 6: Prediction probability distribution ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
normal_probs = y_pred_prob[y_true == 0]
pneumo_probs = y_pred_prob[y_true == 1]
ax.hist(normal_probs,   bins=40, color='#4C72B0', alpha=0.7, label='NORMAL (true)')
ax.hist(pneumo_probs,   bins=40, color='#DD8452', alpha=0.7, label='PNEUMONIA (true)')
ax.axvline(0.5, color='red', lw=2, ls='--', label='Decision threshold (0.5)')
ax.set_title('Predicted Probability Distribution — Test Set', fontweight='bold', fontsize=13)
ax.set_xlabel('Predicted Probability (PNEUMONIA)')
ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_prob_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_prob_distribution.png")

# ── PRINT FINAL METRICS SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL TEST SET METRICS")
print("=" * 60)
print(f"  Accuracy   : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Precision  : {report_dict['weighted avg']['precision']:.4f}")
print(f"  Recall     : {report_dict['weighted avg']['recall']:.4f}")
print(f"  F1-score   : {f1:.4f}")
print(f"  ROC-AUC    : {roc_auc:.4f}")
print(f"  Loss       : {test_loss:.4f}")
print(f"  Parameters : {total_params:,}")
print("=" * 60)
print(f"\n  Per-class report:\n")
print(classification_report(y_true, y_pred,
                              target_names=['NORMAL','PNEUMONIA']))

# Save metrics to JSON for report generation
metrics_out = {
    "accuracy": round(float(test_acc), 4),
    "precision": round(float(report_dict['weighted avg']['precision']), 4),
    "recall": round(float(report_dict['weighted avg']['recall']), 4),
    "f1": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "pr_auc": round(float(pr_auc), 4),
    "loss": round(float(test_loss), 4),
    "total_params": int(total_params),
    "epochs_trained": len(hist['loss']),
    "normal_precision": round(float(report_dict['NORMAL']['precision']), 4),
    "normal_recall": round(float(report_dict['NORMAL']['recall']), 4),
    "normal_f1": round(float(report_dict['NORMAL']['f1-score']), 4),
    "pneumonia_precision": round(float(report_dict['PNEUMONIA']['precision']), 4),
    "pneumonia_recall": round(float(report_dict['PNEUMONIA']['recall']), 4),
    "pneumonia_f1": round(float(report_dict['PNEUMONIA']['f1-score']), 4),
    "cm_tn": int(tn), "cm_fp": int(fp),
    "cm_fn": int(fn), "cm_tp": int(tp),
}
with open('test_metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n  All figures saved to: {OUT_DIR}/")
print("  Metrics saved to:     test_metrics.json")
print("  Model saved to:       best_cnn_model.keras")
print("\n  DONE! Send the figures + test_metrics.json back to generate the report.")
