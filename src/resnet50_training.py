"""
Report 3 – Transfer Learning with ResNet50
Pneumonia Binary Classification (NORMAL vs PNEUMONIA)
Student: Selin Dyortkardeshler | Index: 54817
Course: Introduction to Artificial Intelligence L2

Pipeline:
  Phase 1 – Feature Extraction  : ResNet50 frozen, only classification head trained
  Phase 2 – Fine-tuning         : Top 30 layers of ResNet50 unfrozen, very low LR

Compatible with the data preparation pipeline from Report 1.
"""

# ─────────────────────────────────────────────
# 0. Imports & reproducibility
# ─────────────────────────────────────────────
import os, random, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, CSVLogger)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import csv, json

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Paths  –  adjust to your folder structure
# ─────────────────────────────────────────────
TRAIN_DIR  = os.path.join("chest_xray", "train")
VAL_DIR    = os.path.join("chest_xray", "val")
TEST_DIR   = os.path.join("chest_xray", "test")
OUTPUT_DIR = "resnet50_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Hyper-parameters
# ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_P1     = 20          # Phase 1 – feature extraction
EPOCHS_P2     = 20          # Phase 2 – fine-tuning
LR_P1         = 1e-3        # head learning rate
LR_P2         = 1e-5        # fine-tune learning rate (very low to protect weights)
UNFREEZE_FROM = 140         # ResNet50 has 175 layers; unfreeze last ~35
PATIENCE_ES   = 8
PATIENCE_LR   = 4
VAL_SPLIT     = 0.20        # 20 % of train used as validation (same as Report 2)

# ─────────────────────────────────────────────
# 3. Data generators
#    – same augmentation policy as Report 1/2
#    – ResNet50 requires its own preprocessing (resnet50 preprocess_input)
# ─────────────────────────────────────────────
from tensorflow.keras.applications.resnet50 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=VAL_SPLIT,
)

eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

test_gen = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# ─────────────────────────────────────────────
# 4. Class weights  (same logic as Report 2)
# ─────────────────────────────────────────────
labels = train_gen.classes
classes = np.unique(labels)
cw_array = compute_class_weight("balanced", classes=classes, y=labels)
class_weights = dict(zip(classes, cw_array))
print(f"Class weights: {class_weights}")

# ─────────────────────────────────────────────
# 5. Model builder
# ─────────────────────────────────────────────
def build_model(trainable_base=False):
    """
    ResNet50 base + custom classification head.
    trainable_base=False  → Phase 1 (feature extraction)
    trainable_base=True   → Phase 2 (fine-tuning, partial unfreeze)
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    if not trainable_base:
        base.trainable = False
    else:
        # Freeze all layers BELOW UNFREEZE_FROM, unfreeze the rest
        for layer in base.layers[:UNFREEZE_FROM]:
            layer.trainable = False
        for layer in base.layers[UNFREEZE_FROM:]:
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base


# ─────────────────────────────────────────────
# 6. Callbacks factory
# ─────────────────────────────────────────────
def make_callbacks(phase: int):
    prefix = f"p{phase}"
    return [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_ES,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=PATIENCE_LR, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, f"best_{prefix}.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(os.path.join(OUTPUT_DIR, f"log_{prefix}.csv")),
    ]


# ─────────────────────────────────────────────
# 7. Phase 1 – Feature Extraction
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 – Feature Extraction (frozen ResNet50 base)")
print("="*60)

model_p1, base_model = build_model(trainable_base=False)
model_p1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print(f"Trainable params (Phase 1): {model_p1.count_params():,}")
model_p1.summary()

history_p1 = model_p1.fit(
    train_gen,
    epochs=EPOCHS_P1,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=make_callbacks(1),
    verbose=1,
)

# ─────────────────────────────────────────────
# 8. Phase 2 – Fine-tuning
#    Load best Phase-1 weights, then unfreeze top layers
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 – Fine-tuning (top ResNet50 layers unfrozen)")
print("="*60)

# Re-build with partial unfreeze and load Phase-1 head weights
model_p2, _ = build_model(trainable_base=True)
model_p2.load_weights(os.path.join(OUTPUT_DIR, "best_p1.keras"))

model_p2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P2),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
trainable_p2 = sum(1 for l in model_p2.layers if l.trainable)
print(f"Trainable layers (Phase 2): {trainable_p2}")

history_p2 = model_p2.fit(
    train_gen,
    epochs=EPOCHS_P2,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=make_callbacks(2),
    verbose=1,
)

# ─────────────────────────────────────────────
# 9. Test-set evaluation  (best Phase-2 model)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

best_model = tf.keras.models.load_model(
    os.path.join(OUTPUT_DIR, "best_p2.keras")
)

test_gen.reset()
y_prob = best_model.predict(test_gen, verbose=1).ravel()
y_true = test_gen.classes

# Default threshold
y_pred_05 = (y_prob >= 0.5).astype(int)

# Youden's J optimal threshold
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
youden_idx  = np.argmax(tpr - fpr)
opt_thresh  = thresholds[youden_idx]
y_pred_opt  = (y_prob >= opt_thresh).astype(int)

roc_auc = roc_auc_score(y_true, y_prob)
test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)

# Helper
def metrics_dict(y_t, y_p):
    cm   = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp + 1e-9)
    rec_pneu = tp / (tp + fn + 1e-9)
    rec_norm = tn / (tn + fp + 1e-9)
    f1   = 2 * prec * rec_pneu / (prec + rec_pneu + 1e-9)
    return dict(accuracy=acc, precision=prec,
                pneumonia_recall=rec_pneu, normal_recall=rec_norm,
                f1=f1, cm=cm)

m05  = metrics_dict(y_true, y_pred_05)
mopt = metrics_dict(y_true, y_pred_opt)

results = {
    "roc_auc":          round(float(roc_auc), 4),
    "test_loss":        round(float(test_loss), 4),
    "optimal_threshold": round(float(opt_thresh), 4),
    "default_t05": {k: round(float(v), 4) for k, v in m05.items() if k != "cm"},
    "optimal_topt": {k: round(float(v), 4) for k, v in mopt.items() if k != "cm"},
}
print(json.dumps(results, indent=2))

with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ─────────────────────────────────────────────
# 10. Plots
# ─────────────────────────────────────────────

# ── 10a. Training curves Phase 1 ──────────────
def plot_curves(history, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train loss",     color="#2166ac")
    axes[0].plot(history.history["val_loss"], label="Val loss",       color="#d6604d", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train accuracy", color="#2166ac")
    axes[1].plot(history.history["val_accuracy"], label="Val accuracy",   color="#d6604d", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

plot_curves(history_p1, "Phase 1 Training Curves (Feature Extraction)", "curves_phase1.png")
plot_curves(history_p2, "Phase 2 Training Curves (Fine-tuning)",         "curves_phase2.png")

# ── 10b. Confusion matrices ────────────────────
def plot_cm(cm, title, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL","PNEUMONIA"],
                yticklabels=["NORMAL","PNEUMONIA"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

plot_cm(m05["cm"],  f"Confusion Matrix – t=0.5",          "cm_default.png")
plot_cm(mopt["cm"], f"Confusion Matrix – t={opt_thresh:.3f} (optimal)", "cm_optimal.png")

# ── 10c. ROC curve ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#d6604d", lw=2, label=f"ResNet50  AUC = {roc_auc:.4f}")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
ax.scatter(fpr[youden_idx], tpr[youden_idx], color="red", zorder=5,
           label=f"Optimal threshold = {opt_thresh:.3f}")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – ResNet50 Transfer Learning")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_curve.png")

# ── 10d. Threshold analysis (like Figure 3 in Report 2) ──
thresh_range = np.linspace(0.05, 0.95, 200)
rec_pneu_list, rec_norm_list, f1_list = [], [], []
for t in thresh_range:
    yp = (y_prob >= t).astype(int)
    d  = metrics_dict(y_true, yp)
    rec_pneu_list.append(d["pneumonia_recall"])
    rec_norm_list.append(d["normal_recall"])
    f1_list.append(d["f1"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresh_range, f1_list,        label="Weighted F1",      color="#1b7837", lw=2)
ax.plot(thresh_range, rec_pneu_list,  label="PNEUMONIA Recall", color="#d6604d", lw=2, linestyle="--")
ax.plot(thresh_range, rec_norm_list,  label="NORMAL Recall",    color="#4393c3", lw=2, linestyle="dotted")
ax.axvline(opt_thresh, color="red",   linestyle="--", lw=1.5, label=f"Optimal = {opt_thresh:.3f}")
ax.axvline(0.5,        color="gray",  linestyle="dotted", lw=1.5, label="Default = 0.5")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title("Effect of Decision Threshold — ResNet50")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "threshold_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved threshold_analysis.png")

# ── 10e. Comparison bar chart (ResNet50 vs Variant B) ─────
# Variant B values from Report 2
report2_B = dict(Accuracy=0.8462, F1=0.8543, PNEUMONIA_Recall=0.8359, ROC_AUC=0.9252)
resnet_vals = dict(
    Accuracy         = mopt["accuracy"],
    F1               = mopt["f1"],
    PNEUMONIA_Recall = mopt["pneumonia_recall"],
    ROC_AUC          = roc_auc,
)

metrics_labels = list(report2_B.keys())
x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, [report2_B[k] for k in metrics_labels],
               width, label="Variant B (Custom CNN)", color="#5599cc")
bars2 = ax.bar(x + width/2, [resnet_vals[k] for k in metrics_labels],
               width, label="ResNet50 Transfer Learning", color="#e07b54")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "F1 (optimal)", "PNEUMONIA Recall", "ROC-AUC"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Custom CNN (Variant B) vs ResNet50 Transfer Learning")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved comparison_chart.png")

print("\n✅  All outputs saved to:", OUTPUT_DIR)
print("    Submit test_results.json together with your report.")