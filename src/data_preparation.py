"""
Report 1 - Data Preparation
CNN Medical Image Classification - Chest X-Ray Pneumonia Detection
AI 2026 Project 1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 0: Install / import required libraries
# ─────────────────────────────────────────────
# pip install tensorflow scikit-learn matplotlib seaborn pillow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# STEP 1: Dataset paths
# ─────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to: ./chest_xray/

BASE_DIR   = Path("chest_xray")
TRAIN_DIR  = BASE_DIR / "train"
VAL_DIR    = BASE_DIR / "val"
TEST_DIR   = BASE_DIR / "test"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# STEP 2: Dataset statistics
# ─────────────────────────────────────────────
def count_images(directory):
    counts = {}
    for cls in sorted(os.listdir(directory)):
        cls_path = directory / cls
        if cls_path.is_dir():
            counts[cls] = len(list(cls_path.glob("*.jpeg")) +
                              list(cls_path.glob("*.jpg")) +
                              list(cls_path.glob("*.png")))
    return counts

train_counts = count_images(TRAIN_DIR)
val_counts   = count_images(VAL_DIR)
test_counts  = count_images(TEST_DIR)

total_train = sum(train_counts.values())
total_val   = sum(val_counts.values())
total_test  = sum(test_counts.values())
total_all   = total_train + total_val + total_test

print("=" * 50)
print("DATASET STATISTICS")
print("=" * 50)
print(f"\nTrain   ({total_train:>5} images): {train_counts}")
print(f"Val     ({total_val:>5} images): {val_counts}")
print(f"Test    ({total_test:>5} images): {test_counts}")
print(f"\nTotal: {total_all} images")
print(f"Train: {total_train/total_all*100:.1f}%  "
      f"Val: {total_val/total_all*100:.1f}%  "
      f"Test: {total_test/total_all*100:.1f}%")

# ─────────────────────────────────────────────
# STEP 3: Class imbalance analysis
# ─────────────────────────────────────────────
classes = sorted(train_counts.keys())
class_labels = np.repeat(np.arange(len(classes)),
                          [train_counts[c] for c in classes])
weights = compute_class_weight('balanced', classes=np.unique(class_labels),
                                y=class_labels)
print("\nClass weights (for imbalance):", dict(zip(classes, weights.round(3))))

# ─────────────────────────────────────────────
# STEP 4: Preprocessing pipeline
# ─────────────────────────────────────────────

# TRAIN: augmentation + normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Pixel normalization [0,1]
    rotation_range=15,         # Random rotation ±15°
    width_shift_range=0.1,     # Horizontal shift
    height_shift_range=0.1,    # Vertical shift
    zoom_range=0.1,            # Random zoom
    horizontal_flip=True,      # Mirror images
    brightness_range=[0.8,1.2],# Brightness variation
    fill_mode='nearest'
)

# VAL / TEST: only normalization (NO augmentation!)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=True, seed=42
)
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)

print("\nClass indices:", train_generator.class_indices)

# ─────────────────────────────────────────────
# STEP 5: Visualizations
# ─────────────────────────────────────────────
os.makedirs("report_figures", exist_ok=True)

# --- Fig 1: Class distribution ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
splits = [("Train", train_counts), ("Validation", val_counts), ("Test", test_counts)]
colors = ['#4C72B0', '#DD8452']

for ax, (name, counts) in zip(axes, splits):
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Images")
    for bar, (cls, val) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.values()) * 1.2)
    ax.spines[['top','right']].set_visible(False)

plt.suptitle("Class Distribution Across Splits", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("report_figures/fig1_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_class_distribution.png")

# --- Fig 2: Sample original images ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, cls in enumerate(classes):
    cls_path = TRAIN_DIR / cls
    images = list(cls_path.glob("*.jpeg"))[:4] + list(cls_path.glob("*.jpg"))[:4]
    for col, img_path in enumerate(images[:4]):
        img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"{cls}", fontsize=10, fontweight='bold',
                                  color='#DD8452' if cls == 'PNEUMONIA' else '#4C72B0')
        axes[row, col].axis('off')

plt.suptitle("Sample Original X-Ray Images", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("report_figures/fig2_sample_images.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_sample_images.png")

# --- Fig 3: Original vs Augmented ---
# Pick one sample image from NORMAL
normal_imgs = list((TRAIN_DIR / "NORMAL").glob("*.jpeg")) + \
              list((TRAIN_DIR / "NORMAL").glob("*.jpg"))
sample_img_path = normal_imgs[0]
sample_img = load_img(sample_img_path, target_size=IMG_SIZE)
sample_arr = img_to_array(sample_img)
sample_arr = sample_arr.reshape((1,) + sample_arr.shape)

aug_gen = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1,
    horizontal_flip=True, brightness_range=[0.8,1.2]
)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
# Row 0: Original (repeated for reference)
for i in range(5):
    axes[0, i].imshow(sample_img)
    axes[0, i].set_title("Original" if i == 0 else "", fontsize=9)
    axes[0, i].axis('off')
axes[0, 0].set_ylabel("Original", fontsize=12, fontweight='bold', rotation=90, labelpad=10)

# Row 1: Augmented versions
for i, aug_batch in enumerate(aug_gen.flow(sample_arr, batch_size=1, seed=i)):
    axes[1, i].imshow(aug_batch[0].astype('uint8'))
    axes[1, i].set_title(f"Aug #{i+1}", fontsize=9)
    axes[1, i].axis('off')
    if i == 4:
        break
axes[1, 0].set_ylabel("Augmented", fontsize=12, fontweight='bold', rotation=90, labelpad=10)

plt.suptitle("Original vs. Augmented Images", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("report_figures/fig3_augmentation.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_augmentation.png")

# --- Fig 4: Pixel intensity histogram (before vs after normalization) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sample_raw = img_to_array(load_img(sample_img_path, target_size=IMG_SIZE))

axes[0].hist(sample_raw.flatten(), bins=50, color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].set_title("Before Normalization", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Pixel Value")
axes[0].set_ylabel("Frequency")

axes[1].hist((sample_raw/255.).flatten(), bins=50, color='#DD8452', alpha=0.8, edgecolor='white')
axes[1].set_title("After Normalization (÷255)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Pixel Value [0,1]")
axes[1].set_ylabel("Frequency")

for ax in axes:
    ax.spines[['top','right']].set_visible(False)

plt.suptitle("Pixel Intensity Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("report_figures/fig4_normalization.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_normalization.png")

# --- Fig 5: Train/Val/Test split pie chart ---
fig, ax = plt.subplots(figsize=(7, 5))
split_sizes = [total_train, total_val, total_test]
split_labels = [f'Train\n{total_train} ({total_train/total_all*100:.1f}%)',
                f'Validation\n{total_val} ({total_val/total_all*100:.1f}%)',
                f'Test\n{total_test} ({total_test/total_all*100:.1f}%)']
explode = (0.05, 0.05, 0.05)
ax.pie(split_sizes, labels=split_labels, explode=explode,
       colors=['#4C72B0','#55A868','#DD8452'],
       autopct='%1.1f%%', startangle=140,
       textprops={'fontsize': 11})
ax.set_title("Train / Validation / Test Split", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("report_figures/fig5_split_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_split_pie.png")

print("\n✅ All figures saved to report_figures/")
print("✅ Data preparation complete! Use these generators in your model training.")
print("\nSummary:")
print(f"  - Train samples   : {total_train}")
print(f"  - Val samples     : {total_val}")
print(f"  - Test samples    : {total_test}")
print(f"  - Image size      : {IMG_SIZE}")
print(f"  - Classes         : {classes}")
print(f"  - Class weights   : {dict(zip(classes, weights.round(3)))}")
