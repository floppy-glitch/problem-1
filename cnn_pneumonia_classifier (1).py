"""
Problem Set 01: Chest X-Ray Pneumonia Classification using CNN
Dataset: Pediatric chest X-ray images (Pneumonia vs Normal)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 1           # binary → sigmoid output
LR          = 1e-4

BASE_DIR   = "chest_xray"          # update to your extracted dataset root
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
TEST_DIR   = os.path.join(BASE_DIR, "test")

# ─────────────────────────────────────────────
# 2. Data Generators
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Class indices:", train_gen.class_indices)
# Expected: {'NORMAL': 0, 'PNEUMONIA': 1}

# ─────────────────────────────────────────────
# 3. Handle Class Imbalance
# ─────────────────────────────────────────────
total   = train_gen.samples
n_normal    = train_gen.classes.tolist().count(0)
n_pneumonia = train_gen.classes.tolist().count(1)

weight_normal    = total / (2 * n_normal)
weight_pneumonia = total / (2 * n_pneumonia)
class_weights = {0: weight_normal, 1: weight_pneumonia}
print(f"Class weights → Normal: {weight_normal:.3f}  |  Pneumonia: {weight_pneumonia:.3f}")

# ─────────────────────────────────────────────
# 4. CNN Architecture
# ─────────────────────────────────────────────
def build_cnn(input_shape=(150, 150, 3)):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.35),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.35),

        # Classifier head
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),  # binary output
    ])
    return model

model = build_cnn()
model.summary()

# ─────────────────────────────────────────────
# 5. Compile
# ─────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.Precision(name="precision")],
)

# ─────────────────────────────────────────────
# 6. Callbacks
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_auc", patience=6, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint("best_cnn_model.keras", monitor="val_auc", save_best_only=True, mode="max"),
]

# ─────────────────────────────────────────────
# 7. Training
# ─────────────────────────────────────────────
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 8. Evaluation on Test Set
# ─────────────────────────────────────────────
print("\n── Test Set Evaluation ──")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

# Predictions
test_gen.reset()
y_pred_prob = model.predict(test_gen, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# ─────────────────────────────────────────────
# 9. Plots
# ─────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history.history["auc"], label="Train AUC")
    axes[2].plot(history.history["val_auc"], label="Val AUC")
    axes[2].set_title("AUC over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved: training_history.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


plot_training_history(history)
plot_confusion_matrix(y_true, y_pred)

# ─────────────────────────────────────────────
# 10. Save Final Model
# ─────────────────────────────────────────────
model.save("cnn_pneumonia_final.keras")
print("\nModel saved to cnn_pneumonia_final.keras")
