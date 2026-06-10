"""
TB Chest X-Ray Classifier using VGG16
=====================================
Classifies chest X-ray images as Tuberculosis Positive or Negative.
Supports training on custom datasets and prediction on new images.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE     = (224, 224)   # VGG16 input size
BATCH_SIZE   = 16
EPOCHS       = 30
LEARNING_RATE = 1e-4
MODEL_PATH   = "tb_vgg16_model.h5"
CLASS_NAMES  = ["Normal", "Tuberculosis"]   # 0 = Normal, 1 = TB


# ─────────────────────────────────────────
#  BUILD MODEL
# ─────────────────────────────────────────
def build_model(fine_tune_layers: int = 4) -> tf.keras.Model:
    """
    Build a VGG16-based transfer learning model.

    Args:
        fine_tune_layers: Number of last VGG16 layers to unfreeze for fine-tuning.
                          Set to 0 to freeze the entire base (faster training).

    Returns:
        Compiled Keras model
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    # Unfreeze last N layers for fine-tuning
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")   # Binary classification
    ], name="TB_VGG16_Classifier")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    return model


# ─────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────
def create_data_generators(data_dir: str):
    """
    Create train/validation data generators with augmentation.

    Expected folder structure:
        data_dir/
            train/
                Normal/       ← chest X-rays without TB
                Tuberculosis/ ← chest X-rays with TB
            val/
                Normal/
                Tuberculosis/

    Args:
        data_dir: Root directory containing train/ and val/ subdirectories.

    Returns:
        (train_gen, val_gen) tuple of ImageDataGenerators
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False
    )

    print(f"\n📂 Class mapping: {train_gen.class_indices}")
    print(f"   Train samples : {train_gen.samples}")
    print(f"   Val   samples : {val_gen.samples}")
    return train_gen, val_gen


# ─────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────
def train(data_dir: str, epochs: int = EPOCHS, fine_tune: int = 4):
    """
    Train the VGG16 model on chest X-ray data.

    Args:
        data_dir   : Root data directory (must contain train/ and val/).
        epochs     : Total training epochs.
        fine_tune  : Number of VGG16 layers to fine-tune.
    """
    print("\n" + "="*55)
    print("  🫁  TB Classifier — Training Mode")
    print("="*55)

    train_gen, val_gen = create_data_generators(data_dir)
    model = build_model(fine_tune_layers=fine_tune)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\n🚀 Starting training for up to {epochs} epochs …\n")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    _plot_history(history)
    _evaluate(model, val_gen)
    print(f"\n✅ Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────
def predict(image_path: str, model_path: str = MODEL_PATH, threshold: float = 0.5):
    """
    Predict TB positive / negative for a single chest X-ray image.

    Args:
        image_path : Path to the chest X-ray image file.
        model_path : Path to the saved .h5 model.
        threshold  : Decision threshold (default 0.5).
    """
    print("\n" + "="*55)
    print("  🫁  TB Classifier — Prediction Mode")
    print("="*55)

    if not os.path.exists(model_path):
        print(f"❌  Model not found at '{model_path}'.")
        print("    Please train the model first:  python tb_classifier.py train --data_dir <path>")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌  Image not found at '{image_path}'")
        sys.exit(1)

    print(f"\n📥  Loading model from: {model_path}")
    model = load_model(model_path)

    # Preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)

    # Run inference
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
    confidence = prob if prob >= threshold else 1 - prob

    # Display result
    print(f"\n{'─'*45}")
    print(f"  Image     : {Path(image_path).name}")
    print(f"  Prediction: {'🔴 TUBERCULOSIS POSITIVE' if label == 'Tuberculosis' else '🟢 NORMAL (TB Negative)'}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print(f"  Raw score : {prob:.4f}  (threshold = {threshold})")
    print(f"{'─'*45}\n")

    # Show image with result
    _show_prediction(image_path, label, confidence, prob)

    return label, confidence


def predict_batch(folder_path: str, model_path: str = MODEL_PATH, threshold: float = 0.5):
    """
    Predict TB classification for all images in a folder.

    Args:
        folder_path: Folder containing chest X-ray images.
        model_path : Path to the saved .h5 model.
        threshold  : Decision threshold.
    """
    if not os.path.exists(model_path):
        print(f"❌  Model not found: '{model_path}'")
        sys.exit(1)

    model = load_model(model_path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"❌  No images found in '{folder_path}'")
        sys.exit(1)

    print(f"\n📂  Found {len(images)} image(s) in '{folder_path}'\n")
    print(f"{'File':<35} {'Prediction':<25} {'Confidence':>12}")
    print("─" * 75)

    results = []
    for img_path in sorted(images):
        img = load_img(str(img_path), target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
        conf  = prob if prob >= threshold else 1 - prob
        icon  = "🔴" if label == "Tuberculosis" else "🟢"
        print(f"{img_path.name:<35} {icon} {label:<23} {conf*100:>10.1f}%")
        results.append({"file": img_path.name, "label": label, "confidence": conf, "prob": prob})

    tb_count = sum(1 for r in results if r["label"] == "Tuberculosis")
    print("─" * 75)
    print(f"\n  Summary: {tb_count} TB Positive  |  {len(results) - tb_count} Normal  (out of {len(results)} images)")
    return results


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def _plot_history(history):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History — TB VGG16 Classifier", fontsize=14, fontweight="bold")

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(history.history[metric],       label="Train", linewidth=2)
        ax.plot(history.history[f"val_{metric}"], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title);  ax.set_xlabel("Epoch");  ax.legend();  ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("📊  Training curves saved → training_curves.png")
    plt.show()


def _evaluate(model, val_gen):
    """Print final evaluation metrics on validation set."""
    print("\n📊  Validation Metrics:")
    results = model.evaluate(val_gen, verbose=0)
    names = model.metrics_names
    for name, val in zip(names, results):
        print(f"   {name:<12}: {val:.4f}")


def _show_prediction(image_path, label, confidence, raw_prob):
    """Display the image with prediction overlay."""
    img = load_img(image_path)
    color = "#e74c3c" if label == "Tuberculosis" else "#2ecc71"

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"{'🔴 TB POSITIVE' if label == 'Tuberculosis' else '🟢 NORMAL'}\n"
        f"Confidence: {confidence*100:.1f}%  |  Score: {raw_prob:.3f}",
        fontsize=13, fontweight="bold", color=color, pad=10
    )
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    print("🖼   Prediction image saved → prediction_result.png")
    plt.show()


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🫁  TB Chest X-Ray Classifier using VGG16",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── train ──
    train_p = subparsers.add_parser("train", help="Train the model on a labelled dataset")
    train_p.add_argument("--data_dir",   required=True, help="Root dir with train/ and val/ subfolders")
    train_p.add_argument("--epochs",     type=int, default=EPOCHS,   help=f"Training epochs (default {EPOCHS})")
    train_p.add_argument("--fine_tune",  type=int, default=4,        help="VGG16 layers to unfreeze (default 4)")

    # ── predict ──
    pred_p = subparsers.add_parser("predict", help="Predict a single X-ray image")
    pred_p.add_argument("--image",      required=True, help="Path to chest X-ray image")
    pred_p.add_argument("--model",      default=MODEL_PATH, help=f"Saved model path (default: {MODEL_PATH})")
    pred_p.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold (default 0.5)")

    # ── predict_batch ──
    batch_p = subparsers.add_parser("predict_batch", help="Predict all images in a folder")
    batch_p.add_argument("--folder",    required=True, help="Folder containing X-ray images")
    batch_p.add_argument("--model",     default=MODEL_PATH, help=f"Saved model path (default: {MODEL_PATH})")
    batch_p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.epochs, args.fine_tune)

    elif args.mode == "predict":
        predict(args.image, args.model, args.threshold)

    elif args.mode == "predict_batch":
        predict_batch(args.folder, args.model, args.threshold)


if __name__ == "__main__":
    main()
