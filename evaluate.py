"""
Evaluation & Grad-CAM Visualization
=====================================
• Full classification report + confusion matrix on a test set
• Grad-CAM heatmap — highlights which lung regions drove the prediction

Usage:
    python evaluate.py --data_dir data --model tb_vgg16_model.h5
    python evaluate.py --gradcam --image chest.png --model tb_vgg16_model.h5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["Normal", "Tuberculosis"]


# ─────────────────────────────────────────
#  FULL EVALUATION
# ─────────────────────────────────────────
def evaluate(data_dir: str, model_path: str, split: str = "val"):
    model = load_model(model_path)
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        os.path.join(data_dir, split),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False
    )

    probs = model.predict(gen, verbose=1).flatten()
    preds = (probs >= 0.5).astype(int)
    true  = gen.classes

    print("\n📊  Classification Report")
    print("─" * 50)
    print(classification_report(true, preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — TB Classifier", fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("🖼   Saved → confusion_matrix.png")
    plt.show()


# ─────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):
    """Generate Grad-CAM heatmap for the last VGG16 conv layer."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary sigmoid output, gradient w.r.t. prediction
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def gradcam_visualize(image_path: str, model_path: str, threshold: float = 0.5):
    model = load_model(model_path)

    original_img = load_img(image_path, target_size=IMG_SIZE)
    img_array    = img_to_array(original_img) / 255.0
    img_input    = np.expand_dims(img_array, axis=0)

    prob  = float(model.predict(img_input, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob >= threshold else CLASS_NAMES[0]
    conf  = prob if prob >= threshold else 1 - prob

    heatmap = make_gradcam_heatmap(img_input, model)

    # Resize heatmap to match image
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = tf.image.resize(heatmap_colored, IMG_SIZE).numpy()

    # Superimpose
    superimposed = heatmap_colored * 0.4 + img_array
    superimposed = np.clip(superimposed / superimposed.max(), 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM  —  {'🔴 TB POSITIVE' if label == 'Tuberculosis' else '🟢 NORMAL'}  "
        f"({conf*100:.1f}% confidence)",
        fontsize=13, fontweight="bold",
        color="#e74c3c" if label == "Tuberculosis" else "#27ae60"
    )

    axes[0].imshow(original_img);         axes[0].set_title("Original X-Ray");   axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet");  axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(superimposed);         axes[2].set_title("Overlay");           axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150, bbox_inches="tight")
    print(f"\n✅  Grad-CAM saved → gradcam_result.png")
    print(f"   Prediction: {label}  |  Confidence: {conf*100:.1f}%")
    plt.show()


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate TB model / generate Grad-CAM")
    parser.add_argument("--model",    default="tb_vgg16_model.h5", help="Saved model path")
    parser.add_argument("--data_dir", help="Dataset root (required for --eval)")
    parser.add_argument("--split",    default="val", choices=["train", "val", "test"])
    parser.add_argument("--gradcam",  action="store_true", help="Run Grad-CAM visualisation")
    parser.add_argument("--image",    help="Image path for Grad-CAM")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.gradcam:
        if not args.image:
            parser.error("--gradcam requires --image <path>")
        gradcam_visualize(args.image, args.model, args.threshold)
    else:
        if not args.data_dir:
            parser.error("Evaluation requires --data_dir <path>")
        evaluate(args.data_dir, args.model, args.split)


if __name__ == "__main__":
    main()
