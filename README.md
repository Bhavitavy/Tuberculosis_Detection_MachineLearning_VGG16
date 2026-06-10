# Tuberculosis_Detection_MachineLearning_VGG16
# 🫁 TB Chest X-Ray Classifier (VGG16)

A deep learning system that classifies chest X-ray images as **Tuberculosis Positive** or **Normal (Negative)** using transfer learning on the **VGG16** architecture.

---

## 📁 Project Structure
```
tb_classifier/
├── tb_classifier.py     ← Main script (train + predict)
├── evaluate.py          ← Metrics, confusion matrix, Grad-CAM
├── setup_dataset.py     ← Auto-download & prepare dataset from Kaggle
```

## 📦 Dataset

### Option A — Kaggle (Recommended, ~1 000 images)
The recommended dataset is **"Tuberculosis (TB) Chest X-ray Dataset"** by Tawsifur Rahman on Kaggle.
1. Create a free Kaggle account → https://www.kaggle.com
2. Go to **Account → API → Create New Token** → saves `kaggle.json`
3. Place `kaggle.json` in:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourName>\.kaggle\kaggle.json`
4. Run the setup script:
This downloads, extracts, and splits the dataset automatically.

### Option B — Manual Setup

Organise your images like this:

```
data/
├── train/
│   ├── Normal/           ← healthy chest X-rays
│   └── Tuberculosis/     ← TB-positive X-rays
└── val/
    ├── Normal/
    └── Tuberculosis/
```

A **70 / 30** or **80 / 20** train/val split is recommended.  
Accepted image formats: `.jpg`, `.jpeg`, `.png`

---

## 🚀 Training


### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | *(required)* | Root folder containing `train/` and `val/` |
| `--epochs` | `30` | Maximum number of training epochs |
| `--fine_tune` | `4` | Number of last VGG16 layers to unfreeze |

**Examples:**

# Standard training
python tb_classifier.py train --data_dir data

# More epochs, full fine-tuning
python tb_classifier.py train --data_dir data --epochs 50 --fine_tune 8

# Freeze entire VGG16 base (fastest, good for small datasets)
python tb_classifier.py train --data_dir data --fine_tune 0

```
**What happens during training:**
- VGG16 weights (pretrained on ImageNet) are loaded
- Data augmentation is applied (rotation, flips, zoom, brightness)
- Best model (by validation AUC) is saved to `tb_vgg16_model.h5`
- Training curves are saved to `training_curves.png`
- Early stopping prevents overfitting (patience = 7 epochs)

---

## 🔬 Prediction

### Single Image

```
python tb_classifier.py predict --image path/to/xray.jpg
```

### Single Image with Custom Threshold

```
python tb_classifier.py predict --image xray.jpg --threshold 0.4
```

### Batch Prediction (Folder)

```
python tb_classifier.py predict_batch --folder path/to/xray_folder/
```

---

## 📊 Evaluation

### Classification Report + Confusion Matrix

```
python evaluate.py --data_dir data --model tb_vgg16_model.h5
```

Outputs:
- Precision, Recall, F1-score per class
- Confusion matrix image → `confusion_matrix.png`

### Grad-CAM Heatmap (Visual Explainability)

Shows **which regions of the lung** the model focused on:

```
python evaluate.py --gradcam --image path/to/xray.jpg


Output image saved to `gradcam_result.png`.

---

## 📈 Expected Performance

On the Kaggle TB dataset with default settings:

| Metric | Typical Value |
|--------|--------------|
| Validation Accuracy | ~92–96% |
| AUC | ~0.96–0.99 |
| Sensitivity (Recall for TB) | ~90–95% |
| Specificity | ~93–97% |

Results vary depending on dataset size, split, and training epochs.

---

## 🗂️ Output Files

| File | Description |
|------|-------------|
| `tb_vgg16_model.h5` | Trained model weights |
| `training_curves.png` | Accuracy / Loss / AUC plots |
| `prediction_result.png` | Prediction image with label |
| `confusion_matrix.png` | Evaluation confusion matrix |
| `gradcam_result.png` | Grad-CAM explainability overlay |

---

## 📚 References

- Simonyan & Zisserman, *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (VGG, 2014)
- Rahman et al., *"Reliable Tuberculosis Detection using Chest X-ray with Deep Learning"* (2020)
- Dataset: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
