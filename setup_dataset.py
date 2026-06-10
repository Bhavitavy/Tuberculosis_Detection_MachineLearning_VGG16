"""
Dataset Setup Utility
=====================
Downloads the TB Chest X-Ray dataset from Kaggle and organises it into the
train/val split expected by tb_classifier.py.

Usage:
    python setup_dataset.py

Prerequisites:
    pip install kaggle
    Place your kaggle.json API key in ~/.kaggle/kaggle.json
"""

import os
import shutil
import random
from pathlib import Path

DATASET_SLUG = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
RAW_DIR      = "raw_data"
DATA_DIR     = "data"
VAL_SPLIT    = 0.2
SEED         = 42
CLASSES      = ["Normal", "Tuberculosis"]


def download_dataset():
    print("📥  Downloading TB dataset from Kaggle …")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d {DATASET_SLUG} -p {RAW_DIR} --unzip")
    print("✅  Download complete.")


def organise_dataset():
    """
    Finds all class images in raw_data/ and splits them into
    data/train/  and  data/val/  subfolders.
    """
    random.seed(SEED)
    for split in ["train", "val"]:
        for cls in CLASSES:
            Path(f"{DATA_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        # Search recursively for images belonging to this class
        all_imgs = list(Path(RAW_DIR).rglob(f"*/{cls}/*.png")) + \
                   list(Path(RAW_DIR).rglob(f"*/{cls}/*.jpg")) + \
                   list(Path(RAW_DIR).rglob(f"*/{cls}/*.jpeg"))

        if not all_imgs:
            print(f"⚠️   No images found for class '{cls}'. Check raw_data/ structure.")
            continue

        random.shuffle(all_imgs)
        n_val   = int(len(all_imgs) * VAL_SPLIT)
        val_set = set(str(p) for p in all_imgs[:n_val])

        copied_train, copied_val = 0, 0
        for img in all_imgs:
            dest_split = "val" if str(img) in val_set else "train"
            dest = Path(DATA_DIR) / dest_split / cls / img.name
            shutil.copy2(img, dest)
            if dest_split == "train":
                copied_train += 1
            else:
                copied_val += 1

        print(f"  {cls:<15}  train: {copied_train}  |  val: {copied_val}")

    print(f"\n✅  Dataset ready at '{DATA_DIR}/'")
    print(f"    Run training with:  python tb_classifier.py train --data_dir {DATA_DIR}")


if __name__ == "__main__":
    if not shutil.which("kaggle"):
        print("❌  Kaggle CLI not found. Run:  pip install kaggle")
        print("    Then add your API key to ~/.kaggle/kaggle.json")
        raise SystemExit(1)

    download_dataset()
    organise_dataset()
