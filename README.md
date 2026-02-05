# yolo-cls-kit 🛠️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%2F11-green)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

**A production-ready, highly configurable pipeline for training YOLO classification models.**

---

## 📖 Table of Contents

- [Why use this pipeline?](#-why-use-this-pipeline)
- [Features](#-features)
- [Getting Started](#-getting-started)
  - [Installation](#1-installation)
  - [Dataset Preparation](#2-dataset-preparation)
  - [Configuration](#3-configuration)
  - [Training](#4-training)
- [Validation & Visualization](#-validation--visualization)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [License](#-license)

---

## 💡 Why use this pipeline?

While Ultralytics provides a powerful CLI, managing complex experiments, reproducibility, and diverse dataset formats can be challenging. This repository wraps the Ultralytics engine into a **robust, professional workflow** designed for:

1. **🚀 Zero-Code Training**: Control every aspect of training via a single YAML configuration file. No need to edit Python scripts.
2. **📂 Flexible Data Handling**: Supports **Excel files** for metadata, pre-split folders, or simple text file splits. Perfect for custom datasets.
3. **🧠 Advanced Analysis**: Automatically generates **EigenCAM** visualizations to show exactly *where* your model is looking.
4. **⚡ Automated Optimization**: Built-in support for **Optuna** and **Ray Tune** for hyperparameter hyper-tuning.

---

## ✨ Features

- **Models**: Native support for YOLOv8 and YOLO11 (n, s, m, l, x).
- **Augmentations**: Fine-grained control over 15+ augmentation parameters (MixUp, Mosaic, HSV, etc.) directly from config.
- **Reproducibility**: Config-driven execution ensures every experiment is repeatable.
- **Visuals**: Auto-generates confusion matrices, training curves, and attention maps.

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/hosseinshirali/yolo-cls-kit.git
cd yolo-cls-kit

# Create environment (Recommended)
conda create -n yolo_cls_kit python=3.10
conda activate yolo_cls_kit

# Install dependencies
pip install -r requirements_pipeline.txt
```

### 2. Dataset Preparation

Choose the method that fits your data:

| Method | Best For | Structure |
| :--- | :--- | :--- |
| **A. Folders** | Standard Datasets | `train/class_a`, `val/class_a` |
| **B. Text Splits** | Large Datasets | Single image folder + `train.txt` list |
| **C. Excel/CSV** | Scientific Data | Single image folder + `.xlsx` with labels |

**Example config for Excel method:**

```yaml
dataset:
  image_root: "./data/images"
  excel_path: "./data/metadata.xlsx"  # Columns: filename, label, phase
```

### 3. Configuration

1. **Create your config:**

    ```bash
    cp config_template.yaml config.yaml
    ```

2. **Edit `config.yaml`:**
    Set your dataset path, model size, and hyperparameters.

### 4. Training

**Linux / Mac:**

```bash
./train.sh --config config.yaml
```

**Windows:**

```cmd
python run_pipeline.py --config config.yaml
```

---

## 🔍 Validation & Visualization

After training, the pipeline automatically:

1. Evaluates on the test set.
2. Saves the best model to `runs/your_experiment/weights/best.pt`.
3. Generates **EigenCAM** heatmaps for misclassified images (if enabled).

---

## 🔧 Advanced Usage

### Hyperparameter Optimization (HPO)

Enable Optuna in `config.yaml` to auto-discover the best learning rate, momentum, and augmentation settings:

```yaml
optimization:
  enabled: true
  use_optuna: true
  iterations: 50  # Number of trials
```

### Resuming Training

Interrupted? Just set `resume: True` in your config and point to the same output directory.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest features.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the file for details.

---

**Built with ❤️ using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).**
