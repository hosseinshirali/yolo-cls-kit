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
- [ONNX Export (Deployment)](#-onnx-export-deployment)
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

## 📦 ONNX Export (Deployment)

After selecting your best model, you can export it to **ONNX format** for fast, framework-agnostic inference in production (ONNX Runtime, TensorRT, OpenVINO, web browsers, etc.).

### Quick Start

```bash
# Basic export
python export_onnx.py --model runs/train_results/weights/best.pt

# Custom image size
python export_onnx.py --model best.pt --imgsz 224

# FP16 half precision (smaller & faster, requires CUDA)
python export_onnx.py --model best.pt --half

# Dynamic batch size (for variable batch inference)
python export_onnx.py --model best.pt --dynamic

# Simplify the ONNX graph + verify the export
python export_onnx.py --model best.pt --simplify --verify
```

### All Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--model` | Path to your trained `.pt` model **(required)** | — |
| `--imgsz` | Input image size | `640` |
| `--output` | Custom output path for `.onnx` file | Same dir as model |
| `--half` | Export with FP16 half precision | `False` |
| `--dynamic` | Enable dynamic batch axis | `False` |
| `--simplify` | Simplify ONNX graph (requires `onnxsim`) | `False` |
| `--opset` | ONNX opset version | Auto |
| `--verify` | Run a quick inference check after export | `False` |

### Running the Exported Model

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("best.onnx")

# Prepare input (adjust size to match --imgsz used during export)
img = Image.open("test_image.jpg").resize((640, 640))
img_np = np.array(img).astype(np.float32) / 255.0
img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
img_np = np.expand_dims(img_np, axis=0)    # Add batch dim

# Inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_np})
predicted_class = np.argmax(outputs[0])
print(f"Predicted class index: {predicted_class}")
```

### Optional Dependencies

```bash
pip install onnx onnxruntime  # or onnxruntime-gpu for GPU inference
pip install onnxsim            # only needed for --simplify
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest features.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the file for details.

---

**Built with ❤️ using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).**
