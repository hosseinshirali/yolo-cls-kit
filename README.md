# YOLO Classification Pipeline

A comprehensive pipeline for training and evaluating YOLO classification models using different dataset organization methods.

## Features

- Support for three dataset input methods:
  1. Class-organized dataset with text split files
  2. Pre-split dataset (train/val/test folders)
  3. Flat dataset with Excel metadata
- Automated dataset preparation
- Model training with customizable parameters
- Post-processing and evaluation
- Explainability with EigenCAM
- Support for different YOLO models (YOLOv8, YOLO11, etc.)

## Setup

### CUDA Setup

For GPU acceleration, install the appropriate CUDA version for your PyTorch/YOLO requirements:

1. Check your graphics card and driver version:
   ```bash
   # For NVIDIA GPUs
   nvidia-smi
   ```

2. Install the compatible CUDA version:
   - For YOLOv8 (PyTorch 2.0+): CUDA 11.7 or CUDA 12.1
   - For YOLO11: CUDA 12.1

3. Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### Virtual Environment

Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv yolo_cls_env

# Activate on Windows
yolo_cls_env\Scripts\activate

# Activate on Linux/MacOS
source yolo_cls_env/bin/activate
```

### Dependencies

Install the required packages:

```bash
# Install PyTorch with the appropriate CUDA version
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements_pipeline.txt
```

## Dataset Organization

The pipeline supports three different ways to organize your dataset:

### 1. Class-organized with text split files

- Images organized in class folders
- Three text files (train.txt, val.txt, test.txt) containing image names for each split

Example structure:
```
dataset/
  ├── class1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class2/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── train.txt
  ├── val.txt
  └── test.txt
```

### 2. Pre-split dataset (train/val/test folders)

- Images already organized in train/val/test folders
- Each split folder contains class subfolders

Example structure:
```
dataset/
  ├── train/
  │   ├── class1/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── class2/
  │       ├── image1.jpg
  │       └── ...
  ├── val/
  │   ├── class1/
  │   └── class2/
  └── test/
      ├── class1/
      └── class2/
```

### 3. Flat dataset with Excel metadata

- All images in a single folder
- Excel file containing filename, label, and phase columns

Example structure:
```
dataset/
  ├── all/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── metadata.xlsx
```

Excel file format:
| filename | label | phase |
|----------|-------|-------|
| image1.jpg | class1 | train |
| image2.jpg | class2 | test |
| ... | ... | ... |

## Usage

### Command Line

Run the pipeline from the command line using `run_pipeline.py`:

```bash
# Method 1: Class-organized with text split files
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir --model yolov8s-cls.pt

# Method 2: Pre-split dataset
python run_pipeline.py output_dir --presplit-input path/to/presplit_dataset --model yolov8l-cls.pt

# Method 3: Flat dataset with Excel metadata
python run_pipeline.py output_dir --excel-input path/to/images path/to/metadata.xlsx --model yolov8n-cls.pt
```

### Training Parameters

Customize training parameters:

```bash
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir \
  --epochs 100 --imgsz 224 --batch 32 --patience 15 --dropout 0.3 --device 0
```

### Augmentation Parameters

Specify augmentation parameters (all are optional):

```bash
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir \
  --hsv_h 0.015 --flipud 0.5 --fliplr 0.7 --scale 0.6
```

## Python API

You can also use the pipeline in your Python code:

```python
from yolo_complete_cls_pipeline import main

# Define parameters
training_parameter = {
    "epochs": 50,
    "imgsz": 224,
    "batch": 16,
    "patience": 10,
    "device": 0,
    "dropout": 0.3
}

# Optional augmentations
augmentation_params = {
    "flipud": 0.5,
    "fliplr": 0.7,
    "scale": 0.6
}

# Method 1: Class-organized with text split files
splits = {
    "train": "path/to/train.txt",
    "val": "path/to/val.txt",
    "test": "path/to/test.txt"
}
main("path/to/images", "output_dir", splits, training_parameter, 
     model_name="yolov8s-cls.pt", augmentations=augmentation_params)

# Method 2: Pre-split dataset
main("path/to/presplit_dataset", "output_dir", training_parameter=training_parameter, 
     model_name="yolov8l-cls.pt", augmentations=augmentation_params)

# Method 3: Flat dataset with Excel metadata
main("path/to/images", "output_dir", training_parameter=training_parameter, 
     excel_path="path/to/metadata.xlsx", model_name="yolov8n-cls.pt", 
     augmentations=augmentation_params)
```

## Output Structure

The pipeline creates the following output structure:

```
output_dir/
  ├── train_results/            # Training results
  │   ├── weights/              # Trained model weights
  │   │   ├── best.pt           # Best model weights
  │   │   └── last.pt           # Last model weights
  │   └── ...                   # Training logs and plots
  ├── test_results/             # Testing results
  │   └── labels/               # Prediction labels
  ├── confusion_matrix.png      # Confusion matrix visualization
  ├── classification_report.txt # Detailed metrics
  └── cam_*.jpg                 # EigenCAM visualizations
```

## Models

The pipeline supports various YOLO classification models:

- YOLOv8:
  - yolov8n-cls.pt (nano)
  - yolov8s-cls.pt (small)
  - yolov8m-cls.pt (medium)
  - yolov8l-cls.pt (large)
  - yolov8x-cls.pt (xlarge)
- YOLO11:
  - yolo11n.pt (nano)
  - etc.

## Troubleshooting

### CUDA Issues
- If you encounter CUDA errors, verify your CUDA and PyTorch versions are compatible
- Try setting `device=-1` to force CPU usage if GPU issues persist

### Memory Errors
- Reduce batch size (`--batch`) for large models or limited memory
- Lower image size (`--imgsz`) to reduce memory requirements

## License

[MIT License](LICENSE) 