# YOLO Classification Pipeline

A comprehensive pipeline for training and evaluating YOLO classification models using different dataset organization methods.

## Features

- **Multiple Dataset Input Methods:**
  1. Class-organized dataset with text split files
  2. Pre-split dataset (train/val/test folders)
  3. Flat dataset with Excel metadata
- **YAML Configuration Support:** Define all parameters in a config file
- **Comprehensive Logging:** Automatic logging to file and console
- **Input Validation:** Pre-flight checks to catch errors early
- **Automated Dataset Preparation:** Handles various dataset formats
- **Model Training:** Customizable parameters and augmentations
- **Post-processing and Evaluation:**
  - Confusion matrix visualization
  - Classification reports
  - Prediction results saved to Excel
- **Explainability:** EigenCAM, GradCAM, and GradCAM++ heatmaps with flexible comparison utilities
- **Memory Optimization:** Automatic cleanup after intensive operations
- **Progress Tracking:** Rich progress bars for long operations
- **Support for Multiple YOLO Models:** YOLOv8, YOLO11, and custom models

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

```text
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

```text
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

```text
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

### Quick Start with Configuration File (Recommended)

The easiest way to run the pipeline is using a YAML configuration file:

```bash
# Create a config file from the example
cp config_example.yaml my_config.yaml

# Edit my_config.yaml with your settings

# Run the pipeline
python run_pipeline.py --config my_config.yaml
```

**Example config_example.yaml:**

```yaml
dataset:
  presplit_root: "dataset/smnk_split"

output:
  output_dir: "output_dir/run1"

model:
  name: "yolov8n-cls.pt"
  device: 0

training:
  epochs: 100
  imgsz: 640
  batch: 16
  patience: 10
  dropout: 0.3

augmentations:
  fliplr: 0.5
  scale: 0.5

postprocessing:
  sample_size: 30  # Number of images for EigenCAM visualization
```


### Command Line Usage

You can also run the pipeline directly from command line:

```bash
# Method 1: Class-organized with text split files
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir --model yolov8s-cls.pt --sample-size 50

# Method 2: Pre-split dataset
python run_pipeline.py output_dir --presplit-input path/to/presplit_dataset --model yolov8l-cls.pt

# Method 3: Flat dataset with Excel metadata
python run_pipeline.py output_dir --excel-input path/to/images path/to/metadata.xlsx --model yolov8n-cls.pt
```

### Training Parameters

Customize training parameters:

```bash
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir \
  --epochs 100 --imgsz 224 --batch 32 --patience 15 --dropout 0.3 --device 0 \
  --sample-size 50
```

### Augmentation Parameters

Specify augmentation parameters (all are optional):

```bash
python run_pipeline.py output_dir --splits-input path/to/images path/to/splits_dir \
  --hsv_h 0.015 --flipud 0.5 --fliplr 0.7 --scale 0.6
```

## CAM Heatmaps

The repository ships a dedicated CLI, `generate_eigencam.py`, for creating YOLO classification heatmaps with EigenCAM, GradCAM, and GradCAM++.

### Prerequisites

1. Activate the project environment: `yolo_cls_env\Scripts\activate` (Windows) or `source yolo_cls_env/bin/activate` (Linux/Mac).
2. Run `python generate_eigencam.py --help` from the repository root to view all options.

### Single Image Visualisation

```bash
python generate_eigencam.py \
  --model path/to/best.pt \
  --image path/to/image.jpg \
  --output heatmap
```

Key defaults:

- `--target` selects the layer index (negative values count from the end, default `-2`).
- `--method` chooses the CAM backend (`eigencam`, `gradcam`, `gradcam++`).

### Batch Processing

```bash
python generate_eigencam.py \
  --model path/to/best.pt \
  --input path/to/image_folder \
  --output heatmap \
  --method gradcam++ \
  --max-images 200
```

`--input` accepts any directory (recursively scanned). Use `--max-images` to limit processing.

### Comparing Layers

```bash
python generate_eigencam.py \
  --model path/to/best.pt \
  --image path/to/image.jpg \
  --output heatmap \
  --compare \
  --compare-layers -1 -3 -5
```

`--compare` renders one figure per image with side-by-side overlays for the requested layer indices.

### Comparing Methods

```bash
python generate_eigencam.py \
  --model path/to/best.pt \
  --image path/to/image.jpg \
  --output heatmap \
  --compare-methods \
  --method-target -3
```

This produces EigenCAM vs GradCAM++ overlays for the chosen layer. Omit `--method-target` to reuse the `--target` layer.

### Output

- Results land in the directory passed via `--output` (created if missing).
- Filenames follow `eigencam_<image_name>.png`, `comparison_<image_name>.png`, or `method_comparison_<image_name>.png` depending on the mode.
- Logs report total successes and failures; enable `--verbose` for detailed layer listings.

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

```text
output_dir/
  ├── train_results/              # Training results
  │   ├── weights/                # Trained model weights
  │   │   ├── best.pt             # Best model weights (use this for evaluation!)
  │   │   └── last.pt             # Last epoch weights
  │   ├── args.yaml               # Training arguments
  │   ├── results.csv             # Training metrics per epoch
  │   ├── val_results/            # Validation evaluation (FOR BENCHMARKING!)
  │   │   ├── validation_metrics.txt  # Top-1 and Top-5 accuracy
  │   │   ├── confusion_matrix.png    # Validation confusion matrix
  │   │   └── ...                     # Other validation plots
  │   ├── test_results/           # Test evaluation (detailed analysis)
  │   │   ├── labels/             # Prediction label files
  │   │   ├── confusion_matrix.png    # Test confusion matrix
  │   │   ├── classification_report.txt  # Detailed test metrics
  │   │   ├── predictions.xlsx         # Test predictions with ground truth
  │   │   └── eigencam/                # EigenCAM visualizations
  │   │       ├── cam_image1.jpg
  │   │       └── ...
  │   └── pipeline_*.log          # Execution log (timestamped)
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

## Logging

The pipeline automatically creates detailed logs:

- **Console Output:** INFO level messages displayed during execution
- **Log File:** Detailed logs saved to `yolo_pipeline_YYYYMMDD_HHMMSS.log`
- **Log Levels:** DEBUG, INFO, WARNING, ERROR

To view logs:

```bash
# View latest log
tail -f yolo_pipeline_*.log

# Search for errors
grep ERROR yolo_pipeline_*.log

# Search for warnings
grep WARNING yolo_pipeline_*.log
```

## Evaluation Methodology

The pipeline performs two separate evaluations after training:

### 1. Validation Evaluation (STEP 1)

**Purpose:** For model benchmarking and comparison  
**Method:** Uses YOLO's built-in `.val()` method  
**Output:** `val_results/validation_metrics.txt`  
**Metrics:**

- Top-1 Accuracy (main metric for benchmarking)
- Top-5 Accuracy
- Confusion matrix

⚠️ **Important:** Always use validation results for model benchmarking, NOT test results!

### 2. Test Evaluation (STEP 2)

**Purpose:** Detailed analysis and model interpretability  
**Method:** Custom inference + sklearn metrics + EigenCAM  
**Output:** `test_results/`  
**Includes:**

- Confusion matrix
- Classification report (precision, recall, F1-score per class)
- Predictions Excel file
- EigenCAM visualizations

## Output Files

After training and evaluation, you'll find:

```text
output_dir/
├── train_results/
│   ├── weights/
│   │   ├── best.pt                      # Best model checkpoint
│   │   └── last.pt                      # Last epoch checkpoint
│   ├── val_results/                     # ⭐ USE THESE FOR BENCHMARKING
│   │   ├── validation_metrics.txt       # Top-1 and Top-5 accuracy
│   │   ├── confusion_matrix.png         # Validation confusion matrix
│   │   └── ...                          # Other YOLO plots
│   ├── test_results/                    # Detailed analysis
│   │   ├── labels/                      # Prediction label files
│   │   ├── confusion_matrix.png         # Test confusion matrix
│   │   ├── classification_report.txt    # Per-class metrics
│   │   ├── predictions.xlsx             # All predictions with ground truth
│   │   └── eigencam/                    # Model interpretability
│   │       ├── cam_image1.jpg
│   │       └── ...
│   ├── args.yaml                        # Training arguments
│   ├── results.csv                      # Training metrics per epoch
│   └── pipeline_*.log                   # Execution log
```

## Troubleshooting

### CUDA Issues

- If you encounter CUDA errors, verify your CUDA and PyTorch versions are compatible
- Try setting `device=-1` to force CPU usage if GPU issues persist

### Memory Errors

- Reduce batch size (`--batch`) for large models or limited memory
- Lower image size (`--imgsz`) to reduce memory requirements
- Reduce EigenCAM sample size (`--sample-size`)

### Missing Predictions

- If more than 10% of test images have no predictions, the pipeline will raise an error
- Check that your model is compatible with the image formats
- Verify test images are valid and readable

### Input Validation Errors

- The pipeline validates all inputs before processing
- Check error messages for specific missing files or directories
- Verify Excel files have required columns: `filename`, `label`, `phase`

## License

[MIT License](LICENSE)
