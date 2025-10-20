#!/usr/bin/env python3
import argparse
from pathlib import Path
from yolo_complete_cls_pipeline import main

"""
This script is designed to run the YOLO classification pipeline.

The script supports three different ways of providing dataset input:

1. Structured dataset with split files:
   - Provide an image root directory containing class folders
   - Provide a splits root directory containing train.txt, val.txt, and test.txt files

2. Pre-split dataset:
   - Provide a dataset root directory containing train, val, and test folders
   - Each folder should contain class folders with images

3. Flat dataset with Excel file:
   - Provide an image root directory containing all images (not in class folders)
   - Provide an Excel file path with columns: filename, label, and phase

Required arguments depend on the dataset type:
- output_dir: Directory where the output files will be saved (always required)
- One of the following combinations:
  a) image_root + splits_root
  b) dataset_root (containing train/val/test folders)
  c) image_root + excel_path

Optional arguments for training parameters:
- -e/--epochs: Number of training epochs (default: 100)
- -i/--imgsz: Image size for training (default: 640)
- -b/--batch: Batch size (default: 16)
- -p/--patience: Training patience (default: 10)

Split files should contain just the image names or filenames without path:
    image.jpg       
    image2 
    etc.

Excel file should have the following columns:
- filename: Name of the image file
- label: Class label (will be used as folder name)
- phase: One of 'train', 'val', or 'test'

See yolo_complete_cls_pipeline.py for more details on the main function.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO classification pipeline')
    
    # Configuration file (optional - overrides other arguments)
    parser.add_argument('--config', type=str,
                        help='Path to YAML configuration file (optional)')
    
    # Output directory (can be in config or command line)
    parser.add_argument('output_dir', type=str, nargs='?',
                        help='Directory for output files (required unless using --config)')
    
    # Create a mutually exclusive group for the dataset input methods
    input_group = parser.add_mutually_exclusive_group(required=False)
    
    # Method 1: Structured dataset with split files
    input_group.add_argument('--splits-input', nargs=2, metavar=('IMAGE_ROOT', 'SPLITS_ROOT'),
                       help='Provide IMAGE_ROOT containing class folders and SPLITS_ROOT containing split text files')
    
    # Method 2: Pre-split dataset
    input_group.add_argument('--presplit-input', type=str, metavar='DATASET_ROOT',
                       help='Provide DATASET_ROOT containing train, val, and test folders')
    
    # Method 3: Flat dataset with Excel file
    input_group.add_argument('--excel-input', nargs=2, metavar=('IMAGE_ROOT', 'EXCEL_PATH'),
                       help='Provide IMAGE_ROOT containing all images and EXCEL_PATH with columns: filename, label, phase')
    
    # Basic training parameters
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('-i', '--imgsz', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('-p', '--patience', type=int, default=10,
                        help='Training patience (default: 10)')
    parser.add_argument('--model', type=str, default='yolov8n-cls.pt',
                        help='YOLO model to use (default: yolov8n-cls.pt)')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to use (default: 0 for GPU:0, -1 for CPU)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--project', type=str, default=None,
                        help='Project name for saving outputs (optional)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name within project (optional)')
    parser.add_argument('--sample-size', type=int, default=30,
                        help='Number of images for EigenCAM visualization (default: 30)')
    
    # Augmentation parameters - these are optional and will use defaults from main file if not provided
    aug_group = parser.add_argument_group('Augmentation Parameters (optional)')
    aug_group.add_argument('--hsv_h', type=float, help='HSV-Hue augmentation fraction')
    aug_group.add_argument('--hsv_s', type=float, help='HSV-Saturation augmentation fraction')
    aug_group.add_argument('--hsv_v', type=float, help='HSV-Value augmentation fraction')
    aug_group.add_argument('--degrees', type=float, help='Image rotation degrees')
    aug_group.add_argument('--translate', type=float, help='Image translation fraction')
    aug_group.add_argument('--scale', type=float, help='Image scale fraction')
    aug_group.add_argument('--shear', type=float, help='Image shear degrees')
    aug_group.add_argument('--perspective', type=float, help='Image perspective fraction')
    aug_group.add_argument('--flipud', type=float, help='Vertical flip probability')
    aug_group.add_argument('--fliplr', type=float, help='Horizontal flip probability')
    aug_group.add_argument('--mosaic', type=float, help='Mosaic augmentation probability')
    aug_group.add_argument('--mixup', type=float, help='Mixup augmentation probability')
    aug_group.add_argument('--copy_paste', type=float, help='Copy-paste augmentation probability')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check if using config file
    if args.config:
        from yolo_complete_cls_pipeline import load_config
        config = load_config(args.config)
        
        # Extract configuration
        dataset_config = config.get('dataset', {})
        output_config = config.get('output', {})
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        augmentations = config.get('augmentations', None)
        postprocessing_config = config.get('postprocessing', {})
        
        # Determine dataset input method from config
        if 'presplit_root' in dataset_config:
            image_root = dataset_config['presplit_root']
            splits = None
            excel_path = None
        elif 'excel_path' in dataset_config:
            image_root = dataset_config['image_root']
            splits = None
            excel_path = dataset_config['excel_path']
        elif 'splits' in dataset_config:
            image_root = dataset_config['image_root']
            splits = dataset_config['splits']
            excel_path = None
        else:
            raise ValueError("Config file must specify dataset input method")
        
        # Setup parameters from config
        output_dir = output_config.get('output_dir')
        train_name = output_config.get('train_name')
        test_name = output_config.get('test_name')
        model_name = model_config.get('name', 'yolov8n-cls.pt')
        sample_size = postprocessing_config.get('sample_size', 30)
        
        training_parameter = training_config
        if 'device' in model_config:
            training_parameter['device'] = model_config['device']
        
        # Run main with config parameters
        main(image_root, output_dir, splits, training_parameter, excel_path, 
             model_name, augmentations, train_name, test_name, sample_size)
    
    else:
        # Original command-line argument handling
        if not args.output_dir:
            raise ValueError("output_dir is required when not using --config")
        
        # Create training parameters dictionary
        training_parameter = {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "device": args.device,
            "dropout": args.dropout,
            "project": args.project,
            "name": args.name
        }
        
        # Create augmentations dictionary - only include provided parameters
        augmentations = {}
        aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
                      'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste']
        
        for param in aug_params:
            if getattr(args, param) is not None:
                augmentations[param] = getattr(args, param)
        
        # If no augmentations provided, use None to trigger defaults in main()
        if not augmentations:
            augmentations = None
        
        # Determine which input method was chosen and set up variables accordingly
        if args.splits_input:
            image_root, splits_root = args.splits_input
            splits_root = Path(splits_root)
            splits = {
                "train": str(splits_root / "train.txt"),
                "val": str(splits_root / "val.txt"),
                "test": str(splits_root / "test.txt")
            }
            main(image_root, args.output_dir, splits, training_parameter, model_name=args.model, 
                 augmentations=augmentations, train_name=args.name, sample_size=args.sample_size)
        elif args.presplit_input:
            dataset_root = args.presplit_input
            main(dataset_root, args.output_dir, training_parameter=training_parameter, model_name=args.model, 
                 augmentations=augmentations, train_name=args.name, sample_size=args.sample_size)
        elif args.excel_input:
            image_root, excel_path = args.excel_input
            main(image_root, args.output_dir, training_parameter=training_parameter, excel_path=excel_path, 
                 model_name=args.model, augmentations=augmentations, train_name=args.name, sample_size=args.sample_size)
