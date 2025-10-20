import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display requirements in server/automated environments
import matplotlib.pyplot as plt

import os
from pathlib import Path
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch
import cv2
import shutil
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from typing import Union, List, Tuple, Optional, Dict, Any
import argparse
import logging
import gc
import yaml
from datetime import datetime

# Configure logging - only to console by default
# Individual run logs will be created in the output directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global constants
IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', 
    '.webp', '.dng', '.pfm', '.mpo',
    '.JPG', '.JPEG', '.PNG', '.TIFF', '.TIF', '.BMP', '.WEBP'
]

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def validate_inputs(
    image_root: Optional[Union[str, Path]] = None,
    splits: Optional[Dict[str, str]] = None,
    excel_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Validate input paths and files before processing.
    
    Args:
        image_root: Root directory containing images
        splits: Dictionary of split file paths
        excel_path: Path to Excel metadata file
        model_name: Model name or path
        
    Raises:
        FileNotFoundError: If required files or directories don't exist
        ValueError: If input parameters are invalid
    """
    logger.info("Validating inputs...")
    
    # Validate image root if provided
    if image_root is not None:
        image_root = Path(image_root)
        if not image_root.exists():
            raise FileNotFoundError(f"Image root directory not found: {image_root}")
        if not image_root.is_dir():
            raise ValueError(f"Image root path is not a directory: {image_root}")
        logger.debug(f"Image root validated: {image_root}")
    
    # Validate split files if provided
    if splits is not None:
        for split_name, split_path in splits.items():
            split_path = Path(split_path)
            if not split_path.exists():
                raise FileNotFoundError(f"{split_name} split file not found: {split_path}")
            if not split_path.is_file():
                raise ValueError(f"{split_name} split path is not a file: {split_path}")
            logger.debug(f"Split file '{split_name}' validated: {split_path}")
    
    # Validate Excel file if provided
    if excel_path is not None:
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        if not excel_path.is_file():
            raise ValueError(f"Excel path is not a file: {excel_path}")
        if not excel_path.suffix.lower() in ['.xlsx', '.xls']:
            raise ValueError(f"Invalid Excel file format: {excel_path}")
        logger.debug(f"Excel file validated: {excel_path}")
    
    # Validate model name if it's a local path
    if model_name and os.path.exists(model_name):
        model_path = Path(model_name)
        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")
        logger.debug(f"Model file validated: {model_path}")
    
    logger.info("Input validation completed successfully")


def pathify(input: Union[str, Tuple[str], List[str]]) -> Union[Path, Tuple[Path], List[Path]]:
    """ Convert a path string or tuple or list thereof to a pathlib.Path object or tuple or list thereof.

    Args:
        input (Union[str, Tuple[str], List[str]]): Pathstring or tuple or list of pathsrings.

    Returns:
        Union[Path, Tuple[Path], List[Path]]: Path or tuple or list of Paths
    """

    # check if input is a string
    if isinstance(input, str):
        return Path(input)
    elif isinstance(input, tuple):
        return tuple([Path(item) for item in input])
    # check if input is a list
    elif isinstance(input, list):
        return [Path(item) for item in input]
    else:
        logger.error("Input must be a string, tuple, or list of strings")
        return None

def scan_folders(root_folder: str, 
                 search_extensions: List[str], 
                 output_type=str, abs_paths=False) -> list:
    """
    Walk through all folders and subfolders starting from root_folder,
    and return a list of relative file paths for files matching the extensions in search_extensions.
    
    Args:
        root_folder (str): The root directory to start the search from.
        search_extensions (list): A list of file extensions to search for (e.g., ['.txt', '.jpg']).
        output_type (type): The output type for the relative file paths. Can be set to Path (default is str)
        abs_paths (bool): Return absolute file paths instead of relative paths (default is False).

    Returns:
        list: A list of relative file paths for files that have the desired extensions.
    """
    matching_files = []
    
    # Walk through all the folders and files
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Get the file extension and check if it matches any in search_extensions
            if any(filename.lower().endswith(ext.lower()) for ext in search_extensions):
                # Create relative file path and append to the result list
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, root_folder)
                
                if abs_paths:
                    matching_files.append(output_type(full_path))
                else:
                    matching_files.append(output_type(relative_path))
    
    logger.info(f"Found {len(matching_files)} files with extensions {search_extensions}")
    logger.debug(f"Root folder: {root_folder}")
    return matching_files

def get_folder_paths(root_folder):
    folder_paths = []
    for root, dirs, files in os.walk(root_folder):
        for name in dirs:
            folder_paths.append(os.path.join(root, name))
    return folder_paths

def create_content_dataframe(files: list, file_name_column: str="filename") -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of file paths.
    Dataframes contains the columns 'file', 'filename', 'filepath', 'extension'.

    file: The full file name.
    filename: The file name without extension.
    filepath: The full filepaths (as provided).
    extension: The file extension.

    Args:
        files (list): A list of file paths.
        file_name_column (str): The column name for the filename column (default is 'Filename').
    
    Returns:
        pd.DataFrame: A DataFrame with the columns 'file', 'filename', 'filepath', 'extension'.
    """

    
    # Create a DataFrame from the list of files
    df = pd.DataFrame(files, columns=["filepath"])    
    # Extract the filename and extension
    df["file"] = df["filepath"].apply(lambda x: os.path.basename(x))
    df["extension"] = df["file"].apply(lambda x: os.path.splitext(x)[1])
    df[file_name_column] = df["file"].apply(lambda x: os.path.splitext(x)[0])
    
    return df

def copy_temp_dataset(image_root: Union[str, Path], dest_root: Union[str, Path], split_files: Dict[str, str]) -> None:
    """
    Copy the dataset to a temporary location for training
    
    Args:
        image_root: Root directory containing images organized by class
        dest_root: Destination directory for temporary dataset
        split_files: Dictionary mapping split names to file paths
    """
    image_root = Path(image_root)
    dest_root = Path(dest_root)
    
    logger.info(f"Copying dataset from {image_root} to {dest_root}")
    
    # create the destination folder if it doesn't exist
    dest_root.mkdir(parents=True, exist_ok=True)

    # load image paths from image source
    image_paths = create_content_dataframe(scan_folders(str(image_root), IMAGE_EXTENSIONS, abs_paths=True))
    
    for k in ["train", "val", "test"]:
        if k in split_files.keys():
            # create a new folder for the split
            split_folder = Path(dest_root) / k
            split_folder.mkdir(parents=True, exist_ok=True)

            # read text file 
            with open(split_files[k], 'r') as f:
                lines = f.readlines()

            # adding progress bar to evaluate lenghy processes
            with Progress(
                    TextColumn("[cyan]{task.description}"),  # Task description
                    BarColumn(),  # Progress bar
                    TaskProgressColumn(),  # Percentage and progress
                    TimeRemainingColumn(),  # Estimated time remaining
                    TextColumn("[green]Processed: {task.completed}/{task.total}  {task.fields[image_name]}"),  # Custom text showing processed/total
                ) as progress:
                    task = progress.add_task(f"Copying {k} images ", total=len(lines), image_name="")

                    

                    # for each line in the text file find the image path in image_paths (image_paths['filepath']) and copy the image to the split folder
                    for line in lines:
                        line = line.strip()

                        # remove file extension from line
                        line = os.path.splitext(line)[0]

                        # find the image path in image_paths (image_paths is a dataframe)
                        image_path = image_paths[image_paths['filename'] == line]['filepath'].values[0]
                        image_path = Path(image_path)

                        # copy the image to the split with the correct folder structure
                        # find class folder of current image
                        current_class_folder = image_path.parent.name
                        # create class folder in split folder
                        dest_class_folder = split_folder / current_class_folder
                        dest_class_folder.mkdir(parents=True, exist_ok=True)

                        # copy the image to the class folder
                        dest_path = dest_class_folder / image_path.name
                        
                        shutil.copy(image_path, dest_path)
                
                        # Update progress bar
                        progress.update(task, advance=1 ,image_name=image_path.name)
            
        else:
            logger.warning(f"Split '{k}' not found in split_files dictionary. Available splits: {list(split_files.keys())}")
            continue  # Skip this split if not provided
                
def validate_presplit_dataset(split_root: Union[str, Path]) -> bool:
    """
    Validate that a pre-split dataset has the required structure.
    
    Args:
        split_root: Root directory containing train, val, test folders
        
    Returns:
        bool: True if valid, False otherwise
    """
    split_root = Path(split_root)
    expected_folders = ["train", "val", "test"]
    
    logger.info(f"Validating pre-split dataset at {split_root}")
    
    all_valid = True
    for folder in expected_folders:
        folder_path = split_root / folder
        if not folder_path.exists():
            logger.error(f"Required folder '{folder}' not found in {split_root}")
            all_valid = False
        else:
            # Check if folder contains class subdirectories
            class_folders = [f for f in folder_path.iterdir() if f.is_dir()]
            if not class_folders:
                logger.warning(f"Folder '{folder}' exists but contains no class subdirectories")
            else:
                logger.info(f"Found {len(class_folders)} classes in {folder} split")
    
    return all_valid

def create_dataset_from_excel(image_root: str | Path, excel_path: str | Path, dest_root: str | Path):
    """
    Create a dataset from a single folder of images and an Excel file with filename, label, and phase columns.
    
    Args:
        image_root (str | Path): Root directory containing all images
        excel_path (str | Path): Path to Excel file with filename, label, and phase columns
        dest_root (str | Path): Destination directory for the temporary dataset
    """
    image_root = Path(image_root)
    excel_path = Path(excel_path)
    dest_root = Path(dest_root)
    
    # Create the destination folder if it doesn't exist
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # Read the Excel file
    try:
        print(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        print(f"Excel file columns: {df.columns.tolist()}")
        
        # Check if there are required columns or if we need to normalize column names
        required_columns = ["filename", "label", "phase"]
        column_mapping = {}
        
        # Try to find case-insensitive matches for required columns
        for req_col in required_columns:
            for col in df.columns:
                if col.lower() == req_col.lower():
                    column_mapping[col] = req_col
        
        # If mappings were found, rename columns
        if column_mapping and len(column_mapping) == len(required_columns):
            print(f"Renaming columns: {column_mapping}")
            df = df.rename(columns=column_mapping)
        
        # Final check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns not found in Excel file: {missing_columns}")
            
        print(f"Found {len(df)} rows in Excel file")
        
        # Print class distribution by phase
        print("\n=== EXCEL DATA DISTRIBUTION ===")
        for phase in ['train', 'val', 'test']:
            phase_df = df[df['phase'] == phase]
            print(f"\nPhase: {phase} - {len(phase_df)} images")
            class_counts = phase_df['label'].value_counts()
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count} images")
        print("=============================\n")
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise
    
    # Check that all phases are valid
    valid_phases = ["train", "val", "test"]
    invalid_phases = df[~df["phase"].isin(valid_phases)]["phase"].unique()
    if len(invalid_phases) > 0:
        print(f"Warning: Invalid phases found in Excel file: {invalid_phases}")
        print(f"Valid phases are: {valid_phases}")
        # Filter out invalid phases
        df = df[df["phase"].isin(valid_phases)]
    
    # Create destination folders for each phase
    for phase in valid_phases:
        phase_folder = dest_root / phase
        phase_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created {phase} folder at {phase_folder}")
    
    # Get list of image files in the source directory
    logger.info(f"Scanning for images in {image_root}")
    
    # Try both methods to find image files
    image_files = []
    # Method 1: Use glob
    for ext in IMAGE_EXTENSIONS:
        found_files = list(image_root.glob(f"*{ext}"))
        image_files.extend(found_files)
    
    # Method 2: Use scan_folders if glob didn't find anything
    if not image_files:
        logger.info("No images found with glob, using recursive scan...")
        image_files_str = scan_folders(str(image_root), IMAGE_EXTENSIONS, str, abs_paths=True)
        image_files = [Path(f) for f in image_files_str]
    
    logger.info(f"Found {len(image_files)} image files in {image_root}")
    
    if len(image_files) == 0:
        error_msg = f"No image files found in {image_root} with supported extensions: {IMAGE_EXTENSIONS}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create a lookup dictionary of image files by name (without extension)
    image_lookup = {}
    for img_path in image_files:
        image_lookup[img_path.stem.lower()] = img_path  # Convert to lowercase for case-insensitive matching
    
    print(f"Created lookup dictionary with {len(image_lookup)} images")
    
    # Count of processed and not found images
    processed_count = 0
    not_found_count = 0
    
    # Track processed files by phase and class
    phase_class_counts = {phase: {} for phase in valid_phases}
    
    # Copy images to destination folders based on Excel data
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]Processed: {task.completed}/{task.total}  {task.fields[image_name]}"),
    ) as progress:
        task = progress.add_task("Copying images", total=len(df), image_name="")
        
        # Process each row in the Excel file
        for _, row in df.iterrows():
            filename = row["filename"]
            label = row["label"]
            phase = row["phase"]
            
            # Strip file extension if present
            filename_stem = Path(filename).stem.lower()  # Convert to lowercase for case-insensitive matching
            
            # Find the image file
            if filename_stem in image_lookup:
                source_path = image_lookup[filename_stem]
                
                # Create label folder if it doesn't exist
                label_folder = dest_root / phase / label
                label_folder.mkdir(parents=True, exist_ok=True)
                
                # Copy the image
                dest_path = label_folder / source_path.name
                shutil.copy(source_path, dest_path)
                
                # Update counter
                processed_count += 1
                
                # Update phase-class tracking
                if label not in phase_class_counts[phase]:
                    phase_class_counts[phase][label] = 0
                phase_class_counts[phase][label] += 1
                
                # Update progress bar
                progress.update(task, advance=1, image_name=source_path.name)
            else:
                print(f"Warning: Image file for '{filename_stem}' not found in {image_root}")
                not_found_count += 1
                progress.update(task, advance=1, image_name=f"{filename_stem} (not found)")
    
    print(f"Excel dataset creation complete: {processed_count} images processed, {not_found_count} images not found")
    
    # Print summary of created dataset
    print("\n=== CREATED DATASET SUMMARY ===")
    for phase in valid_phases:
        phase_folder = dest_root / phase
        if phase_folder.exists():
            print(f"\nPhase: {phase}")
            class_folders = [f for f in phase_folder.iterdir() if f.is_dir()]
            for class_folder in class_folders:
                class_images = list(class_folder.glob('*.*'))
                print(f"  - {class_folder.name}: {len(class_images)} images")
        else:
            print(f"Phase folder {phase} does not exist!")
    print("=============================\n")
    
    # Verify actual counts match expected counts from Excel
    print("\n=== VALIDATION ===")
    all_valid = True
    for phase in valid_phases:
        print(f"Phase: {phase}")
        for label, expected_count in phase_class_counts[phase].items():
            actual_folder = dest_root / phase / label
            if actual_folder.exists():
                actual_count = len(list(actual_folder.glob('*.*')))
                match = actual_count == expected_count
                if not match:
                    all_valid = False
                print(f"  - {label}: Expected {expected_count}, Actual {actual_count}, Match: {match}")
            else:
                all_valid = False
                print(f"  - {label}: Folder does not exist!")
    print("================\n")
    
    # Verify that we created the required folders
    for phase in valid_phases:
        phase_folder = dest_root / phase
        if not phase_folder.exists() or not any(phase_folder.iterdir()):
            print(f"WARNING: {phase} folder is empty or not created!")
    
    # Return if we have all the required folders and validation passed
    if all((dest_root / phase).exists() for phase in valid_phases):
        if not all_valid:
            print("WARNING: Created dataset doesn't match expected counts!")
        return True
    else:
        missing_folders = [phase for phase in valid_phases if not (dest_root / phase).exists()]
        raise RuntimeError(f"Failed to create all required folders. Missing: {missing_folders}")

def detect_dataset_type(dataset_path, splits=None, excel_path=None):
    """
    Detect the type of dataset input provided.
    
    Args:
        dataset_path (str | Path): Path to the dataset
        splits (dict, optional): Dictionary with split file paths
        excel_path (str | Path, optional): Path to Excel file
        
    Returns:
        str: Dataset type ('txt_splits', 'presplit', or 'excel')
    """
    dataset_path = Path(dataset_path)
    
    # Check if it's a pre-split dataset (has train, val, test folders)
    if all((dataset_path / folder).exists() for folder in ["train", "val", "test"]):
        return "presplit"
    
    # Check if Excel file is provided
    if excel_path is not None and Path(excel_path).exists():
        return "excel"
    
    # Check if splits dictionary is provided
    if splits is not None and all(k in splits for k in ["train", "val", "test"]):
        return "txt_splits"
    
    # Default case
    return "unknown"

def train_yolo_classification(dataset_path, project_dir, train_name, training_parameter, model_name='yolov8n-cls.pt', augmentations=None):
    """
    Train YOLO classification model
    
    Args:
        dataset_path (str): Path to dataset
        project_dir (str): Output directory
        train_name (str): Name for this training run
        training_parameter (dict): Training parameters
        model_name (str): Name or path of YOLO model (default: yolov8n-cls.pt)
        augmentations (dict): Augmentation parameters (optional)
    """
    # Load a model
    model = YOLO(model_name)  # load a pretrained model
    
    # Merge training parameters and augmentations
    train_args = dict(training_parameter)
    if augmentations:
        train_args.update(augmentations)
    
    # Remove project and name from train_args if they exist
    if 'project' in train_args:
        del train_args['project']
    if 'name' in train_args:
        del train_args['name']
    
    # Train the model
    results = model.train(
        data=dataset_path,
        project=project_dir,
        name=train_name,
        **train_args,
    )
    
    return model, results

def run_validation_evaluation(
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    output_root: Union[str, Path],
    val_name: str = "val_results"
) -> None:
    """
    Run validation evaluation using YOLO's built-in validation method.
    This is the proper way to evaluate on validation set for benchmarking.
    
    Args:
        model_path: Path to trained model (best.pt)
        dataset_path: Path to dataset root (contains train/val/test folders)
        output_root: Root directory for outputs
        val_name: Name for validation results folder
    """
    logger.info("="*60)
    logger.info("RUNNING VALIDATION EVALUATION")
    logger.info("="*60)
    
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Validation results will be saved to: {output_root / val_name}")
    
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Run validation - YOLO will automatically use the 'val' folder
    logger.info("Running validation on validation dataset using YOLO's .val() method...")
    val_results = model.val(
        data=str(dataset_path),
        split='val',  # Use validation split
        save_json=True,  # Save results in JSON format
        save_hybrid=True,  # Save hybrid results
        project=str(output_root),
        name=val_name,
        plots=True,  # Generate plots
    )
    
    # Log validation metrics
    logger.info("\n" + "="*50)
    logger.info("VALIDATION METRICS:")
    logger.info(f"Top-1 Accuracy: {val_results.top1:.4f}")
    logger.info(f"Top-5 Accuracy: {val_results.top5:.4f}")
    logger.info("="*50)
    
    # Save validation metrics to a text file
    val_results_dir = output_root / val_name
    metrics_file = val_results_dir / 'validation_metrics.txt'
    
    with open(metrics_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("VALIDATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Top-1 Accuracy: {val_results.top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {val_results.top5:.4f}\n")
        f.write("\n")
        f.write("Note: These metrics are calculated by YOLO on the validation set.\n")
        f.write("Use these results for model benchmarking and comparison.\n")
    
    logger.info(f"Validation metrics saved to: {metrics_file}")
    logger.info("Validation evaluation completed!")
    logger.info("="*60 + "\n")

def run_postprocessing(
    model_path: Union[str, Path], 
    test_data_folder: Union[str, Path], 
    output_root: Union[str, Path], 
    classes: List[str], 
    test_name: str = "test_results", 
    sample_size: int = 30, 
    excel_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Run postprocessing: inference, evaluation, and visualization.
    
    Args:
        model_path: Path to trained model
        test_data_folder: Path to test data folder
        output_root: Root directory for outputs
        classes: List of class names
        test_name: Name for test results folder
        sample_size: Number of images to generate EigenCAM visualizations for
        excel_df: Optional Excel dataframe for ground truth labels
    """
    
    ##########################
    # PREPARE POSTPROCESSING #
    ##########################
    logger.info("="*60)
    logger.info("STARTING POSTPROCESSING")
    logger.info("="*60)
    
    output_root = Path(output_root)
    model_path = Path(model_path)
    test_data_folder = Path(test_data_folder)
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test data folder: {test_data_folder}")
    logger.info(f"Output root: {output_root}")

    model = YOLO(str(model_path))
    
    # get groundtruth value of test dataset
    logger.info(f"Loading ground truth from: {test_data_folder}")
    test_images = scan_folders(str(test_data_folder), IMAGE_EXTENSIONS, abs_paths=True)
    test_data_df = create_content_dataframe(test_images)
    
    # create ground truth dataframe from folder structure or Excel file
    df_gt = pd.DataFrame(columns=['Filename', 'Label'])
    df_gt['Filename'] = test_data_df['filename'].to_list()
    
    # Get ground truth labels
    if excel_df is not None:
        # If we have Excel data, use it for ground truth
        logger.info("Using Excel file for ground truth labels")
        # Create a lookup dictionary from Excel data for test phase
        excel_test_data = excel_df[excel_df['phase'] == 'test']
        excel_labels = {}
        for _, row in excel_test_data.iterrows():
            filename = Path(row['filename']).stem.lower()
            excel_labels[filename] = row['label']
        
        # Map filenames to labels using the Excel data
        df_gt['Label'] = df_gt['Filename'].apply(
            lambda x: excel_labels.get(x.lower(), "Unknown")
        )
        
        # Check if any files are missing or have "Unknown" labels
        unknown_count = (df_gt['Label'] == "Unknown").sum()
        if unknown_count > 0:
            logger.warning(f"{unknown_count} test images not found in Excel file!")
    else:
        # Otherwise, use folder structure
        logger.info("Using folder structure for ground truth labels")
        # Extract the label from the filepath
        df_gt['Label'] = test_data_df['filepath'].apply(lambda x: Path(x).parent.name)

    # Print ground truth label distribution
    logger.info("\n" + "="*50)
    logger.info("Ground truth label distribution:")
    label_counts = df_gt['Label'].value_counts()
    for label, count in label_counts.items():
        logger.info(f"  - {label}: {count} images")
    logger.info("="*50)



    ########################
    # COMPOSE TEST DATASET #
    ########################
    # create test_compose folder in the test_data_folder
    # to apply yolo model, all images must be in one folder
    test_compose_folder = test_data_folder.parent / "test_compose"
    logger.info(f"Creating test compose folder: {test_compose_folder}")
    
    # delete if test_compose folder already exists and create a new one
    if test_compose_folder.exists():
        shutil.rmtree(test_compose_folder)
    test_compose_folder.mkdir(parents=True, exist_ok=True)    
    
    # Copy all files to test_compose folder
    test_images_list = [Path(img) for img in test_images] if isinstance(test_images, list) else [test_images]
    logger.info(f"Copying {len(test_images_list)} images to {test_compose_folder}")
    
    for image in test_images_list:
        image_path = Path(image)
        # Copy image (we copy instead of move to preserve original structure)
        dest_path = test_compose_folder / image_path.name
        shutil.copy(image_path, dest_path)



    ##################
    # TEST INFERENCE #
    ##################
    # Run inference on test dataset
    logger.info("Running inference on test dataset...")
    results = model.predict(str(test_compose_folder),
                            name=test_name,  # Run name for this training session
                            project=str(output_root),  # Save to project directory
                            save_txt=True)
    pred_folder = Path(results[0].save_dir)    
    logger.info(f"Predictions saved to: {pred_folder}")

    pred = pd.DataFrame({'Classes': classes, 'Score': np.zeros(len(classes), dtype=float)})

    pred_results = pd.DataFrame(columns=['Filename', 'Label'])
    
    # Count total number of images to process - fix glob pattern
    # Use set to avoid duplicates on case-insensitive filesystems (Windows)
    image_set = set()
    for ext in IMAGE_EXTENSIONS:
        for img in Path(test_compose_folder).glob(f'*{ext}'):
            image_set.add(img)
    image_list = list(image_set)
    
    logger.info(f"Found {len(image_list)} images in test_compose_folder for prediction")
    logger.info(f"Using results from {pred_folder}")

    # Track images that don't have prediction files
    missing_prediction_files = []
    
    for img_path in image_list:
        pred_ensemble = pred.copy()

        txt_path = pred_folder / 'labels' / img_path.with_suffix('.txt').name
        
        if not txt_path.exists():
            logger.warning(f"No prediction file found for {img_path.name}")
            missing_prediction_files.append(img_path.name)
            continue
            
        with txt_path.open('r+') as txt_file:
            txt_pred = {}
            for txt_file_line in txt_file:
                txt_file_line_split = txt_file_line.split(' ')
                cls = ' '.join(txt_file_line_split[1:]).strip()
                score = float(txt_file_line_split[0])
                txt_pred[cls] = score
            pred_ensemble['Score'] += pred_ensemble['Classes'].map(txt_pred,na_action='ignore').fillna(0.)
        pred_ensemble['Score'] /= 1
        ensemble_txt_result = [img_path.name] + [pred_ensemble.iloc[pred_ensemble['Score'].idxmax()]['Classes']]
        pred_results = pd.concat([pred_results.loc[:], pd.DataFrame([ensemble_txt_result], columns=['Filename', 'Label'])])

    num_unique_labels = pred_results['Label'].nunique()

    logger.info(f"Number of unique labels: {num_unique_labels}")
    logger.info(f"Unique predicted labels: {pred_results['Label'].unique()}")
    logger.info(f"Expected classes: {classes}")
    
    # Summary of prediction results
    logger.info("\n" + "="*50)
    logger.info("Prediction summary:")
    logger.info(f"Total images found in test folder: {len(test_images_list)}")
    logger.info(f"Total images processed for prediction: {len(image_list)}")
    logger.info(f"Total predictions generated: {len(pred_results)}")
    
    # Check for missing predictions and raise error if too many
    if missing_prediction_files:
        missing_pct = len(missing_prediction_files) / len(image_list) * 100 if len(image_list) > 0 else 0
        logger.warning(f"{len(missing_prediction_files)} images ({missing_pct:.1f}%) had no prediction files")
        
        if len(missing_prediction_files) <= 5:
            logger.warning(f"Missing prediction files: {missing_prediction_files}")
        else:
            logger.warning(f"First 5 missing prediction files: {missing_prediction_files[:5]}")
        
        # Raise error if more than 10% of predictions are missing
        if missing_pct > 10:
            error_msg = f"Too many missing predictions ({missing_pct:.1f}%)! Check model inference."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    # Distribution of predicted labels
    logger.info("\nPredicted label distribution:")
    pred_counts = pred_results['Label'].value_counts()
    for label, count in pred_counts.items():
        logger.info(f"  {label}: {count}")
    logger.info("="*50)




    ###################
    # PREPARE RESULTS #
    ###################
    # Log the size of dataframes before any filtering
    print(f"\nInitial ground truth dataframe size: {df_gt.shape}")
    print(f"Initial predictions dataframe size: {pred_results.shape}")
    
    # keep only rows of df_gt where 'Filename' is equal to 'file' in test_data_df
    df_gt = df_gt[df_gt['Filename'].isin(test_data_df['filename'].tolist())]
    print(f"Ground truth dataframe after filtering: {df_gt.shape}")
    
    # Strip the Filename from its extension in pred_results
    # Make a copy of the original filenames before modifying
    pred_results['Original_Filename'] = pred_results['Filename']
    pred_results['Filename'] = pred_results['Filename'].apply(lambda x: Path(x).stem)
    print(f"Prediction results after filename stripping: {pred_results.shape}")
    
    # Check for any discrepancies between ground truth and prediction filenames
    gt_filenames = set(df_gt['Filename'])
    pred_filenames = set(pred_results['Filename'])
    
    missing_in_preds = gt_filenames - pred_filenames
    missing_in_gt = pred_filenames - gt_filenames
    common_filenames = gt_filenames.intersection(pred_filenames)
    
    print(f"\nFilename matching: {len(common_filenames)} files in both ground truth and predictions")
    print(f"Files in ground truth but missing in predictions: {len(missing_in_preds)}")
    print(f"Files in predictions but missing in ground truth: {len(missing_in_gt)}")
    
    if len(missing_in_preds) > 0:
        print(f"Sample of files missing in predictions: {list(missing_in_preds)[:5]}")
    if len(missing_in_gt) > 0:
        print(f"Sample of files missing in ground truth: {list(missing_in_gt)[:5]}")
    
    ####################
    # CONFUSION MATRIX #
    ####################    
    # Merge the DataFrames on 'Filename' - use inner join to include only files with both predictions and ground truth
    merged_df = pd.merge(df_gt, pred_results, on='Filename', suffixes=('_gt', '_pred'))
    
    # Now calculate accuracy on the merged dataframe to ensure same number of rows
    if len(merged_df) > 0:
        correct_predictions = (merged_df['Label_gt'] == merged_df['Label_pred']).sum()
        total_predictions = len(merged_df)
        accuracy = correct_predictions / total_predictions * 100
        logger.info(f"Quick accuracy check: {correct_predictions}/{total_predictions} correct ({accuracy:.2f}%)")
    
    # OneHotEncoder (keep for backwards compatibility if needed)
    one_hot_encoding = lambda series: torch.tensor((pd.get_dummies(series) * range(num_unique_labels)).sum(axis=1).values, device='cpu')
    
    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print(f"This means {merged_df.shape[0]} files were successfully matched between ground truth and predictions")
    
    if merged_df.shape[0] < len(gt_filenames) or merged_df.shape[0] < len(pred_filenames):
        print(f"\nWARNING: The merged dataframe has fewer rows ({merged_df.shape[0]}) than the input dataframes.")
        print(f"This indicates some files could not be matched between ground truth and predictions.")
        
        # Get counts of remaining files after the merge
        remaining_gt = len(gt_filenames) - merged_df.shape[0]
        remaining_pred = len(pred_filenames) - merged_df.shape[0]
        
        if remaining_gt > 0:
            print(f"Files in ground truth but not matched in merge: {remaining_gt}")
        if remaining_pred > 0:
            print(f"Files in predictions but not matched in merge: {remaining_pred}")
    
    # Save predictions to Excel file in test_results folder
    predictions_df = merged_df[['Filename', 'Label_gt', 'Label_pred']].copy()
    predictions_df.columns = ['file_name', 'true_label', 'prediction']  # Rename columns to requested format
    predictions_excel_path = pred_folder / 'predictions.xlsx'
    predictions_df.to_excel(predictions_excel_path, index=False)
    logger.info(f"Saved predictions to: {predictions_excel_path}")
    
    print("\nGround truth labels:", merged_df['Label_gt'].unique())
    print("Predicted labels:", merged_df['Label_pred'].unique())
    
    print("\nMerged DataFrame shape:", merged_df.shape)
    print("Ground truth labels:", merged_df['Label_gt'].unique())
    print("Predicted labels:", merged_df['Label_pred'].unique())
    
    # Get the list of unique labels from ground truth and predictions
    gt_labels = sorted(merged_df['Label_gt'].unique())
    pred_labels = sorted(merged_df['Label_pred'].unique())
    
    # Combine all unique labels to ensure consistency
    all_labels = sorted(list(set(gt_labels) | set(pred_labels)))
    
    print("All unique labels for confusion matrix:", all_labels)
    
    if len(all_labels) == 0:
        print("ERROR: No labels found in the merged dataframe!")
        return
    
    # Create Confusion Matrix with all labels
    cm = confusion_matrix(merged_df['Label_gt'], merged_df['Label_pred'], labels=all_labels)
    
    # Convert to DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    
    # Generate Classification Report with all labels
    try:
        cr = classification_report(merged_df['Label_gt'], merged_df['Label_pred'], labels=all_labels, target_names=all_labels)
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        # Fallback to a simpler report without target_names
        cr = classification_report(merged_df['Label_gt'], merged_df['Label_pred'])
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(30, 28))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    # Save the plot in high quality in test_results folder
    matrix_save_file = pred_folder / 'confusion_matrix.png'
    plt.savefig(matrix_save_file, dpi=600, bbox_inches='tight')  # Adjusting the bounding box
    logger.info(f"Saved confusion matrix to: {matrix_save_file}")

    # Print and save classification report
    logger.info("\nClassification Report:\n" + str(cr))
    # Save classification report to file in test_results folder
    report_save_file = pred_folder / 'classification_report.txt'
    with open(report_save_file, 'w') as f:
        f.write(cr)
    logger.info(f"Saved classification report to: {report_save_file}")


    ############
    # EIGENCAM #
    ############
    logger.info("\n" + "="*50)
    logger.info("Generating EigenCAM visualizations...")
    
    # Ensure we load the best model for visualization
    logger.info(f"Loading best model from {model_path} for EigenCAM")
    vis_model = YOLO(str(model_path))
    
    all_test_image_filepaths = test_data_df['filepath'].tolist()
    if len(all_test_image_filepaths) < sample_size:
        sample_size = len(all_test_image_filepaths)
        logger.warning(f"Sample size ({sample_size}) is larger than available images. Using all {sample_size} images.")
    
    # Randomly select n images
    logger.info(f"Randomly selecting {sample_size} images for EigenCAM from {len(all_test_image_filepaths)} total images")
    selected_images = random.sample(all_test_image_filepaths, sample_size)

    # Create eigencam subdirectory inside test_results folder
    eigencam_dir = pred_folder / 'eigencam'
    eigencam_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"EigenCAM visualizations will be saved to: {eigencam_dir}")

    # Loop over the selected images
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[image_name]}"),
    ) as progress:
        task = progress.add_task("Generating EigenCAM visualizations", total=len(selected_images), image_name="")
        
        for image_filepath in selected_images:
            progress.update(task, advance=1, image_name=Path(image_filepath).name)
            apply_eigencam(vis_model, image_filepath, eigencam_dir, target=-2, task='cls', show=False)
    
    # Memory cleanup after EigenCAM generation
    logger.info("Cleaning up memory after EigenCAM generation...")
    del vis_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("="*50)
        
    # Clean up temporary test_compose folder
    if test_compose_folder.exists():
        logger.info(f"Cleaning up temporary test_compose folder: {test_compose_folder}")
        shutil.rmtree(test_compose_folder)
    
    logger.info("\nPostprocessing completed successfully!")
    logger.info("="*60)

def apply_eigencam(model, image_path, output_dir, target=-2, task='cls', show=False):
    
    # Convert paths to Path objects if they are strings
    image_path = Path(image_path) if isinstance(image_path, str) else image_path
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    
    # Load the YOLO model
    target_layers = [model.model.model[target]]

    # Load and preprocess the image
    img = cv2.imread(str(image_path))
    rgb_img = img.copy()
    img = np.float32(img) / 255

    # Perform EigenCAM and display the CAM image
    cam = EigenCAM(model, target_layers, task=task)
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    plt.title(f"EigenCAM for {image_path}")
    plt.axis('off')
    if show:
        plt.show()

    # Save the CAM image
    cam_image_save_path = output_dir / f"cam_{Path(image_path).name}"
    if not output_dir is None:
        os.makedirs(output_dir, exist_ok=True)
        plt.imsave(str(cam_image_save_path), cam_image)

def main(
    image_root: Union[str, Path], 
    output_dir: Union[str, Path], 
    splits: Optional[Dict[str, str]] = None, 
    training_parameter: Optional[Dict[str, Any]] = None, 
    excel_path: Optional[Union[str, Path]] = None, 
    model_name: str = 'yolov8n-cls.pt', 
    augmentations: Optional[Dict[str, Any]] = None, 
    train_name: Optional[str] = None, 
    test_name: Optional[str] = None,
    sample_size: int = 30
) -> None:
    """
    Main pipeline to train and evaluate YOLO classification models.
    
    The pipeline performs:
    1. Training on train set
    2. Validation evaluation (for benchmarking) - uses YOLO's .val() method
    3. Test evaluation (detailed analysis) - includes confusion matrix, classification report, EigenCAM

    Args:
        image_root: Root directory containing images or pre-split dataset
        output_dir: Directory to save outputs
        splits: Dictionary with paths to train/val/test split files (optional)
        training_parameter: Training parameters dictionary (optional)
        excel_path: Path to Excel file with image metadata (optional)
        model_name: Name or path of YOLO model (default: yolov8n-cls.pt)
        augmentations: Augmentation parameters dictionary (optional)
        train_name: Name for training results folder (default: train_results)
        test_name: Name for test results folder (default: test_results)
        sample_size: Number of images for EigenCAM visualization (default: 30)
        
    Outputs:
        - train_results/: Training metrics and checkpoints (best.pt, last.pt)
        - val_results/: Validation metrics (use these for benchmarking!)
        - test_results/: Detailed test analysis (confusion matrix, classification report, EigenCAM)
    """
    logger.info("="*80)
    logger.info("YOLO CLASSIFICATION PIPELINE STARTED")
    logger.info("="*80)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for this specific run
    log_file = output_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
    
    # Validate inputs
    try:
        validate_inputs(image_root, splits, excel_path, model_name)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Input validation failed: {e}")
        raise
    
    # Use default names if not provided
    if train_name is None:
        train_name = 'train_results'
    if test_name is None:
        test_name = 'test_results'
    
    logger.info(f"Train results will be saved to: {train_name}")
    logger.info(f"Test results will be saved to: {test_name}")
    logger.info(f"EigenCAM sample size: {sample_size}")
    
    # Apply default training parameters if not provided
    if training_parameter is None:
        training_parameter = {
            "epochs": 10,
            "imgsz": 640,
            "batch": 16,
            "patience": 10,
            "device": [-1, -1],  # auto-select  GPU or you can define number of gpus manually
            "dropout": 0.3
        }
    
    # Apply default augmentation parameters if not provided
    if augmentations is None:
        augmentations = {
            "hsv_h": 0.015,  # HSV-Hue augmentation fraction
            "hsv_s": 0.7,    # HSV-Saturation augmentation fraction
            "hsv_v": 0.4,    # HSV-Value augmentation fraction
            "degrees": 0.0,  # Image rotation degrees
            "translate": 0.1, # Image translation fraction
            "scale": 0.5,    # Image scale fraction
            "shear": 0.0,    # Image shear degrees
            "perspective": 0.0, # Image perspective fraction
            "flipud": 0.0,   # Vertical flip probability
            "fliplr": 0.5,   # Horizontal flip probability
            "mosaic": 1.0,   # Mosaic augmentation probability
            "mixup": 0.0,    # Mixup augmentation probability
            "copy_paste": 0.0 # Copy-paste augmentation probability
        }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect dataset type
    dataset_type = detect_dataset_type(image_root, splits, excel_path)
    print(f"Detected dataset type: {dataset_type}")
    
    # Create temp dataset based on dataset type
    temp_dataset_path = Path(output_dir) / "temp_dataset"
    
    # Variable to hold Excel data if we're using that method
    excel_df = None
    
    if dataset_type == "presplit":
        # For pre-split datasets, we can train directly on the original dataset
        presplit_dataset_path = Path(image_root)
        
        # Train YOLO model directly on the pre-split dataset
        # Use output_dir as project and train_name as the experiment name
        train_yolo_classification(presplit_dataset_path, output_dir, train_name, training_parameter, model_name=model_name, augmentations=augmentations)
        
        # Results will be saved to output_dir/train_name
        exp_dir = output_dir / train_name
        
        # Find best.pt files
        results_dir = exp_dir
        model_files = scan_folders(str(results_dir), ['best.pt'], Path, abs_paths=True)
        if len(model_files) > 0:
            model_file = model_files[0]
        else:
            print(f"\nSomething went wrong. No model checkpoint found in: {exp_dir}\n")
            return
        
        # For inference, we need the test data in a single folder
        # So we'll create a minimal temp dataset just for the test data
        test_data_folder = presplit_dataset_path / "test"
        
        # Get classes from train folder 
        train_data_folder = presplit_dataset_path / "train"
        folder_paths = get_folder_paths(train_data_folder)
        classes = [Path(folder_path).name for folder_path in folder_paths]
        logger.info(f"Classes: {classes}")
        
        # Run validation evaluation (for benchmarking)
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Evaluating on VALIDATION set (for benchmarking)")
        logger.info("="*80)
        run_validation_evaluation(str(model_file), str(presplit_dataset_path), str(exp_dir), val_name="val_results")
        
        # Run test postprocessing - save results in the experiment directory
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Evaluating on TEST set (detailed analysis)")
        logger.info("="*80)
        run_postprocessing(str(model_file), str(test_data_folder), str(exp_dir), classes, test_name, sample_size)
        
    else:
        # For other dataset types, use the regular workflow
        if dataset_type == "excel":
            print(f"Creating dataset from Excel file: {excel_path}")
            # First load the Excel file to keep it for ground truth later
            try:
                excel_df = pd.read_excel(excel_path)
                # Normalize column names if needed
                required_columns = ["filename", "label", "phase"]
                column_mapping = {}
                
                # Try to find case-insensitive matches for required columns
                for req_col in required_columns:
                    for col in excel_df.columns:
                        if col.lower() == req_col.lower():
                            column_mapping[col] = req_col
                
                # If mappings were found, rename columns
                if column_mapping and len(column_mapping) == len(required_columns):
                    excel_df = excel_df.rename(columns=column_mapping)
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                return
                
            # Use Excel file to split dataset
            try:
                create_dataset_from_excel(image_root, excel_path, temp_dataset_path)
            except Exception as e:
                print(f"Error creating dataset from Excel: {e}")
                return
            
            # Verify that temp_dataset_path/train exists
            train_folder = temp_dataset_path / "train"
            if not train_folder.exists():
                print(f"Error: Train folder not created at {train_folder}")
                return
                
        elif dataset_type == "txt_splits":
            # Use text files to split dataset
            copy_temp_dataset(image_root, temp_dataset_path, splits)
        else:
            print(f"Unknown dataset type. Please provide valid dataset input.")
            return
    
        # Train YOLO model - use output_dir as project and train_name as experiment name
        train_yolo_classification(temp_dataset_path, output_dir, train_name, training_parameter, model_name=model_name, augmentations=augmentations)

        # Results will be saved to output_dir/train_name
        exp_dir = output_dir / train_name

        # Find best.pt files
        results_dir = exp_dir
        model_files = scan_folders(str(results_dir), ['best.pt'], Path, abs_paths=True)
        if len(model_files) > 0:
            model_file = model_files[0]
        else:
            print(f"\nSomething went wrong. No model checkpoint found in: {exp_dir}\n")
            return
        
        # Get classes from train folder 
        test_data_folder = temp_dataset_path / "test"
        train_data_folder = temp_dataset_path / "train"
        folder_paths = get_folder_paths(train_data_folder)
        classes = [Path(folder_path).name for folder_path in folder_paths]
        logger.info(f"Classes: {classes}")    
        
        # Run validation evaluation (for benchmarking)
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Evaluating on VALIDATION set (for benchmarking)")
        logger.info("="*80)
        run_validation_evaluation(str(model_file), str(temp_dataset_path), str(exp_dir), val_name="val_results")
        
        # Run test postprocessing - detailed analysis with confusion matrix
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Evaluating on TEST set (detailed analysis)")
        logger.info("="*80)
        # Pass the Excel dataframe to run_postprocessing if using Excel method
        if dataset_type == "excel" and excel_df is not None:
            run_postprocessing(str(model_file), str(test_data_folder), str(exp_dir), classes, test_name, sample_size, excel_df=excel_df)
        else:
            run_postprocessing(str(model_file), str(test_data_folder), str(exp_dir), classes, test_name, sample_size)
    
    # Copy split files to output directory if they exist
    if dataset_type == "txt_splits" and splits is not None:
        print("Copying split files to output directory...")
        for split_file in splits.values():
            if Path(split_file).exists():
                shutil.copy(split_file, output_dir)
    elif dataset_type == "excel" and excel_path is not None:
        print("Copying Excel file to output directory...")
        shutil.copy(excel_path, output_dir)
    
    # Remove temp dataset if it exists
    if temp_dataset_path.exists() and dataset_type != "presplit":
        shutil.rmtree(temp_dataset_path)
    
    print("\nDone!\n")

if __name__ == "__main__":

    # Define common training parameters
    training_parameter = {
        "epochs": 2,        # Number of epochs
        "imgsz": 224,       # Image size
        "batch": 16,        # Batch size
        "patience": 10,     # Early stopping patience
        "device": [-1, -1], # auto-selection multiple idle GPUs
        "dropout": 0.3
    }
    
    # Define augmentation parameters separately
    augmentation_params = {
        "flipud": 0.5,      # Vertical flip probability
    }
    
    # Define model name
    model_name = 'yolov8l-cls.pt'
    
    # Example 1: Using text files for splits
    # ======================================
    # print("\nExample 1: Using text files for train/val/test splits")
    # output_dir = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/test_run")
    # image_root = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk")
    
    # splits = {
    #     "train": Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk/train.txt"),
    #     "val": Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk/val.txt"),
    #     "test": Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk/test.txt")
    # }
    
    # Uncomment to run this example
    # main(image_root, output_dir, splits, training_parameter, model_name=model_name, augmentations=augmentation_params)
    
    # Example 2: Using pre-split dataset
    # =================================
    # print("\nExample 2: Using pre-split dataset (train/val/test folders)")
    # output_dir = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/test_run_split")
    # presplit_dataset = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk_split")
    
    # # Uncomment to run this example
    # main(presplit_dataset, output_dir, training_parameter=training_parameter, model_name=model_name, augmentations=augmentation_params)
    
    # Example 3: Using Excel file for dataset splitting
    # ==============================================
    print("\nExample 3: Using Excel file for dataset splitting")
    output_dir = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/test_run_excel")
    flat_image_root = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk_excel/all")
    excel_path = Path(r"C:/Users/es4994/Desktop/Project/PhDCodes/yolo_cls/dataset/smnk_excel/smnk.xlsx")
    
    # Uncomment to run this example
    main(flat_image_root, output_dir, training_parameter=training_parameter, excel_path=excel_path, model_name=model_name, augmentations=augmentation_params)
