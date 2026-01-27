"""
Oversampling Script for Imbalanced Dataset Balancing.

This script balances class distribution by oversampling minority classes
through random image duplication to match the majority class count.

Author: User
Date: 2026-01-26
"""

import os
import shutil
import random
import logging
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your ORIGINAL dataset (root folder containing train/val/test)
ORIGINAL_DATASET_DIR = "E:\\Datasets\\Phorids\\orientation\\Final_orientation\\cropov\\split" 

# Path where the NEW balanced dataset will be created
OUTPUT_DATASET_DIR = "E:\\Datasets\\Phorids\\orientation\\Final_orientation\\cropov\\split_balanced"
    
# Which split to balance? (ONLY 'train' usually)
TARGET_SPLIT = "train"

# Random seed for reproducibility
RANDOM_SEED = 42

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp')
# ---------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_original_images(cls_dir: Path) -> list:
    """
    Get list of original images (excluding previously augmented copies).
    
    Args:
        cls_dir: Path to class directory
        
    Returns:
        List of original image filenames
    """
    images = []
    for f in os.listdir(cls_dir):
        # Skip previously augmented copies to avoid copying copies
        if '_aug_copy_' in f:
            continue
        if f.lower().endswith(IMAGE_EXTENSIONS):
            images.append(f)
    return images


def balance_dataset():
    """
    Balance dataset by oversampling minority classes to match majority class.
    
    This function:
    1. Copies the entire dataset to a new location
    2. Analyzes class distribution in the training split
    3. Oversamples minority classes by duplicating random images
    """
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    logger.info(f"Random seed set to: {RANDOM_SEED}")
    
    input_path = Path(ORIGINAL_DATASET_DIR)
    output_path = Path(OUTPUT_DATASET_DIR)

    # Validate input directory exists
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return False
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        return False

    # 1. Copy the entire dataset structure first (Val/Test stay identical)
    if output_path.exists():
        logger.error(f"Output directory already exists: {output_path}")
        logger.error("Please delete it or change the output path.")
        return False
    
    logger.info(f"Copying '{input_path}' to '{output_path}'...")
    try:
        shutil.copytree(input_path, output_path)
    except Exception as e:
        logger.error(f"Failed to copy dataset: {e}")
        return False
    logger.info("Base copy complete. Now balancing the training set...")

    # 2. Analyze the Training Split
    train_dir = output_path / TARGET_SPLIT
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        return False
    
    classes = [d for d in os.listdir(train_dir) if (train_dir / d).is_dir()]
    
    if not classes:
        logger.error(f"No class directories found in: {train_dir}")
        return False
    
    # Count images per class (only original images)
    class_counts = {}
    class_images = {}  # Store image lists for efficiency
    
    for cls in classes:
        cls_dir = train_dir / cls
        images = get_original_images(cls_dir)
        class_counts[cls] = len(images)
        class_images[cls] = images
    
    # Find majority count (Target)
    max_count = max(class_counts.values())
    
    logger.info("Class distribution before balancing:")
    for cls, count in sorted(class_counts.items()):
        deficit = max_count - count
        logger.info(f"  - {cls}: {count} images (need +{deficit})")
    logger.info(f"Target count per class: {max_count}")

    # 3. Oversample Minority Classes
    total_added = 0
    for cls in classes:
        current_count = class_counts[cls]
        
        if current_count < max_count:
            needed = max_count - current_count
            logger.info(f"Balancing '{cls}': Adding {needed} copies...")
            
            cls_dir = train_dir / cls
            images = class_images[cls]
            
            if not images:
                logger.warning(f"No images found in class '{cls}', skipping...")
                continue
            
            # Randomly sample images to duplicate
            for i in tqdm(range(needed), desc=f"Oversampling {cls}"):
                img_to_copy = random.choice(images)
                src_file = cls_dir / img_to_copy
                
                # Create a unique name for the copy
                name, ext = os.path.splitext(img_to_copy)
                dst_file = cls_dir / f"{name}_aug_copy_{i}{ext}"
                
                try:
                    shutil.copy(src_file, dst_file)
                    total_added += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {src_file}: {e}")

    # Verify final counts and create comparison
    final_counts = {}
    for cls in classes:
        cls_dir = train_dir / cls
        final_count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])
        final_counts[cls] = final_count
    
    # Print comparison table
    logger.info("=" * 70)
    logger.info("BALANCING COMPLETE!")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'CLASS':<40} {'BEFORE':>10} {'AFTER':>10} {'ADDED':>10}")
    logger.info("-" * 70)
    
    for cls in sorted(classes):
        before = class_counts[cls]
        after = final_counts[cls]
        added = after - before
        logger.info(f"{cls:<40} {before:>10} {after:>10} {added:>10}")
    
    logger.info("-" * 70)
    total_before = sum(class_counts.values())
    total_after = sum(final_counts.values())
    logger.info(f"{'TOTAL':<40} {total_before:>10} {total_after:>10} {total_added:>10}")
    logger.info("=" * 70)
    logger.info(f"New balanced dataset saved to: {output_path}")
    logger.info("=" * 70)
    
    return True

if __name__ == "__main__":
    balance_dataset()