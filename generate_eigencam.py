"""
Standalone EigenCAM Generator for YOLO Classification Models
=============================================================

This script generates EigenCAM (Explainable AI) visualizations for YOLO
classification models without requiring the full training pipeline.

Features:
- Load any trained YOLO classification model
- Process single image or entire folder
- Multiple target layer options
- Batch processing with progress tracking
- Optional comparison mode (multiple layers side-by-side)

Usage:
------
# Single image
python generate_eigencam.py --model path/to/best.pt --image path/to/image.jpg --output output_cam

# Folder of images
python generate_eigencam.py --model path/to/best.pt --input path/to/images/ --output output_cam

# Compare different layers
python generate_eigencam.py --model path/to/best.pt --image path/to/image.jpg --output output_cam --compare

# Custom target layer
python generate_eigencam.py --model path/to/best.pt --image path/to/image.jpg --output output_cam --target -1

# Custom target layer for method comparison
python generate_eigencam.py --model path/to/best.pt --image path/to/image.jpg --output output_cam --compare-methods --method-target -3

# Custom layer list for layer comparison
python generate_eigencam.py --model path/to/best.pt --image path/to/image.jpg --output output_cam --compare --compare-layers -1 -4 -6

Author: Auto-generated
Date: October 22, 2025
"""

import argparse
import os
from pathlib import Path
import sys
import logging
from typing import Union, List, Optional, Callable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from PIL import Image
from ultralytics.data.augment import classify_transforms


# Import YOLO and CAM modules
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not found. Install with: pip install ultralytics")
    sys.exit(1)

try:
    from yolo_cam.eigen_cam import EigenCAM
    from yolo_cam.grad_cam import GradCAM
    from yolo_cam.grad_cam_plusplus import GradCAMPlusPlus
    from yolo_cam.utils.image import show_cam_on_image
except ImportError as e:
    print(f"Error: yolo_cam module not found. Make sure you're running from the project root.")
    print(f"Details: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', 
    '.webp', '.dng', '.JPG', '.JPEG', '.PNG', '.TIFF', '.TIF', '.BMP', '.WEBP'
]


class ClassificationCAMModel(torch.nn.Module):
    """Wrapper that injects preprocessing before forwarding through the classifier."""

    def __init__(self, model: torch.nn.Module, preprocess_fn: Callable[[np.ndarray], torch.Tensor]):
        super().__init__()
        self.model = model
        self.preprocess_fn = preprocess_fn

    def forward(self, input_data: Union[np.ndarray, torch.Tensor]):
        if isinstance(input_data, np.ndarray):
            tensor = self.preprocess_fn(input_data)
        elif isinstance(input_data, torch.Tensor):
            tensor = input_data
        else:
            raise TypeError(f"Unsupported input type for CAM preprocessing: {type(input_data)}")
        return self.model(tensor)


class EigenCAMGenerator:
    """
    Standalone generator for EigenCAM visualizations.
    
    Handles model loading, image processing, and visualization generation
    with proper error handling and progress tracking.
    """
    
    def __init__(self, model_path: Union[str, Path], device: str = 'auto'):
        """
        Initialize the EigenCAM generator.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Prepare core model components for CAM generation
        self.model_core = self.model.model  # ClassificationModel
        self.model_core.to(self.device)
        self.model_core.eval()
        self.model_layers = self.model_core.model  # Sequential layers

        # Build preprocessing pipeline for classification inputs
        try:
            if hasattr(self.model_core, 'transforms') and self.model_core.transforms is not None:
                self.preprocess_transforms = self.model_core.transforms
            else:
                imgsz = getattr(getattr(self.model_core, 'args', {}), 'get', lambda *_: 640)('imgsz')
                if isinstance(imgsz, (list, tuple)):
                    imgsz = max(imgsz)
                self.preprocess_transforms = classify_transforms(imgsz)
        except Exception as transform_err:
            logger.warning(f"Falling back to default classification transforms: {transform_err}")
            self.preprocess_transforms = classify_transforms(640)

        self.cam_model = ClassificationCAMModel(self.model_core, self._prepare_tensor_for_cam)
        self.cam_model.to(self.device)
        self.cam_model.eval()
        
        # Get class names if available
        try:
            self.class_names = self.model.names
            logger.info(f"Loaded {len(self.class_names)} classes")
        except:
            self.class_names = None
            logger.warning("Could not load class names from model")
    
    def _prepare_tensor_for_cam(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single RGB image into a batched tensor with gradients enabled."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array for CAM preprocessing, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        tensor = self.preprocess_transforms(pil_image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)
        tensor = tensor.float()
        tensor.requires_grad_(True)
        return tensor

    def get_image_files(self, input_path: Union[str, Path]) -> List[Path]:
        """
        Get list of image files from path (file or directory).
        
        Args:
            input_path: Path to image file or directory
            
        Returns:
            List of image file paths
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        if input_path.is_file():
            if input_path.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]:
                return [input_path]
            else:
                raise ValueError(f"File is not a supported image format: {input_path}")
        
        # Directory - find all images
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.glob(f"**/*{ext}"))
        
        if not image_files:
            raise ValueError(f"No image files found in: {input_path}")
        
        # Sort for consistent ordering
        image_files = sorted(image_files)
        logger.info(f"Found {len(image_files)} image(s)")
        
        return image_files
    
    def generate_cam(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_layer: int = -2,
        save_prefix: str = "eigencam",
        show_prediction: bool = True,
        method: str = 'eigencam'
    ) -> Optional[np.ndarray]:
        """
        Generate EigenCAM visualization for a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            target_layer: Target layer index (default: -2)
            save_prefix: Prefix for saved filename
            show_prediction: Whether to add prediction info to title
            method: CAM method to use ('eigencam', 'gradcam', 'gradcam++')
            
        Returns:
            CAM image as numpy array, or None if failed
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        
        try:
            # Get model prediction first (for title/info)
            prediction_info = ""
            if show_prediction:
                try:
                    results = self.model(img_rgb, verbose=False)
                    pred_class = results[0].probs.top1
                    pred_conf = results[0].probs.top1conf.item()
                    
                    if self.class_names and pred_class < len(self.class_names):
                        class_name = self.class_names[pred_class]
                    else:
                        class_name = f"Class {pred_class}"
                    
                    prediction_info = f"\nPrediction: {class_name} ({pred_conf:.2%})"
                except Exception as e:
                    logger.warning(f"Could not get prediction info: {e}")
            
            # Select target layer with improved error handling
            try:
                # Access the model layers
                model_layers = self.model_layers
                
                # Handle negative indices
                if target_layer < 0:
                    actual_index = len(model_layers) + target_layer
                else:
                    actual_index = target_layer
                
                # Validate index
                if actual_index < 0 or actual_index >= len(model_layers):
                    logger.error(f"Target layer {target_layer} (index {actual_index}) out of range")
                    logger.info(f"Valid range: 0 to {len(model_layers)-1} (or -{len(model_layers)} to -1)")
                    return None
                
                target_layer_obj = model_layers[actual_index]
                
                # Check if layer is a tuple (some layers return tuples)
                if isinstance(target_layer_obj, tuple):
                    logger.warning(f"Layer {target_layer} returns tuple, using first element")
                    target_layer_obj = target_layer_obj[0]
                
                target_layers = [target_layer_obj]
                logger.debug(f"Using layer {target_layer} (type: {type(target_layer_obj).__name__})")
                
            except (IndexError, AttributeError, TypeError) as e:
                logger.error(f"Failed to access target layer {target_layer}: {e}")
                logger.info("Available layers:")
                try:
                    for i, layer in enumerate(self.model_layers):
                        logger.info(f"  {i}: {type(layer).__name__}")
                except:
                    logger.error("Could not enumerate model layers")
                return None
            
            # Generate CAM using selected method
            if method.lower() == 'eigencam':
                cam = EigenCAM(self.cam_model, target_layers, task='cls')
                method_name = "EigenCAM"
            elif method.lower() == 'gradcam':
                cam = GradCAM(self.cam_model, target_layers, task='cls')
                method_name = "GradCAM"
            elif method.lower() in ['gradcam++', 'gradcampp', 'gradcam_plusplus']:
                cam = GradCAMPlusPlus(self.cam_model, target_layers, task='cls')
                method_name = "GradCAM++"
            else:
                logger.error(f"Unknown CAM method: {method}. Use 'eigencam', 'gradcam', or 'gradcam++'")
                return None
            
            grayscale_cam = cam(img_rgb)[0, :, :]
            
            # Ensure valid range
            grayscale_cam = np.clip(grayscale_cam, 0, 1)
            
            # Create overlay visualization
            cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
            
            # Create figure with visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"Original Image{prediction_info}", fontsize=12)
            axes[0].axis('off')
            
            # CAM overlay
            axes[1].imshow(cam_image)
            axes[1].set_title(f"{method_name} (Layer {target_layer})", fontsize=12)
            axes[1].axis('off')
            
            plt.suptitle(f"{image_path.name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            save_path = output_dir / f"{save_prefix}_{image_path.name}"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.debug(f"Saved: {save_path}")
            
            return cam_image
            
        except Exception as e:
            logger.error(f"Failed to generate CAM for {image_path.name}: {e}")
            return None
    
    def generate_comparison(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
    target_layers: Optional[List[int]] = None
    ) -> bool:
        """
        Generate comparison of EigenCAM across multiple layers.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            target_layers: List of layer indices to compare (default: [-1, -2, -3, -4])
            
        Returns:
            True if successful, False otherwise
        """
        if not target_layers:
            target_layers = [-1, -2, -3, -4]
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        
        # Get prediction
        try:
            results = self.model(img_rgb, verbose=False)
            pred_class = results[0].probs.top1
            pred_conf = results[0].probs.top1conf.item()
            
            if self.class_names and pred_class < len(self.class_names):
                class_name = self.class_names[pred_class]
            else:
                class_name = f"Class {pred_class}"
            
            prediction_info = f"Prediction: {class_name} ({pred_conf:.2%})"
        except:
            prediction_info = ""
        
        # Create comparison figure
        n_layers = len(target_layers)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, target_idx in enumerate(target_layers):
            try:
                # Access layer with improved error handling
                model_layers = self.model_layers
                if target_idx < 0:
                    actual_index = len(model_layers) + target_idx
                else:
                    actual_index = target_idx
                
                target_layer_obj = model_layers[actual_index]
                if isinstance(target_layer_obj, tuple):
                    target_layer_obj = target_layer_obj[0]
                
                target_layer_list = [target_layer_obj]
                cam = EigenCAM(self.cam_model, target_layer_list, task='cls')
                grayscale_cam = cam(img_rgb)[0, :, :]
                grayscale_cam = np.clip(grayscale_cam, 0, 1)
                
                cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
                
                axes[idx].imshow(cam_image)
                axes[idx].set_title(f"Layer {target_idx}", fontsize=12, fontweight='bold')
                axes[idx].axis('off')
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Failed\nLayer {target_idx}\n{str(e)[:30]}", 
                              ha='center', va='center', fontsize=10, color='red')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"{image_path.name}\n{prediction_info}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f"comparison_{image_path.name}"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved comparison: {save_path}")
        return True
    
    def generate_method_comparison(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_layer: int = -2,
    methods: Optional[List[str]] = None
    ) -> bool:
        """
        Generate comparison of different CAM methods (EigenCAM vs GradCAM++).
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            target_layer: Target layer index (default: -2)
            methods: List of methods to compare (default: ['eigencam', 'gradcam++'])
            
        Returns:
            True if successful, False otherwise
        """
        if methods is None:
            methods = ['eigencam', 'gradcam++']
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        
        # Get prediction
        try:
            results = self.model(img_rgb, verbose=False)
            pred_class = results[0].probs.top1
            pred_conf = results[0].probs.top1conf.item()
            
            if self.class_names and pred_class < len(self.class_names):
                class_name = self.class_names[pred_class]
            else:
                class_name = f"Class {pred_class}"
            
            prediction_info = f"Prediction: {class_name} ({pred_conf:.2%})"
        except:
            prediction_info = ""
        
        # Get target layer with error handling
        try:
            model_layers = self.model_layers
            if target_layer < 0:
                actual_index = len(model_layers) + target_layer
            else:
                actual_index = target_layer
            
            target_layer_obj = model_layers[actual_index]
            if isinstance(target_layer_obj, tuple):
                target_layer_obj = target_layer_obj[0]
            
            target_layers = [target_layer_obj]
        except Exception as e:
            logger.error(f"Failed to access target layer {target_layer}: {e}")
            return False
        
        # Create comparison figure
        n_methods = len(methods) + 1  # +1 for original image
        fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 8))
        if n_methods == 2:
            axes = [axes[0], axes[1]]
        
        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original\n{prediction_info}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Generate CAM for each method
        for idx, method in enumerate(methods, start=1):
            try:
                if method.lower() == 'eigencam':
                    cam = EigenCAM(self.cam_model, target_layers, task='cls')
                    method_name = "EigenCAM"
                elif method.lower() == 'gradcam':
                    cam = GradCAM(self.cam_model, target_layers, task='cls')
                    method_name = "GradCAM"
                elif method.lower() in ['gradcam++', 'gradcampp']:
                    cam = GradCAMPlusPlus(self.cam_model, target_layers, task='cls')
                    method_name = "GradCAM++"
                else:
                    logger.warning(f"Unknown method: {method}, skipping")
                    axes[idx].text(0.5, 0.5, f"Unknown method:\n{method}", 
                                  ha='center', va='center', fontsize=12, color='red')
                    axes[idx].axis('off')
                    continue
                
                grayscale_cam = cam(img_rgb)[0, :, :]
                grayscale_cam = np.clip(grayscale_cam, 0, 1)
                
                cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
                
                axes[idx].imshow(cam_image)
                axes[idx].set_title(f"{method_name}\n(Layer {target_layer})", 
                                   fontsize=12, fontweight='bold')
                axes[idx].axis('off')
                
            except Exception as e:
                import traceback
                logger.error(f"Failed to generate {method}: {e}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                axes[idx].text(0.5, 0.5, f"Failed\n{method}\n{str(e)[:30]}", 
                              ha='center', va='center', fontsize=10, color='red')
                axes[idx].axis('off')
        
        plt.suptitle(f"{image_path.name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f"method_comparison_{image_path.name}"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved method comparison: {save_path}")
        return True
    
    def process_batch(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_layer: int = -2,
        compare_mode: bool = False,
        compare_methods: bool = False,
        method: str = 'eigencam',
        max_images: Optional[int] = None,
        compare_layers_override: Optional[List[int]] = None,
        method_compare_layer: Optional[int] = None
    ) -> dict:
        """
        Process multiple images in batch.
        
        Args:
            input_path: Path to image file or directory
            output_dir: Directory to save outputs
            target_layer: Target layer index (ignored if compare_mode=True)
            compare_mode: If True, generate layer comparisons
            compare_methods: If True, compare EigenCAM vs GradCAM++
            method: CAM method to use ('eigencam', 'gradcam', 'gradcam++')
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            Dictionary with processing statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = self.get_image_files(input_path)
        
        if max_images:
            image_files = image_files[:max_images]
            logger.info(f"Limited to {max_images} images")
        
        # Process images
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        logger.info(f"Processing {stats['total']} image(s)...")
        
        for image_path in tqdm(image_files, desc="Generating CAMs", unit="img"):
            try:
                if compare_methods:
                    effective_method_layer = method_compare_layer if method_compare_layer is not None else target_layer
                    success = self.generate_method_comparison(
                        image_path,
                        output_dir,
                        target_layer=effective_method_layer
                    )
                elif compare_mode:
                    success = self.generate_comparison(
                        image_path,
                        output_dir,
                        target_layers=compare_layers_override
                    )
                else:
                    result = self.generate_cam(
                        image_path, 
                        output_dir, 
                        target_layer=target_layer,
                        method=method
                    )
                    success = result is not None
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_files'].append(str(image_path))
                    
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                stats['failed'] += 1
                stats['failed_files'].append(str(image_path))
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Processing Summary:")
        logger.info(f"  Total images: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        if stats['failed'] > 0:
            logger.info(f"  Failed files: {stats['failed_files']}")
        logger.info(f"  Output directory: {output_dir.absolute()}")
        logger.info("=" * 60)
        
        return stats


def main():
    """Main entry point for standalone EigenCAM generation."""
    
    parser = argparse.ArgumentParser(
        description="Generate EigenCAM visualizations for YOLO classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Single image with EigenCAM
python generate_eigencam.py --model weights/best.pt --image test.jpg --output cam_output

# Compare different layers
python generate_eigencam.py --model weights/best.pt --image test.jpg --output cam_output --compare

# Compare EigenCAM vs GradCAM++ (recommended!)
python generate_eigencam.py --model weights/best.pt --image test.jpg --output cam_output --compare-methods

# Use GradCAM++ instead of EigenCAM
python generate_eigencam.py --model weights/best.pt --image test.jpg --output cam_output --method gradcam++

# Process folder with GradCAM++
python generate_eigencam.py --model weights/best.pt --input test_images/ --output cam_output --method gradcam++

# Limit number of images
python generate_eigencam.py --model weights/best.pt --input test_images/ --output cam_output --max-images 50
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to single input image'
    )
    
    parser.add_argument(
        '--input', '-I',
        type=str,
        help='Path to input directory (alternative to --image)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='eigencam_output',
        help='Output directory for CAM visualizations (default: eigencam_output)'
    )
    
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=-2,
        help='Target layer index for CAM generation (default: -2, second-to-last layer)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Generate comparison across multiple layers (layers -1, -2, -3, -4)'
    )
    
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Compare EigenCAM vs GradCAM++ side-by-side'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='eigencam',
        choices=['eigencam', 'gradcam', 'gradcam++', 'gradcampp'],
        help='CAM method to use (default: eigencam). Options: eigencam (fast, no gradients), gradcam (gradient-based), gradcam++ (best accuracy)'
    )

    parser.add_argument(
        '--method-target',
        type=int,
        default=None,
        help='Target layer index to use when --compare-methods is enabled (default: uses --target value)'
    )

    parser.add_argument(
        '--compare-layers',
        type=int,
        nargs='+',
        default=None,
        help='Layer indices to visualise when --compare is enabled (default: -1 -2 -3 -4)'
    )
    
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input arguments
    if not args.image and not args.input:
        parser.error("Either --image or --input must be specified")
    
    if args.image and args.input:
        parser.error("Cannot specify both --image and --input")

    if args.compare_layers is not None and not args.compare:
        parser.error("--compare-layers can only be used together with --compare")

    if args.method_target is not None and not args.compare_methods:
        parser.error("--method-target can only be used when --compare-methods is enabled")
    
    input_path = args.image if args.image else args.input
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("EigenCAM Generator Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Target Layer: {args.target}")
    logger.info(f"  Compare Layers: {args.compare}")
    logger.info(f"  Compare Methods: {args.compare_methods}")
    if args.compare_methods and args.method_target is not None:
        logger.info(f"  Method Comparison Layer Override: {args.method_target}")
    logger.info(f"  CAM Method: {args.method}")
    if args.compare and args.compare_layers is not None:
        logger.info(f"  Compare Layers Override: {args.compare_layers}")
    logger.info(f"  Device: {args.device}")
    if args.max_images:
        logger.info(f"  Max Images: {args.max_images}")
    logger.info("=" * 60)
    
    try:
        # Initialize generator
        generator = EigenCAMGenerator(args.model, device=args.device)
        
        # Process images
        stats = generator.process_batch(
            input_path=input_path,
            output_dir=args.output,
            target_layer=args.target,
            compare_mode=args.compare,
            compare_methods=args.compare_methods,
            method=args.method,
            max_images=args.max_images,
            compare_layers_override=args.compare_layers,
            method_compare_layer=args.method_target
        )
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.warning(f"Completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("All images processed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
