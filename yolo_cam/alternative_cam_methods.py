"""
Alternative CAM Methods for YOLO Classification
===============================================

If EigenCAM still doesn't work well for your use case, you can implement
these alternative methods. Each has different strengths.

RECOMMENDATION: Start with EigenCAM (gradient-free, fast) but if results
are poor, try GradCAM++ (most accurate but needs gradients).
"""

# Example: GradCAM implementation for YOLO
# Add this to yolo_cam/ directory as grad_cam.py

from yolo_cam.base_cam import BaseCAM
import numpy as np


class GradCAM(BaseCAM):
    """
    GradCAM: Uses gradients to weight activation maps.
    
    More precise than EigenCAM but requires gradient computation.
    Works best when model predictions are confident and correct.
    
    Reference: https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, target_layers, task: str = 'cls',
                 reshape_transform=None):
        super(GradCAM, self).__init__(
            model,
            target_layers,
            task,
            reshape_transform,
            uses_gradients=True  # Requires gradients!
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # Global average pooling of gradients
        # This weights each channel by its gradient importance
        return np.mean(grads, axis=(2, 3))


class GradCAMPlusPlus(BaseCAM):
    """
    GradCAM++: Improved version with better localization.
    
    Uses weighted combination of gradients for better multi-instance
    detection and localization. Best overall performance but slower.
    
    Reference: https://arxiv.org/abs/1710.11063
    """
    def __init__(self, model, target_layers, task: str = 'cls',
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(
            model,
            target_layers,
            task,
            reshape_transform,
            uses_gradients=True
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # Compute alpha weights (equation from paper)
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-10
        
        aij = grads_power_2 / (
            2 * grads_power_2 + 
            sum_activations[:, :, None, None] * grads_power_3 + eps
        )
        
        # Only positive gradients contribute
        aij = np.where(grads != 0, aij, 0)
        
        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        
        return weights


# =============================================================================
# HOW TO USE THE ALTERNATIVE METHODS
# =============================================================================

def apply_gradcam(model, image_path, output_dir, target=-2, task='cls', show=False):
    """
    Use GradCAM instead of EigenCAM.
    
    GradCAM often provides better localization when:
    - Model predictions are correct and confident
    - You need precise object localization
    - EigenCAM shows unclear or scattered attention
    
    Drawback: Slightly slower due to gradient computation
    """
    from pathlib import Path
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from yolo_cam.utils.image import show_cam_on_image
    # from yolo_cam.grad_cam import GradCAM  # Use the code above
    
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    target_layers = [model.model.model[target]]
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = np.float32(img_rgb) / 255.0
    
    # Use GradCAM instead of EigenCAM
    cam = GradCAM(model, target_layers, task=task)
    grayscale_cam = cam(img_rgb)[0, :, :]
    grayscale_cam = np.clip(grayscale_cam, 0, 1)
    
    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    
    # Save result
    plt.figure(figsize=(10, 10))
    plt.imshow(cam_image)
    plt.title(f"GradCAM for {image_path.name}")
    plt.axis('off')
    
    if show:
        plt.show()
    
    cam_image_save_path = output_dir / f"gradcam_{image_path.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(cam_image_save_path), cam_image)
    plt.close()


# =============================================================================
# COMPARISON OF CAM METHODS
# =============================================================================

"""
┌─────────────────┬──────────────┬───────────┬─────────────┬──────────────┐
│ Method          │ Needs Grads? │ Speed     │ Accuracy    │ Best For     │
├─────────────────┼──────────────┼───────────┼─────────────┼──────────────┤
│ EigenCAM        │ No           │ Very Fast │ Good        │ Quick tests, │
│ (current)       │              │           │             │ exploration  │
├─────────────────┼──────────────┼───────────┼─────────────┼──────────────┤
│ GradCAM         │ Yes          │ Fast      │ Very Good   │ Single obj,  │
│                 │              │           │             │ clear cases  │
├─────────────────┼──────────────┼───────────┼─────────────┼──────────────┤
│ GradCAM++       │ Yes          │ Medium    │ Excellent   │ Complex      │
│                 │              │           │             │ scenes, best │
│                 │              │           │             │ localization │
├─────────────────┼──────────────┼───────────┼─────────────┼──────────────┤
│ ScoreCAM        │ No           │ Slow      │ Very Good   │ When grads   │
│                 │              │           │             │ unavailable  │
└─────────────────┴──────────────┴───────────┴─────────────┴──────────────┘

RECOMMENDATION FOR YOLO CLASSIFICATION:
1. Start with improved EigenCAM (fastest, good enough for most cases)
2. If quality is poor, try GradCAM (better but needs gradients)
3. For publication/critical work, use GradCAM++ (best quality)
"""


# =============================================================================
# DEBUGGING TIPS
# =============================================================================

def debug_cam_quality(model, image_path, output_dir):
    """
    Generate visualizations with multiple methods and layers for comparison.
    Helps identify the best approach for your specific dataset.
    """
    from pathlib import Path
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from yolo_cam.eigen_cam import EigenCAM
    from yolo_cam.utils.image import show_cam_on_image
    
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = np.float32(img_rgb) / 255.0
    
    # Test multiple target layers
    target_indices = [-1, -2, -3, -4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, target_idx in enumerate(target_indices):
        try:
            target_layers = [model.model.model[target_idx]]
            cam = EigenCAM(model, target_layers, task='cls')
            grayscale_cam = cam(img_rgb)[0, :, :]
            grayscale_cam = np.clip(grayscale_cam, 0, 1)
            
            cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
            
            axes[idx].imshow(cam_image)
            axes[idx].set_title(f"Layer {target_idx}")
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Failed: {str(e)}", 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.suptitle(f"CAM Comparison for {image_path.name}", fontsize=16)
    plt.tight_layout()
    
    save_path = output_dir / f"debug_comparison_{image_path.name}"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved to: {save_path}")


# =============================================================================
# USAGE IN YOUR PIPELINE
# =============================================================================

"""
To use these alternative methods in your pipeline:

1. Save the GradCAM/GradCAMPlusPlus classes to:
   yolo_cam/grad_cam.py

2. Modify yolo_complete_cls_pipeline.py:

   # Option A: Replace EigenCAM with GradCAM
   from yolo_cam.grad_cam import GradCAM
   cam = GradCAM(model, target_layers, task=task)

   # Option B: Try multiple methods
   from yolo_cam.eigen_cam import EigenCAM
   from yolo_cam.grad_cam import GradCAM, GradCAMPlusPlus
   
   for cam_method, name in [(EigenCAM, 'eigen'), 
                             (GradCAM, 'grad'),
                             (GradCAMPlusPlus, 'gradpp')]:
       cam = cam_method(model, target_layers, task=task)
       grayscale_cam = cam(img_rgb)[0, :, :]
       # Save with different name
       cam_path = output_dir / f"{name}_cam_{image_path.name}"

3. Run debug comparison first:
   debug_cam_quality(model, "path/to/test_image.jpg", "debug_output")
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nThis file contains reference implementations.")
    print("See comments above for integration instructions.")
