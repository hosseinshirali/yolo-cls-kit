"""
GradCAM++ Implementation for YOLO Classification
================================================

GradCAM++ is an improved version of GradCAM that provides better localization,
especially for images with multiple instances of the same class.

Reference: https://arxiv.org/abs/1710.11063
"""

from yolo_cam.base_cam import BaseCAM
import numpy as np


class GradCAMPlusPlus(BaseCAM):
    """
    GradCAM++: Improved Gradient-weighted Class Activation Mapping.
    
    Uses weighted combination of gradients for better localization compared to
    standard GradCAM. Particularly effective for:
    - Multiple instances of the same class
    - Better localization of small objects
    - More robust to model variations
    
    Note: Requires gradient computation, so it's slower than EigenCAM but
    typically more accurate.
    """
    
    def __init__(self, model, target_layers, task: str = 'cls',
                 reshape_transform=None):
        """
        Initialize GradCAM++.
        
        Args:
            model: YOLO model
            target_layers: List of target layers for CAM generation
            task: Task type (default: 'cls' for classification)
            reshape_transform: Optional transform for activations
        """
        super(GradCAMPlusPlus, self).__init__(
            model,
            target_layers,
            task,
            reshape_transform,
            uses_gradients=True  # GradCAM++ requires gradients
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        """
        Calculate GradCAM++ weights using improved gradient weighting.
        
        The key innovation of GradCAM++ is using pixel-wise weighting of
        gradients based on their second and third order derivatives.
        
        Args:
            input_tensor: Input tensor
            target_layer: Target layer
            targets: Target classes
            activations: Layer activations
            grads: Gradients with respect to activations
            
        Returns:
            Weights for each channel
        """
        # Calculate powers of gradients
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        
        # Sum activations over spatial dimensions
        sum_activations = np.sum(activations, axis=(2, 3))
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        
        # Calculate alpha weights (pixel-wise importance)
        # This is the core GradCAM++ formula
        aij = grads_power_2 / (
            2 * grads_power_2 + 
            sum_activations[:, :, None, None] * grads_power_3 + eps
        )
        
        # Only consider pixels where gradient is non-zero
        aij = np.where(grads != 0, aij, 0)
        
        # Weight by ReLU(gradient) - only positive contributions
        weights = np.maximum(grads, 0) * aij
        
        # Global average pooling to get channel weights
        weights = np.sum(weights, axis=(2, 3))
        
        return weights
