"""
GradCAM Implementation for YOLO Classification
==============================================

Standard GradCAM implementation using gradients to weight activation maps.

Reference: https://arxiv.org/abs/1610.02391
"""

from yolo_cam.base_cam import BaseCAM
import numpy as np


class GradCAM(BaseCAM):
    """
    GradCAM: Gradient-weighted Class Activation Mapping.
    
    Uses gradients flowing into the final convolutional layer to produce
    a coarse localization map highlighting important regions.
    
    More precise than EigenCAM when model predictions are confident,
    but requires gradient computation.
    """
    
    def __init__(self, model, target_layers, task: str = 'cls',
                 reshape_transform=None):
        """
        Initialize GradCAM.
        
        Args:
            model: YOLO model
            target_layers: List of target layers for CAM generation
            task: Task type (default: 'cls' for classification)
            reshape_transform: Optional transform for activations
        """
        super(GradCAM, self).__init__(
            model,
            target_layers,
            task,
            reshape_transform,
            uses_gradients=True  # GradCAM requires gradients
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        """
        Calculate GradCAM weights using global average pooling of gradients.
        
        Args:
            input_tensor: Input tensor
            target_layer: Target layer
            targets: Target classes
            activations: Layer activations
            grads: Gradients with respect to activations
            
        Returns:
            Weights for each channel
        """
        # Global average pooling of gradients
        # This weights each channel by its gradient importance
        return np.mean(grads, axis=(2, 3))
