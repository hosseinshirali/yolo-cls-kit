import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_2d_projection(activation_batch):
    """
    Compute 2D projection of activations using SVD (Singular Value Decomposition).
    
    This function projects high-dimensional activation tensors onto their principal
    component to create saliency maps for visualization.
    
    Args:
        activation_batch: Batch of activations with shape (batch, channels, height, width)
        
    Returns:
        Projections as float32 array with shape (batch, height, width)
        
    Note:
        - Handles NaN values by replacing with 0
        - Centers activations before SVD
        - Applies sign correction to handle SVD ambiguity
        - Uses absolute values to focus on magnitude of contribution
    """
    # Handle NaN values
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    
    for activations in activation_batch:
        # Reshape: (channels, height, width) -> (height*width, channels)
        reshaped_activations = activations.reshape(
            activations.shape[0], -1).transpose()
        
        # Centering is crucial for SVD - removes mean to focus on variations
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        
        # Add small epsilon to avoid numerical instability
        reshaped_activations = reshaped_activations + 1e-10
        
        # Perform SVD: U (spatial components), S (singular values), VT (channel components)
        try:
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=False)
        except np.linalg.LinAlgError as e:
            logger.warning(f"SVD failed: {e}. Using zero projection.")
            projection = np.zeros(activations.shape[1:])
            projections.append(projection)
            continue
        
        # Project onto first principal component (most important direction)
        projection = reshaped_activations @ VT[0, :]
        
        # CRITICAL FIX: Handle sign ambiguity in SVD
        # SVD can return arbitrary signs - we need to check which direction
        # is more "positive" to avoid inverted heatmaps
        
        # Method 1: Use absolute values (focuses on magnitude regardless of sign)
        # This is often better for visualization as we care about "importance" not "direction"
        projection = np.abs(projection)
        
        # Method 2 (Alternative): Sign correction based on mean
        # Uncomment if you prefer this approach:
        # if projection.mean() < 0:
        #     projection = -projection
        
        # Reshape back to spatial dimensions
        projection = projection.reshape(activations.shape[1:])
        
        # Normalize projection to [0, 1] range for consistent visualization
        proj_min = projection.min()
        proj_max = projection.max()
        if proj_max - proj_min > 1e-10:  # Avoid division by zero
            projection = (projection - proj_min) / (proj_max - proj_min)
        else:
            # If projection is constant, set to zero
            projection = np.zeros_like(projection)
        
        projections.append(projection)
    
    return np.float32(np.array(projections))
