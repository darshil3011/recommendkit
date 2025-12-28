"""
Base continuous encoder class.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from encoders.base_encoder import BaseEncoder


class BaseContinuousEncoder(BaseEncoder):
    """
    Base class for continuous encoders.
    Handles common continuous feature processing logic.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 normalize: bool = True):
        """
        Initialize base continuous encoder.
        
        Args:
            embedding_dim: Output embedding dimension
            normalize: Whether to normalize inputs
        """
        super().__init__(embedding_dim)
        self.normalize = normalize
    
    def _normalize_features(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize feature tensor"""
        if self.normalize:
            # Clamp extreme values
            feature_tensor = torch.clamp(feature_tensor, min=-1e6, max=1e6)
            
            # Log scaling for large values
            abs_features = torch.abs(feature_tensor)
            large_mask = abs_features > 1000
            if large_mask.any():
                sign = torch.sign(feature_tensor[large_mask])
                feature_tensor[large_mask] = sign * torch.log1p(abs_features[large_mask])
            
            # Standardize
            if feature_tensor.dim() == 1:
                mean = feature_tensor.mean()
                std = feature_tensor.std(unbiased=False) + 1e-8
                feature_tensor = (feature_tensor - mean) / std
            else:
                mean = feature_tensor.mean(dim=0, keepdim=True)
                std = feature_tensor.std(dim=0, keepdim=True, unbiased=False) + 1e-8
                feature_tensor = (feature_tensor - mean) / std
        
        return feature_tensor
    
    def forward(self, continuous_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for continuous feature encoding.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")


