"""
Base encoder class for all modality encoders.
Provides common interface and functionality.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseEncoder(nn.Module):
    """
    Base class for all encoders in the recommendation system.
    
    All encoders should inherit from this class and implement:
    - forward(): Process input and return features
    - get_output_dim(): Return output embedding dimension
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize base encoder.
        
        Args:
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass - must be implemented by subclasses.
        
        Returns:
            Dictionary with feature tensor(s)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_output_dim(self) -> int:
        """
        Get output embedding dimension.
        
        Returns:
            Output dimension
        """
        return self.embedding_dim



