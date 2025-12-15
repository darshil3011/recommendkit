"""
Base text encoder class for all text encoders.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from enum import Enum

from encoders.base_encoder import BaseEncoder


class TextAggregationStrategy(Enum):
    """Supported aggregation strategies for multiple text fields"""
    SEPARATE_CONCAT = "separate_concat"  # Encode each field separately, then concatenate
    JOINT_ENCODING = "joint_encoding"    # Concatenate raw text, then encode jointly
    MEAN = "mean"                        # Encode each field separately, then average embeddings


class BaseTextEncoder(BaseEncoder):
    """
    Base class for all text encoders.
    
    Subclasses must implement forward() to handle text encoding.
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, TextAggregationStrategy],
                 embedding_dim: int,
                 num_text_fields: int = 2):
        """
        Initialize base text encoder.
        
        Args:
            aggregation_strategy: How to combine multiple text fields
            embedding_dim: Output embedding dimension
            num_text_fields: Number of text fields to expect
        """
        super().__init__(embedding_dim)
        
        # Convert string to enum
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = TextAggregationStrategy(aggregation_strategy.lower())
        
        self.aggregation_strategy = aggregation_strategy
        self.num_text_fields = max(num_text_fields, 1)
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def forward(self, text_dict: Dict[str, Union[str, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text encoding - must be implemented by subclasses.
        
        Args:
            text_dict: Dictionary mapping text field names to content
                      Single: {'bio': 'text', 'summary': 'text'}
                      Batched: {'bio': ['text1', 'text2'], 'summary': ['text3', 'text4']}
        
        Returns:
            Dictionary with text features: {"text_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement forward()")
