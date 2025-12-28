"""
Base temporal encoder class.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from encoders.base_encoder import BaseEncoder


class TemporalAggregationStrategy(Enum):
    """Supported aggregation strategies for temporal sequences"""
    LAST_HIDDEN = "last_hidden"        # Use last hidden state
    MEAN_POOLING = "mean_pooling"      # Average all hidden states
    MAX_POOLING = "max_pooling"        # Max pool all hidden states
    ATTENTION = "attention"            # Attention over hidden states


class ModalityType(Enum):
    """Types of modalities to encode from items"""
    IMAGE = "image"
    TEXT = "text"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class ItemLookupInterface:
    """
    Abstract interface for looking up item features
    Users must implement this to provide item data
    """
    
    def get_item_features(self, item_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get features for a specific item
        
        Args:
            item_id: ID of the item to lookup
            
        Returns:
            Dictionary containing item features:
            {
                'image': {'main_image': '/path/to/image.jpg', ...},
                'text': {'title': 'Product Title', 'description': '...'},
                'categorical': {'category': 'electronics', 'brand': 'Apple'},
                'continuous': {'price': 99.99, 'rating': 4.5}
            }
        """
        raise NotImplementedError("Users must implement this method")
    
    def batch_get_item_features(self, item_ids: List[Union[str, int]]) -> List[Dict[str, Any]]:
        """
        Get features for multiple items (can be optimized for batch retrieval)
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            List of feature dictionaries in the same order as item_ids
        """
        return [self.get_item_features(item_id) for item_id in item_ids]


class BaseTemporalEncoder(BaseEncoder):
    """
    Base class for temporal encoders.
    Handles sequential item interaction encoding.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 aggregation_strategy: Union[str, TemporalAggregationStrategy] = TemporalAggregationStrategy.LAST_HIDDEN,
                 max_sequence_length: int = 50):
        """
        Initialize base temporal encoder.
        
        Args:
            embedding_dim: Output embedding dimension
            aggregation_strategy: How to aggregate sequence outputs
            max_sequence_length: Maximum sequence length
        """
        super().__init__(embedding_dim)
        
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = TemporalAggregationStrategy(aggregation_strategy.lower())
        
        self.aggregation_strategy = aggregation_strategy
        self.max_sequence_length = max_sequence_length
    
    def forward(self, temporal_dict: Dict[str, Union[List[Union[str, int]], List[List[Union[str, int]]]]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal encoding - must be implemented by subclasses.
        
        Args:
            temporal_dict: Dictionary mapping temporal field names to item ID lists
                          Single sample: {'prev_50_posts': [34, 56, 7646, 342]}
                          Batched: {'prev_50_posts': [[34, 56], [123, 456], [789, 101]]}
        
        Returns:
            Dictionary with temporal features: {"temporal": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement forward()")


