"""
Base categorical encoder class.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import hashlib

from encoders.base_encoder import BaseEncoder


class CategoricalAggregationStrategy(Enum):
    """Supported aggregation strategies"""
    SEPARATE_CONCAT = "separate_concat"
    JOINT_EMBEDDING = "joint_embedding"


class BaseCategoricalEncoder(BaseEncoder):
    """
    Base class for categorical encoders.
    Handles common categorical encoding logic.
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, CategoricalAggregationStrategy],
                 embedding_dim: int,
                 hash_vocab_size: int = 10000,
                 hash_seed: int = 42,
                 num_categorical_fields: int = 3):
        """
        Initialize base categorical encoder.
        
        Args:
            aggregation_strategy: How to combine multiple categorical fields
            embedding_dim: Output embedding dimension
            hash_vocab_size: Size of hash vocabulary
            hash_seed: Seed for reproducible hashing
            num_categorical_fields: Number of categorical fields to expect
        """
        super().__init__(embedding_dim)
        
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = CategoricalAggregationStrategy(aggregation_strategy.lower())
        
        self.aggregation_strategy = aggregation_strategy
        self.hash_vocab_size = hash_vocab_size
        self.hash_seed = hash_seed
        self.num_categorical_fields = max(num_categorical_fields, 1)
        
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def _hash_category(self, field_name: str, category_value: Any) -> int:
        """Create deterministic hash for category value"""
        if category_value is None:
            hash_input = f"{field_name}:<NULL>"
        else:
            str_value = str(category_value).strip().lower()
            hash_input = f"{field_name}:{str_value}"
        
        hash_obj = hashlib.sha256(f"{self.hash_seed}:{hash_input}".encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.hash_vocab_size
    
    def _hash_joint(self, categorical_dict: Dict[str, Any]) -> int:
        """Create joint hash for all categorical features"""
        sorted_items = sorted(categorical_dict.items())
        combined_parts = []
        for field_name, category_value in sorted_items:
            if category_value is None:
                combined_parts.append(f"{field_name}:<NULL>")
            else:
                str_value = str(category_value).strip().lower()
                combined_parts.append(f"{field_name}:{str_value}")
        
        combined_string = "|".join(combined_parts)
        hash_obj = hashlib.sha256(f"{self.hash_seed}:{combined_string}".encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.hash_vocab_size
    
    def forward(self, categorical_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for categorical encoding.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")



