"""
Base image encoder class.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Union
from enum import Enum
import warnings

from encoders.base_encoder import BaseEncoder


class AggregationStrategy(Enum):
    """Supported aggregation strategies for multiple images"""
    CONCAT = "concat"
    AVERAGE = "average"
    MAX_POOL = "max_pool"


class BaseImageEncoder(BaseEncoder):
    """
    Base class for image encoders.
    Handles common image processing logic.
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, AggregationStrategy],
                 embedding_dim: int,
                 image_size: tuple = (224, 224),
                 num_image_fields: int = 2):
        """
        Initialize base image encoder.
        
        Args:
            aggregation_strategy: How to combine multiple images
            embedding_dim: Output embedding dimension
            image_size: Input image size (height, width)
            num_image_fields: Number of image fields to expect
        """
        super().__init__(embedding_dim)
        
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = AggregationStrategy(aggregation_strategy.lower())
        
        self.aggregation_strategy = aggregation_strategy
        self.image_size = image_size
        self.num_image_fields = max(num_image_fields, 1)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image from path"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, *self.image_size)
    
    def forward(self, image_data: Dict[str, Union[str, torch.Tensor, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for image encoding.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")

