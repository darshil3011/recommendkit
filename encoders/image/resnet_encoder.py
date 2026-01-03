"""
ResNet-based image encoder using pretrained models from torchvision.
Lightweight alternative to Vision Transformers.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional, Union

from .base_image_encoder import BaseImageEncoder, AggregationStrategy


class ResNetWrapper(nn.Module):
    """ResNet wrapper with projection layer"""
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 embedding_dim: int = 256, 
                 pretrained: bool = True):
        """
        Initialize ResNet wrapper.
        
        Args:
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            embedding_dim: Output embedding dimension
            pretrained: Whether to use pretrained ImageNet weights
        """
        super().__init__()
        
        # Map model names to torchvision models and output dimensions
        model_configs = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, 2048),
        }
        
        model_name_lower = model_name.lower()
        if model_name_lower not in model_configs:
            raise ValueError(f"Unsupported ResNet model: {model_name}. Choose from {list(model_configs.keys())}")
        
        model_fn, weights_class, resnet_output_dim = model_configs[model_name_lower]
        
        # Load ResNet model
        if pretrained:
            self.resnet = model_fn(weights=weights_class)
        else:
            self.resnet = model_fn(weights=None)
        
        # Remove the final classification layer
        self.resnet.fc = nn.Identity()
        
        # Projection layer to desired embedding dimension
        self.projection = nn.Linear(resnet_output_dim, embedding_dim)
    
    def forward(self, x):
        """Forward pass through ResNet"""
        features = self.resnet(x)
        return self.projection(features)


class ResNetImageEncoder(BaseImageEncoder):
    """
    ResNet-based image encoder.
    Lightweight alternative to Vision Transformers with pretrained ImageNet weights.
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, AggregationStrategy] = AggregationStrategy.CONCAT,
                 embedding_dim: int = 256,
                 image_size: tuple = (224, 224),
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 num_image_fields: int = 2):
        """
        Initialize ResNet image encoder.
        
        Args:
            aggregation_strategy: How to combine multiple images
            embedding_dim: Output embedding dimension
            image_size: Input image size (height, width)
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
                       Default 'resnet18' is lightweight and fast
            pretrained: Whether to use pretrained ImageNet weights
            num_image_fields: Number of image fields to expect
        """
        super().__init__(aggregation_strategy, embedding_dim, image_size, num_image_fields)
        
        self.backbone = ResNetWrapper(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        
        # Initialize projection for concat strategy
        if self.aggregation_strategy == AggregationStrategy.CONCAT:
            self.field_projection = nn.Linear(self.num_image_fields * embedding_dim, embedding_dim)
        else:
            self.field_projection = None
    
    def _encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image tensor through backbone"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return self.backbone(image_tensor)
    
    def forward(self, image_data: Dict[str, Union[str, torch.Tensor, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for image encoding.
        
        Args:
            image_data: Dictionary mapping field names to image data
                       - {'field': '/path/to/image.jpg'} - single path
                       - {'field': torch.Tensor} - preprocessed tensor
                       - {'field': ['/path1.jpg', '/path2.jpg']} - batched paths
                       - {'field': None} - missing image
        
        Returns:
            Dictionary with image features: {"image_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        if not image_data:
            return {"image_features": self.default_embedding.unsqueeze(0)}
        
        # Check if batched
        first_value = next(iter(image_data.values()))
        is_batched = isinstance(first_value, list)
        batch_size = len(first_value) if is_batched and first_value else 1
        
        # Process each field
        field_embeddings = []
        for field_name, field_value in image_data.items():
            if field_value is None:
                embedding = self.default_embedding.unsqueeze(0).expand(batch_size, -1)
            elif isinstance(field_value, str):
                img_tensor = self._load_image(field_value)
                embedding = self._encode(img_tensor)
            elif isinstance(field_value, list):
                embeddings_list = []
                for path in field_value:
                    if path is not None:
                        img_tensor = self._load_image(path)
                        embeddings_list.append(self._encode(img_tensor))
                    else:
                        embeddings_list.append(self.default_embedding.unsqueeze(0))
                embedding = torch.cat(embeddings_list, dim=0) if embeddings_list else \
                           self.default_embedding.unsqueeze(0).expand(batch_size, -1)
            elif isinstance(field_value, torch.Tensor):
                if field_value.dim() == 4:
                    embedding = self._encode_batch(field_value)
                elif field_value.dim() == 3:
                    embedding = self._encode(field_value)
                else:
                    raise ValueError(f"Expected 3D or 4D tensor, got {field_value.dim()}D")
            else:
                raise ValueError(f"Unsupported data type: {type(field_value)}")
            
            field_embeddings.append(embedding)
        
        if not field_embeddings:
            return {"image_features": self.default_embedding.unsqueeze(0).expand(batch_size, -1)}
        
        # Combine field embeddings
        if len(field_embeddings) == 1:
            combined = field_embeddings[0]
        else:
            concatenated = torch.cat(field_embeddings, dim=-1)
            if self.field_projection is not None:
                combined = self.field_projection(concatenated)
            else:
                combined = concatenated
        
        # Apply aggregation if multiple images per field
        if combined.size(0) > 1 and self.aggregation_strategy != AggregationStrategy.CONCAT:
            if self.aggregation_strategy == AggregationStrategy.AVERAGE:
                combined = combined.mean(dim=0, keepdim=True)
            elif self.aggregation_strategy == AggregationStrategy.MAX_POOL:
                combined = combined.max(dim=0, keepdim=True)[0]
        
        return {"image_features": combined}
    
    def _encode_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encode batch of images"""
        if image_batch.size(0) == 0:
            return self.default_embedding.unsqueeze(0)
        
        # ResNet handles batches efficiently
        embeddings = self.backbone(image_batch)
        
        # Apply aggregation
        if self.aggregation_strategy == AggregationStrategy.CONCAT:
            return embeddings.flatten().unsqueeze(0) if embeddings.size(0) == 1 else embeddings
        elif self.aggregation_strategy == AggregationStrategy.AVERAGE:
            return embeddings.mean(dim=0, keepdim=True)
        elif self.aggregation_strategy == AggregationStrategy.MAX_POOL:
            return embeddings.max(dim=0, keepdim=True)[0]
        else:
            return embeddings

