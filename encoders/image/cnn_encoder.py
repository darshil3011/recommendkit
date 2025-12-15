"""
CNN-based image encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .base_image_encoder import BaseImageEncoder, AggregationStrategy


class SimpleCNN(nn.Module):
    """Simple CNN architecture"""
    
    def __init__(self, num_layers: int = 3, embedding_dim: int = 256):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        
        layers = []
        in_channels = 3
        channel_progression = [64, 128, 256, 512, 1024][:num_layers]
        
        for out_channels in channel_progression:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(in_channels, embedding_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = self.global_avg_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return self.projection(flattened)


class CNNImageEncoder(BaseImageEncoder):
    """
    CNN-based image encoder.
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, AggregationStrategy] = AggregationStrategy.CONCAT,
                 embedding_dim: int = 256,
                 num_cnn_layers: int = 3,
                 image_size: tuple = (224, 224),
                 num_image_fields: int = 2):
        """
        Initialize CNN image encoder.
        
        Args:
            aggregation_strategy: How to combine multiple images
            embedding_dim: Output embedding dimension
            num_cnn_layers: Number of CNN layers
            image_size: Input image size (height, width)
            num_image_fields: Number of image fields to expect
        """
        super().__init__(aggregation_strategy, embedding_dim, image_size, num_image_fields)
        
        self.backbone = SimpleCNN(num_layers=num_cnn_layers, embedding_dim=embedding_dim)
        
        # Initialize projection for concat strategy
        if aggregation_strategy == AggregationStrategy.CONCAT:
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
        
        embeddings = []
        for i in range(image_batch.size(0)):
            embeddings.append(self._encode(image_batch[i]))
        
        stacked = torch.cat(embeddings, dim=0)
        
        # Apply aggregation
        if self.aggregation_strategy == AggregationStrategy.CONCAT:
            return stacked.flatten().unsqueeze(0)
        elif self.aggregation_strategy == AggregationStrategy.AVERAGE:
            return stacked.mean(dim=0, keepdim=True)
        elif self.aggregation_strategy == AggregationStrategy.MAX_POOL:
            return stacked.max(dim=0, keepdim=True)[0]
        else:
            return stacked

