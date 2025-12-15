"""
Factory function for creating image encoders.
"""

from typing import Dict, Any
from .cnn_encoder import CNNImageEncoder
from .vit_encoder import ViTImageEncoder


def create_image_encoder(config: Dict[str, Any]) -> Any:
    """
    Factory function to create an image encoder based on configuration.
    
    Args:
        config: Configuration dictionary with:
            - model_type: 'cnn' or 'vit' (default: 'cnn')
            - aggregation_strategy: 'concat', 'average', or 'max_pool' (default: 'concat')
            - embedding_dim: Output embedding dimension (default: 256)
            - num_cnn_layers: Number of CNN layers (only for CNN, default: 3)
            - image_size: Input image size (default: (224, 224))
            - pretrained: Whether to use pretrained weights (only for ViT, default: True)
            - num_image_fields: Number of image fields (default: 2)
    
    Returns:
        Configured image encoder instance
    """
    model_type = config.get('model_type', 'cnn').lower()
    aggregation_strategy = config.get('aggregation_strategy', 'concat')
    embedding_dim = config.get('embedding_dim', 256)
    image_size = config.get('image_size', (224, 224))
    num_image_fields = config.get('num_image_fields', 2)
    
    if model_type == 'vit':
        pretrained = config.get('pretrained', True)
        return ViTImageEncoder(
            aggregation_strategy=aggregation_strategy,
            embedding_dim=embedding_dim,
            image_size=image_size,
            pretrained=pretrained,
            num_image_fields=num_image_fields
        )
    else:  # CNN
        num_cnn_layers = config.get('num_cnn_layers', 3)
        return CNNImageEncoder(
            aggregation_strategy=aggregation_strategy,
            embedding_dim=embedding_dim,
            num_cnn_layers=num_cnn_layers,
            image_size=image_size,
            num_image_fields=num_image_fields
        )

