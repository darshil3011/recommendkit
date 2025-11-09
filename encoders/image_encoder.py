import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
import warnings

class AggregationStrategy(Enum):
    """Supported aggregation strategies for multiple images"""
    CONCAT = "concat"
    AVERAGE = "average"
    MAX_POOL = "max_pool"


class ModelType(Enum):
    """Supported model architectures"""
    CNN = "cnn"
    VIT = "vit"


class SimpleCNN(nn.Module):
    """Simple CNN architecture with configurable layers"""
    
    def __init__(self, num_layers: int = 3, embedding_dim: int = 256):
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        
        layers = []
        in_channels = 3  # RGB images
        
        # Define channel progression
        channel_progression = [64, 128, 256, 512, 1024][:num_layers]
        
        # Convolutional layers
        for i, out_channels in enumerate(channel_progression):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to desired embedding dimension
        self.projection = nn.Linear(in_channels, embedding_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        features = self.feature_extractor(x)
        # Global average pooling
        pooled = self.global_avg_pool(features)
        # Flatten
        flattened = pooled.view(pooled.size(0), -1)
        # Project to embedding dimension
        embedding = self.projection(flattened)
        return embedding


class ViTWrapper(nn.Module):
    """Vision Transformer wrapper"""
    
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        try:
            # Try to use timm for ViT (more comprehensive)
            import timm
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            vit_output_dim = self.vit.num_features
        except ImportError:
            # Fallback to torchvision ViT
            warnings.warn("timm not available, using torchvision ViT. Install timm for better ViT support.")
            if pretrained:
                self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.vit = models.vit_b_16(weights=None)
            
            # Remove the classification head
            self.vit.heads = nn.Identity()
            vit_output_dim = 768  # ViT-B/16 hidden dimension
        
        # Projection layer to desired embedding dimension
        self.projection = nn.Linear(vit_output_dim, embedding_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        features = self.vit(x)
        embedding = self.projection(features)
        return embedding


class ImageEncoder(nn.Module):
    """
    Image encoder that handles multiple images with different aggregation strategies
    
    Args:
        aggregation_strategy: How to combine multiple images ('concat', 'average', 'max_pool')
        model_type: Architecture to use ('cnn' or 'vit')
        embedding_dim: Output embedding dimension
        num_cnn_layers: Number of CNN layers (only used if model_type='cnn')
        image_size: Input image size (height, width)
        pretrained: Whether to use pretrained weights (for ViT)
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, AggregationStrategy] = AggregationStrategy.CONCAT,
                 model_type: Union[str, ModelType] = ModelType.CNN,
                 embedding_dim: int = 256,
                 num_cnn_layers: int = 3,
                 image_size: tuple = (224, 224),
                 pretrained: bool = True,
                 num_image_fields: int = 2):
        
        super().__init__()
        
        # Convert string arguments to enums
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = AggregationStrategy(aggregation_strategy.lower())
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
            
        self.aggregation_strategy = aggregation_strategy
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.num_image_fields = max(num_image_fields, 1)  # At least 1 field
        
        # Initialize the backbone model
        if model_type == ModelType.CNN:
            self.backbone = SimpleCNN(num_layers=num_cnn_layers, embedding_dim=embedding_dim)
            self.backbone_output_dim = embedding_dim
        elif model_type == ModelType.VIT:
            self.backbone = ViTWrapper(embedding_dim=embedding_dim, pretrained=pretrained)
            self.backbone_output_dim = embedding_dim
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate final output dimension based on aggregation strategy
        if aggregation_strategy == AggregationStrategy.CONCAT:
            # Use field count - NO MORE LAZY INITIALIZATION!
            self.output_dim = self.num_image_fields * embedding_dim
            # Initialize projection layer immediately
            self.field_projection = nn.Linear(self.output_dim, embedding_dim)
            self._projection_initialized = True
        else:  # AVERAGE or MAX_POOL
            self.output_dim = embedding_dim
            self.field_projection = None
            self._projection_initialized = True
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # For handling missing images
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            # Return a zero tensor with correct shape
            return torch.zeros(3, *self.image_size)
    
    def _initialize_field_projection(self, num_fields: int):
        """Initialize the projection layer for multiple image fields"""
        if num_fields > 1:
            input_dim = num_fields * self.embedding_dim
            self.field_projection = nn.Linear(input_dim, self.embedding_dim)
        self._projection_initialized = True
    
    def _encode_single_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a single image tensor"""
        if image_tensor.dim() == 3:  # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
        
        return self.backbone(image_tensor)
    
    def forward_from_paths(self, image_paths_dict: Dict[str, Optional[str]]) -> torch.Tensor:
        """
        Forward pass from image paths
        
        Args:
            image_paths_dict: Dictionary mapping image names to file paths
                             e.g., {'profile_pic': '/path/to/img1.jpg', 'cover_photo': '/path/to/img2.jpg'}
        
        Returns:
            Combined embedding tensor
        """
        # Load and preprocess all images
        image_tensors = []
        image_names = []
        
        for img_name, img_path in image_paths_dict.items():
            if img_path is not None:
                img_tensor = self._load_and_preprocess_image(img_path)
                image_tensors.append(img_tensor)
                image_names.append(img_name)
            else:
                # Handle missing images with default embedding
                image_tensors.append(torch.zeros(3, *self.image_size))
                image_names.append(f"{img_name}_missing")
        
        if not image_tensors:
            # No images provided, return default embedding
            return self.default_embedding.unsqueeze(0)
        
        # Stack into batch
        batch_tensor = torch.stack(image_tensors)  # (num_images, 3, H, W)
        
        return self._forward_batch(batch_tensor)
    
    def forward_from_tensors(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from preprocessed image tensors
        
        Args:
            image_tensors: Tensor of shape (num_images, 3, height, width)
        
        Returns:
            Combined embedding tensor
        """
        return self._forward_batch(image_tensors)
    
    def _forward_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of images and apply aggregation
        
        Args:
            image_batch: Tensor of shape (num_images, 3, height, width)
        
        Returns:
            Aggregated embedding tensor
        """
        if image_batch.size(0) == 0:
            return self.default_embedding.unsqueeze(0)
        
        # Encode each image
        embeddings = []
        for i in range(image_batch.size(0)):
            img_embedding = self._encode_single_image(image_batch[i])
            embeddings.append(img_embedding)
        
        # Stack embeddings: (num_images, embedding_dim)
        stacked_embeddings = torch.cat(embeddings, dim=0)
        
        # Apply aggregation strategy
        if self.aggregation_strategy == AggregationStrategy.CONCAT:
            # Concatenate all embeddings
            return stacked_embeddings.flatten().unsqueeze(0)  # (1, num_images * embedding_dim)
        
        elif self.aggregation_strategy == AggregationStrategy.AVERAGE:
            # Average pooling across images
            return stacked_embeddings.mean(dim=0, keepdim=True)  # (1, embedding_dim)
        
        elif self.aggregation_strategy == AggregationStrategy.MAX_POOL:
            # Max pooling across images
            return stacked_embeddings.max(dim=0, keepdim=True)[0]  # (1, embedding_dim)
        
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
    
    def get_output_dim(self, num_images: int) -> int:
        """
        Get the output dimension for a given number of images
        
        Args:
            num_images: Number of input images
            
        Returns:
            Output embedding dimension
        """
        if self.aggregation_strategy == AggregationStrategy.CONCAT:
            return num_images * self.embedding_dim
        else:  # AVERAGE or MAX_POOL
            return self.embedding_dim
    
    def forward(self, image_data: Dict[str, Union[str, torch.Tensor, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Standard PyTorch forward method that handles different input formats
        
        Args:
            image_data: Dictionary containing image data. Can have different formats:
                       - {'field_name': '/path/to/image.jpg'} for single image paths
                       - {'field_name': torch.Tensor} for preprocessed tensors
                       - {'field_name': ['/path1.jpg', '/path2.jpg']} for multiple image paths
                       - {'field_name': None} for missing images
        
        Returns:
            Dictionary with encoded image features: {'image_features': torch.Tensor}
        """
        if not isinstance(image_data, dict):
            raise ValueError(f"Expected dict input, got {type(image_data)}")
        
        # Determine if this is batched input and get batch size
        first_value = next(iter(image_data.values())) if image_data else None
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            # Get batch size from the first non-empty list
            batch_size = 0
            for field_value in image_data.values():
                if isinstance(field_value, list) and len(field_value) > 0:
                    batch_size = len(field_value)
                    break
            if batch_size == 0:
                batch_size = 1  # Fallback
        else:
            batch_size = 1
        
        all_embeddings = []
        field_names = []
        
        for field_name, field_value in image_data.items():
            if field_value is None:
                # Handle missing image data with correct batch size
                embedding = self.default_embedding.unsqueeze(0).expand(batch_size, -1)
            
            elif isinstance(field_value, str):
                # Single image path
                embedding = self.forward_from_paths({field_name: field_value})
            
            elif isinstance(field_value, list):
                # Multiple image paths or batched data
                if all(isinstance(item, str) or item is None for item in field_value):
                    # List of paths (batched data) - process each item individually and stack
                    embeddings_list = []
                    for path in field_value:
                        if path is not None:
                            # Process single image for this batch item
                            single_embedding = self.forward_from_paths({field_name: path})
                            embeddings_list.append(single_embedding)
                        else:
                            # Use default embedding for missing image
                            embeddings_list.append(self.default_embedding.unsqueeze(0))
                    
                    if embeddings_list:
                        # Stack along batch dimension to get [batch_size, embedding_dim]
                        embedding = torch.cat(embeddings_list, dim=0)
                    else:
                        # All None values in batch
                        embedding = self.default_embedding.unsqueeze(0).expand(batch_size, -1)
                else:
                    raise ValueError(f"Expected list of strings or None for field {field_name}, got mixed types")
            
            elif isinstance(field_value, torch.Tensor):
                # Preprocessed tensor
                if field_value.dim() == 4:  # (batch, channels, height, width)
                    embedding = self.forward_from_tensors(field_value)
                elif field_value.dim() == 3:  # (channels, height, width) - single image
                    embedding = self.forward_from_tensors(field_value.unsqueeze(0))
                else:
                    raise ValueError(f"Expected 3D or 4D tensor for field {field_name}, got {field_value.dim()}D")
            
            else:
                raise ValueError(f"Unsupported data type for field {field_name}: {type(field_value)}")
            
            all_embeddings.append(embedding)
            field_names.append(field_name)
        
        if not all_embeddings:
            # No image data provided, return default with correct batch size
            return {"image_features": self.default_embedding.unsqueeze(0).expand(batch_size, -1)}
        
        # Combine all field embeddings
        if len(all_embeddings) == 1:
            combined_embedding = all_embeddings[0]
        else:
            # Concatenate embeddings from different fields
            concatenated = torch.cat(all_embeddings, dim=-1)
            
            # Initialize and apply projection layer if needed
            if not self._projection_initialized:
                self._initialize_field_projection(len(all_embeddings))
            
            if self.field_projection is not None:
                combined_embedding = self.field_projection(concatenated)
            else:
                combined_embedding = concatenated
        
        return {"image_features": combined_embedding}

    def set_default_embedding(self, embedding: torch.Tensor):
        """Set the default embedding for missing images"""
        if embedding.size(0) != self.embedding_dim:
            raise ValueError(f"Default embedding dimension {embedding.size(0)} doesn't match expected {self.embedding_dim}")
        self.default_embedding = embedding


# Utility functions
def create_image_encoder(aggregation_strategy: str = "concat",
                        model_type: str = "cnn", 
                        embedding_dim: int = 256,
                        num_cnn_layers: int = 3,
                        image_size: tuple = (224, 224),
                        pretrained: bool = True,
                        num_image_fields: int = 2) -> ImageEncoder:
    """
    Factory function to create an ImageEncoder
    
    Args:
        aggregation_strategy: 'concat', 'average', or 'max_pool'
        model_type: 'cnn' or 'vit'
        embedding_dim: Output embedding dimension
        num_cnn_layers: Number of CNN layers (ignored for ViT)
        image_size: Input image size as (height, width)
        pretrained: Whether to use pretrained weights
        num_image_fields: Number of image fields to expect (eliminates lazy initialization)
        
    Returns:
        Configured ImageEncoder instance
    """
    return ImageEncoder(
        aggregation_strategy=aggregation_strategy,
        model_type=model_type,
        embedding_dim=embedding_dim,
        num_cnn_layers=num_cnn_layers,
        image_size=image_size,
        pretrained=pretrained,
        num_image_fields=num_image_fields
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    
    print("Testing Image Encoders...")
    
    # Create different encoder configurations
    configs = [
        {"aggregation_strategy": "concat", "model_type": "cnn", "embedding_dim": 128, "num_cnn_layers": 2},
        {"aggregation_strategy": "average", "model_type": "cnn", "embedding_dim": 256, "num_cnn_layers": 3},
        {"aggregation_strategy": "max_pool", "model_type": "vit", "embedding_dim": 512},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        try:
            encoder = create_image_encoder(**config)
            
            # Test with dummy data
            dummy_image_paths = {
                'profile_pic': None,  # Simulate missing image
                'cover_photo': None,  # Would be actual path in real usage
                'thumbnail': None
            }
            
            # For testing, we'll create dummy tensors instead of loading actual images
            dummy_images = torch.randn(3, 3, 224, 224)  # 3 images, 3 channels, 224x224
            
            # Test forward pass
            with torch.no_grad():
                output = encoder.forward_from_tensors(dummy_images)
                print(f"Input shape: {dummy_images.shape}")
                print(f"Output shape: {output.shape}")
                print(f"Expected output dim for 3 images: {encoder.get_output_dim(3)}")
                
        except Exception as e:
            print(f"Error with configuration {config}: {e}")
    
    print("\nAll tests completed!")