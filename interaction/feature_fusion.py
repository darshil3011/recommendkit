"""
feature_fusion.py
Generates combined user and item embeddings using simple concatenation-based fusion
(Attention-based fusion code is commented out but preserved)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple

# Import helper components
from .attention_utils import (
    FeatureFusionLayer,  # Kept for backward compatibility
    FeatureType, 
    create_feature_type_mapping,
    validate_feature_embeddings
)


class SimpleFusionLayer(nn.Module):
    """
    Simple feature fusion using concatenation + MLP projection
    Much more stable than attention for few features (3-4 tokens)
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Dimension of feature embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # We'll dynamically create the projection layer when we see the first batch
        self.projection = None
        self._projection_initialized = False
    
    def _initialize_projection(self, num_features: int):
        """Initialize projection layer based on number of features"""
        input_dim = num_features * self.embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
        self._projection_initialized = True
    
    def forward(self, 
                feature_embeddings: Dict[str, torch.Tensor],
                feature_types: Dict[str, FeatureType]) -> torch.Tensor:
        """
        Args:
            feature_embeddings: Dict mapping feature names to embeddings [batch_size, embedding_dim]
            feature_types: Dict mapping feature names to types (ignored, kept for compatibility)
        
        Returns:
            fused_embedding: [batch_size, embedding_dim]
        """
        if not feature_embeddings:
            # No features provided, return zero embedding
            batch_size = 1
            device = next(iter(self.parameters())).device if list(self.parameters()) else torch.device('cpu')
            return torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Stack all feature embeddings
        features_list = []
        for feature_name in sorted(feature_embeddings.keys()):  # Sort for deterministic order
            features_list.append(feature_embeddings[feature_name])
        
        # Concatenate: [batch_size, num_features * embedding_dim]
        concatenated = torch.cat(features_list, dim=1)
        
        # Initialize projection if needed
        # CRITICAL: Only initialize if projection doesn't exist AND flag is False
        # If projection exists (from checkpoint), we should NOT re-initialize it
        if self.projection is None and not self._projection_initialized:
            num_features = len(features_list)
            self._initialize_projection(num_features)
        elif self.projection is not None and not self._projection_initialized:
            # Projection exists but flag is False (loaded from checkpoint)
            # Set flag to True to prevent future re-initialization attempts
            self._projection_initialized = True
        
        # Verify input dimension matches projection's expected input
        if self.projection is not None:
            expected_input_dim = self.projection[0].weight.shape[1]  # Get input dim from first Linear layer
            actual_input_dim = concatenated.shape[1]
            if actual_input_dim != expected_input_dim:
                expected_num_features = expected_input_dim // self.embedding_dim
                actual_num_features = actual_input_dim // self.embedding_dim
                feature_names = sorted(feature_embeddings.keys())
                raise RuntimeError(
                    f"Feature dimension mismatch in SimpleFusionLayer: "
                    f"Expected {expected_input_dim} input dimensions ({expected_num_features} features from checkpoint), "
                    f"but got {actual_input_dim} dimensions ({actual_num_features} features from current input). "
                    f"\nCurrent features being used: {feature_names} "
                    f"\nPlease ensure the same features are provided during inference as during training. "
                    f"You may need to exclude one of these features to match the training configuration."
                )
        
        # Project to target dimension
        fused = self.projection(concatenated)  # [batch_size, embedding_dim]
        
        return fused


class UserEmbeddingGenerator(nn.Module):
    """
    Generates combined user embeddings from multiple user features
    Supports both simple fusion (concat + MLP) and attention-based fusion
    """
    
    def __init__(self,
                 embedding_dim: int,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 use_layer_norm: bool = False,
                 use_simple_fusion: bool = True):
        """
        Args:
            embedding_dim: Dimension of all feature embeddings
            num_attention_layers: Number of attention layers for user feature fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_cls_token: Whether to use CLS token for aggregation
            use_layer_norm: Whether to use layer norm
            use_simple_fusion: If True, use SimpleFusion; if False, use attention-based FeatureFusionLayer
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_simple_fusion = use_simple_fusion
        
        if use_simple_fusion:
            # Simple concatenation + MLP fusion
            self.user_fusion = SimpleFusionLayer(embedding_dim=embedding_dim)
        else:
            # Attention-based fusion with transformers
            self.user_fusion = FeatureFusionLayer(
                embedding_dim=embedding_dim,
                num_attention_layers=num_attention_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_cls_token=use_cls_token,
                use_layer_norm=use_layer_norm
            )
        
    def forward(self,
                user_features: Dict[str, torch.Tensor],
                user_feature_types: Union[Dict[str, str], Dict[str, FeatureType]]) -> torch.Tensor:
        """
        Generate combined user embedding from multiple user features
        
        Args:
            user_features: Dict mapping user feature names to embeddings
                          Each embedding: [batch_size, embedding_dim]
            user_feature_types: Dict mapping feature names to types (str or FeatureType)
        
        Returns:
            combined_user_embedding: [batch_size, embedding_dim]
        """
        # Validate inputs
        if not validate_feature_embeddings(user_features, self.embedding_dim):
            raise ValueError("Invalid user feature embeddings")
        
        # Convert string types to FeatureType enums if needed
        if user_feature_types and isinstance(next(iter(user_feature_types.values())), str):
            user_feature_types = create_feature_type_mapping(user_feature_types)
        
        # Generate fused user embedding
        combined_user_embedding = self.user_fusion(user_features, user_feature_types)
        
        return combined_user_embedding


class ItemEmbeddingGenerator(nn.Module):
    """
    Generates combined item embeddings from multiple item features
    Supports both simple fusion (concat + MLP) and attention-based fusion
    """
    
    def __init__(self,
                 embedding_dim: int,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 use_simple_fusion: bool = True):
        """
        Args:
            embedding_dim: Dimension of all feature embeddings
            num_attention_layers: Number of attention layers for item feature fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_cls_token: Whether to use CLS token for aggregation
            use_simple_fusion: If True, use SimpleFusion; if False, use attention-based FeatureFusionLayer
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_simple_fusion = use_simple_fusion
        
        if use_simple_fusion:
            # Simple concatenation + MLP fusion
            self.item_fusion = SimpleFusionLayer(embedding_dim=embedding_dim)
        else:
            # Attention-based fusion with transformers
            self.item_fusion = FeatureFusionLayer(
                embedding_dim=embedding_dim,
                num_attention_layers=num_attention_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_cls_token=use_cls_token
            )
        
    def forward(self,
                item_features: Dict[str, torch.Tensor],
                item_feature_types: Union[Dict[str, str], Dict[str, FeatureType]]) -> torch.Tensor:
        """
        Generate combined item embedding from multiple item features
        
        Args:
            item_features: Dict mapping item feature names to embeddings
                          Each embedding: [batch_size, embedding_dim]
            item_feature_types: Dict mapping feature names to types (str or FeatureType)
        
        Returns:
            combined_item_embedding: [batch_size, embedding_dim]
        """
        # Validate inputs
        if not validate_feature_embeddings(item_features, self.embedding_dim):
            raise ValueError("Invalid item feature embeddings")
        
        # Convert string types to FeatureType enums if needed
        if item_feature_types and isinstance(next(iter(item_feature_types.values())), str):
            item_feature_types = create_feature_type_mapping(item_feature_types)
        
        # Generate fused item embedding
        combined_item_embedding = self.item_fusion(item_features, item_feature_types)
        
        return combined_item_embedding


class AsymmetricTowerModel(nn.Module):
    """
    Complete model with separate user and item towers that can have different architectures
    """
    
    def __init__(self,
                 embedding_dim: int,
                 # User tower configuration
                 user_num_attention_layers: int = 2,
                 user_num_heads: int = 8,
                 user_dropout: float = 0.1,
                 user_use_cls_token: bool = True,
                 # Item tower configuration
                 item_num_attention_layers: int = 2,
                 item_num_heads: int = 8,
                 item_dropout: float = 0.1,
                 item_use_cls_token: bool = True):
        """
        Args:
            embedding_dim: Dimension of all feature embeddings
            user_*: Configuration parameters for user tower
            item_*: Configuration parameters for item tower
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Create asymmetric towers
        self.user_generator = UserEmbeddingGenerator(
            embedding_dim=embedding_dim,
            num_attention_layers=user_num_attention_layers,
            num_heads=user_num_heads,
            dropout=user_dropout,
            use_cls_token=user_use_cls_token
        )
        
        self.item_generator = ItemEmbeddingGenerator(
            embedding_dim=embedding_dim,
            num_attention_layers=item_num_attention_layers,
            num_heads=item_num_heads,
            dropout=item_dropout,
            use_cls_token=item_use_cls_token
        )
        
    def forward(self,
                user_features: Dict[str, torch.Tensor],
                user_feature_types: Union[Dict[str, str], Dict[str, FeatureType]],
                item_features: Dict[str, torch.Tensor],
                item_feature_types: Union[Dict[str, str], Dict[str, FeatureType]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both user and item embeddings
        
        Args:
            user_features: Dict of user feature embeddings
            user_feature_types: Dict of user feature types
            item_features: Dict of item feature embeddings
            item_feature_types: Dict of item feature types
        
        Returns:
            combined_user_embedding: [batch_size, embedding_dim]
            combined_item_embedding: [batch_size, embedding_dim]
        """
        
        # Generate user embedding
        combined_user_embedding = self.user_generator(user_features, user_feature_types)
        
        # Generate item embedding
        combined_item_embedding = self.item_generator(item_features, item_feature_types)
        
        return combined_user_embedding, combined_item_embedding


# Removed unnecessary factory functions - use direct class instantiation instead


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    embedding_dim = 256
    batch_size = 32
    
    # Example user features (would come from your encoders)
    user_features = {
        'bio_text': torch.randn(batch_size, embedding_dim),
        'age': torch.randn(batch_size, embedding_dim),
        'location': torch.randn(batch_size, embedding_dim),
        'profile_image': torch.randn(batch_size, embedding_dim),
        'interaction_history': torch.randn(batch_size, embedding_dim)
    }
    
    user_feature_types = {
        'bio_text': 'text',
        'age': 'continuous',
        'location': 'categorical',
        'profile_image': 'image',
        'interaction_history': 'temporal'
    }
    
    # Example item features
    item_features = {
        'title': torch.randn(batch_size, embedding_dim),
        'category': torch.randn(batch_size, embedding_dim),
        'price': torch.randn(batch_size, embedding_dim),
        'item_image': torch.randn(batch_size, embedding_dim),
        'description': torch.randn(batch_size, embedding_dim)
    }
    
    item_feature_types = {
        'title': 'text',
        'category': 'categorical',
        'price': 'continuous',
        'item_image': 'image',
        'description': 'text'
    }
    
    # Method 1: Using direct class instantiation (recommended)
    print("=== Testing Direct Class Instantiation ===")
    
    # Create user generator directly with clear parameters
    user_generator = UserEmbeddingGenerator(
        embedding_dim=embedding_dim,
        num_attention_layers=2,
        num_heads=8,  # Must divide embedding_dim
        dropout=0.1
    )
    
    # Create item generator directly  
    item_generator = ItemEmbeddingGenerator(
        embedding_dim=embedding_dim,
        num_attention_layers=1,
        num_heads=8,
        dropout=0.1
    )
    
    combined_user = user_generator(user_features, user_feature_types)
    combined_item = item_generator(item_features, item_feature_types)
    
    print(f"Combined user embedding shape: {combined_user.shape}")  # [32, 256]
    print(f"Combined item embedding shape: {combined_item.shape}")  # [32, 256]
    
    # Method 2: Using asymmetric tower model directly
    print("\n=== Testing Asymmetric Tower Model ===")
    
    tower_model = AsymmetricTowerModel(
        embedding_dim=embedding_dim,
        user_num_attention_layers=4,
        user_num_heads=16,
        user_dropout=0.15,
        item_num_attention_layers=1,
        item_num_heads=8,
        item_dropout=0.1
    )
    
    combined_user_v2, combined_item_v2 = tower_model(
        user_features, user_feature_types,
        item_features, item_feature_types
    )
    
    print(f"Tower model user embedding shape: {combined_user_v2.shape}")  # [32, 256]
    print(f"Tower model item embedding shape: {combined_item_v2.shape}")  # [32, 256]
    
    print("✅ Feature fusion completed successfully!")