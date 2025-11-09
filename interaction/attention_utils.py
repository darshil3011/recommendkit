"""
attention_utils.py
Helper functions and reusable components for attention-based fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import math


class FeatureType(Enum):
    """Types of features we can encode"""
    IMAGE = "image"
    TEXT = "text" 
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    TEMPORAL = "temporal"


class InteractionType(Enum):
    """Types of embeddings in the user-item interaction sequence"""
    USER = "user"
    ITEM = "item"


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer for feature fusion"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, embedding_dim]
            key: [batch_size, seq_len, embedding_dim] 
            value: [batch_size, seq_len, embedding_dim]
            attention_mask: [batch_size, seq_len] - 1 for valid positions, 0 for masked
        
        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch_size, num_heads, seq_len, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(-1, self.num_heads, seq_len, -1)  # [batch_size, num_heads, seq_len, seq_len]
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """A transformer block with self-attention and feed-forward layers"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, 
                 ff_dim: Optional[int] = None, dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * embedding_dim
            
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(embedding_dim)
            self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, attention_mask)
        if self.use_layer_norm:
            x = self.norm1(x + attn_output)
        else:
            x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        if self.use_layer_norm:
            x = self.norm2(x + ff_output)
        else:
            x = x + ff_output
        
        return x


class FeatureFusionLayer(nn.Module):
    """
    Generic feature fusion layer using attention mechanism with CLS token
    Can be reused for user features, item features, or any multi-modal fusion
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 use_layer_norm: bool = False,
                 feature_types: Optional[List[FeatureType]] = None):
        """
        Args:
            embedding_dim: Dimension of feature embeddings (all features should have same dim)
            num_attention_layers: Number of transformer layers to apply
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_cls_token: Whether to use CLS token for global representation
            feature_types: List of feature types to support (if None, supports all)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_attention_layers = num_attention_layers
        self.use_cls_token = use_cls_token
        
        # CLS token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            
        # Positional encoding for feature types
        if feature_types is None:
            feature_types = list(FeatureType)
        self.supported_feature_types = feature_types
        self.feature_type_embeddings = nn.Embedding(len(FeatureType), embedding_dim)
        
        # Stack of transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout=dropout, use_layer_norm=use_layer_norm)
            for _ in range(num_attention_layers)
        ])
        
        # Final projection layer
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
            feature_embeddings: Dict[str, torch.Tensor],
            feature_types: Dict[str, FeatureType]) -> torch.Tensor:
    
        if not feature_embeddings:
            batch_size = 1
            device = next(iter(self.parameters())).device
            return torch.zeros(batch_size, self.embedding_dim, device=device)
            
        batch_size = next(iter(feature_embeddings.values())).size(0)
        device = next(iter(feature_embeddings.values())).device
        
        # Stack all feature embeddings
        features = []
        type_indices = []
        
        for feature_name, embedding in feature_embeddings.items():
            if feature_name in feature_types:
                features.append(embedding.unsqueeze(1))
                feature_type = feature_types[feature_name]
                type_indices.append(list(FeatureType).index(feature_type))
        
        if not features:
            return torch.zeros(batch_size, self.embedding_dim, device=device)
            
        # Stack features: [batch_size, num_features, embedding_dim]
        stacked_features = torch.cat(features, dim=1)
        
        # Add positional encodings for feature types
        type_indices_tensor = torch.tensor(type_indices, device=device)
        type_embeddings = self.feature_type_embeddings(type_indices_tensor)
        type_embeddings = type_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add type embeddings to features
        stacked_features = stacked_features + type_embeddings
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls_tokens, stacked_features], dim=1)
        else:
            sequence = stacked_features
        
        # Create attention mask
        seq_len = sequence.size(1)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Apply transformer layers
        hidden_states = sequence
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Extract final representation
        if self.use_cls_token:
            final_embedding = hidden_states[:, 0, :]
        else:
            final_embedding = hidden_states.mean(dim=1)
        
        # Apply final projection
        final_embedding = self.final_projection(final_embedding)
        
        return final_embedding

def create_feature_type_mapping(feature_dict: Dict[str, str]) -> Dict[str, FeatureType]:
    """
    Helper function to convert string feature types to FeatureType enum
    
    Args:
        feature_dict: Dict mapping feature names to string types
        
    Returns:
        Dict mapping feature names to FeatureType enums
    """
    type_mapping = {}
    for feature_name, type_str in feature_dict.items():
        if type_str.lower() in [ft.value for ft in FeatureType]:
            type_mapping[feature_name] = FeatureType(type_str.lower())
        else:
            # Default to categorical if unknown type
            type_mapping[feature_name] = FeatureType.CATEGORICAL
    
    return type_mapping


def validate_feature_embeddings(feature_embeddings: Dict[str, torch.Tensor], 
                               expected_dim: int) -> bool:
    """
    Validate that all feature embeddings have the correct dimensions
    
    Args:
        feature_embeddings: Dict of feature embeddings
        expected_dim: Expected embedding dimension
        
    Returns:
        True if all embeddings are valid, False otherwise
    """
    if not feature_embeddings:
        return False
        
    for name, embedding in feature_embeddings.items():
        if len(embedding.shape) != 2:
            print(f"Error: Feature '{name}' should have 2 dimensions, got {len(embedding.shape)}")
            return False
        if embedding.shape[1] != expected_dim:
            print(f"Error: Feature '{name}' should have dimension {expected_dim}, got {embedding.shape[1]}")
            return False
            
    return True


# Utility functions for tensor operations
def safe_mean_pooling(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Safe mean pooling that handles empty tensors"""
    if tensor.size(dim) == 0:
        return torch.zeros_like(tensor).mean(dim=dim)
    return tensor.mean(dim=dim)


def create_attention_mask(batch_size: int, seq_len: int, 
                         mask_positions: Optional[List[int]] = None,
                         device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Create attention mask for transformer layers
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_positions: List of positions to mask (0-indexed), None means no masking
        device: Device to create tensor on
        
    Returns:
        Attention mask: [batch_size, seq_len] with 1 for valid, 0 for masked
    """
    mask = torch.ones(batch_size, seq_len, device=device)
    
    if mask_positions is not None:
        for pos in mask_positions:
            if 0 <= pos < seq_len:
                mask[:, pos] = 0
                
    return mask

class SimpleFusionLayer(nn.Module):
    """
    Simple fusion layer using concatenation + MLP instead of attention
    More stable for small number of features
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 max_features: int = 10,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2
        
        # Simple MLP for fusion
        # Input will be variable size (num_features * embedding_dim)
        # We'll handle this dynamically
        self.fusion_mlp = None
        self._initialized = False
        self.hidden_dim = hidden_dim
        
    def _initialize_mlp(self, input_dim: int):
        """Initialize MLP with known input dimension"""
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        self._initialized = True
    
    def forward(self, 
                feature_embeddings: Dict[str, torch.Tensor],
                feature_types: Dict[str, FeatureType]) -> torch.Tensor:
        
        if not feature_embeddings:
            batch_size = 1
            device = torch.device('cpu')
            return torch.zeros(batch_size, self.embedding_dim, device=device)
            
        batch_size = next(iter(feature_embeddings.values())).size(0)
        device = next(iter(feature_embeddings.values())).device
        
        # Simply concatenate all features
        features = []
        for feature_name in sorted(feature_embeddings.keys()):
            if feature_name in feature_types:
                features.append(feature_embeddings[feature_name])
        
        if not features:
            return torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Concatenate: [batch_size, num_features * embedding_dim]
        concatenated = torch.cat(features, dim=1)
        
        # Initialize MLP if needed
        if not self._initialized:
            self._initialize_mlp(concatenated.size(1))
            self.fusion_mlp = self.fusion_mlp.to(device)
        
        # Pass through MLP
        fused = self.fusion_mlp(concatenated)
        
        return fused