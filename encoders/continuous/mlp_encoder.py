"""
MLP-based continuous encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .base_continuous_encoder import BaseContinuousEncoder


class MLPContinuousEncoder(BaseContinuousEncoder):
    """
    MLP-based continuous encoder that encodes numerical features.
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 normalize: bool = True):
        """
        Initialize MLP continuous encoder.
        
        Args:
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            normalize: Whether to normalize inputs
        """
        super().__init__(embedding_dim, normalize)
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        
        # MLP will be initialized dynamically based on input features
        self.mlp = None
        self._mlp_initialized = False
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _initialize_mlp(self, input_dim: int):
        """Initialize MLP with known input dimension"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(self.activation),
                nn.Dropout(self.dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.embedding_dim))
        self.mlp = nn.Sequential(*layers)
        self._mlp_initialized = True
    
    def forward(self, continuous_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for continuous feature encoding.
        
        Args:
            continuous_dict: Dictionary mapping feature names to values
                            Single: {'age': 25.5, 'income': 50000}
                            Batched: {'age': [25.5, 30.0], 'income': [50000, 60000]}
        
        Returns:
            Dictionary with continuous features: {"continuous_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        if not continuous_dict:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return {"continuous_features": torch.zeros(1, self.embedding_dim, device=device)}
        
        # Check if batched
        first_value = next(iter(continuous_dict.values()))
        is_batched = isinstance(first_value, list)
        
        # Get sorted field names for consistent ordering
        field_names = sorted(continuous_dict.keys())
        num_features = len(field_names)
        
        # Initialize MLP if needed
        if not self._mlp_initialized:
            self._initialize_mlp(num_features)
        
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if is_batched:
            # Build batch tensor
            batch_size = len(first_value) if first_value else 1
            batch_features = []
            for i in range(batch_size):
                sample_values = []
                for field_name in field_names:
                    field_values = continuous_dict.get(field_name, [])
                    if isinstance(field_values, list) and i < len(field_values):
                        value = field_values[i]
                    else:
                        value = None
                    sample_values.append(float(value) if value is not None else 0.0)
                batch_features.append(sample_values)
            
            feature_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
        else:
            # Single sample
            values = []
            for field_name in field_names:
                value = continuous_dict.get(field_name)
                values.append(float(value) if value is not None else 0.0)
            feature_tensor = torch.tensor(values, dtype=torch.float32, device=device)
            feature_tensor = feature_tensor.unsqueeze(0)  # Add batch dimension
        
        # Normalize
        feature_tensor = self._normalize_features(feature_tensor)
        
        # Pass through MLP
        embeddings = self.mlp(feature_tensor)
        
        return {"continuous_features": embeddings}

