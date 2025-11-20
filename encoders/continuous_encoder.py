"""
continuous_encoder.py
Encoder for continuous numerical features with normalization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import warnings


class ContinuousEncoder(nn.Module):
    """
    Encodes continuous numerical features with proper normalization
    
    Args:
        embedding_dim: Output embedding dimension
        hidden_dims: Hidden layer dimensions for MLP
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'tanh')
        normalize: Whether to apply layer normalization to input
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 normalize: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Track feature statistics for normalization
        self.feature_means = {}
        self.feature_stds = {}
        self._stats_initialized = False
        
        # Build MLP layers
        # Note: Input dimension will be determined dynamically based on number of features
        self.mlp = None
        self._mlp_initialized = False
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        
        # Layer normalization for input (optional but recommended)
        if normalize:
            self.input_norm = None  # Will be initialized dynamically
    
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
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Use LayerNorm only for hidden layers, not input
            # This prevents collapse when input_dim is small
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(self.activation))
            layers.append(nn.Dropout(self.dropout))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, self.embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # DON'T use LayerNorm on input - it destroys variation for low-dim inputs
        # Instead, use simple standardization
        self.input_norm = None
        
        self._mlp_initialized = True
    
    def _normalize_features(self, 
                           continuous_dict: Dict[str, float],
                           update_stats: bool = False) -> torch.Tensor:
        """
        Normalize continuous features using running statistics or z-score
        
        Args:
            continuous_dict: Dictionary of feature_name -> value
            update_stats: Whether to update running statistics (training mode)
            
        Returns:
            Normalized feature tensor
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Get sorted field names for consistent ordering
        field_names = sorted(continuous_dict.keys())
        
        # Extract values - handle None properly
        values = []
        for field_name in field_names:
            value = continuous_dict.get(field_name)
            if value is None:
                values.append(0.0)  # Handle missing values
            else:
                values.append(float(value))
        
        # Convert to tensor
        feature_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Apply normalization
        if self.normalize:
            # Clamp extreme values to prevent explosion
            feature_tensor = torch.clamp(feature_tensor, min=-1e6, max=1e6)
            
            # Apply log scaling for very large values
            abs_features = torch.abs(feature_tensor)
            large_mask = abs_features > 1000
            if large_mask.any():
                # Apply log transform to large values
                sign = torch.sign(feature_tensor[large_mask])
                feature_tensor[large_mask] = sign * torch.log1p(abs_features[large_mask])
        
        return feature_tensor
    
    def forward(self, continuous_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for continuous feature encoding
        
        Args:
            continuous_dict: Dictionary mapping feature names to values
                            Single: {'age': 25.5, 'income': 50000}
                            Batched: {'age': [25.5, 30.0], 'income': [50000, 60000]}
        
        Returns:
            Dictionary with continuous features: {"continuous_features": torch.Tensor}
        """
        if not continuous_dict:
            # Empty input, return default embedding
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return {"continuous_features": torch.zeros(1, self.embedding_dim, device=device)}
        
        # Determine if this is batched input
        first_value = next(iter(continuous_dict.values()))
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            return self._forward_batched(continuous_dict)
        else:
            # Single sample
            result = self._forward_single(continuous_dict)
            return {"continuous_features": result}
    
    def _forward_single(self, continuous_dict: Dict[str, Any]) -> torch.Tensor:
        """Process a single sample"""
        # Get sorted field names for consistent ordering
        field_names = sorted(continuous_dict.keys())
        num_features = len(field_names)
        
        # Initialize MLP if needed (BEFORE normalizing features)
        if not self._mlp_initialized:
            self._initialize_mlp(num_features)
        
        # Normalize features (this must return tensor of size num_features)
        normalized_features = self._normalize_features(continuous_dict, update_stats=self.training)
        
        # Verify dimensions match
        assert normalized_features.size(0) == num_features, \
            f"Feature dimension mismatch: expected {num_features}, got {normalized_features.size(0)}"
        
        # Reshape for batch processing
        feature_input = normalized_features.unsqueeze(0)  # [1, num_features]
        
        # Simple standardization instead of LayerNorm
        # LayerNorm destroys variation when num_features is small
        if self.normalize and num_features > 1:
            mean = feature_input.mean(dim=1, keepdim=True)
            std = feature_input.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            feature_input = (feature_input - mean) / std
        elif self.normalize and num_features == 1:
            # For single feature, just center it (subtract mean)
            # Can't compute std with only 1 feature
            mean = feature_input.mean(dim=1, keepdim=True)
            feature_input = feature_input - mean
        
        # Pass through MLP
        embedding = self.mlp(feature_input)  # [1, embedding_dim]
        
        return embedding
    
    def _forward_batched(self, continuous_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a batch of samples efficiently"""
        # Get batch size from first non-empty field
        batch_size = 0
        for field_values in continuous_dict.values():
            if isinstance(field_values, list) and len(field_values) > 0:
                batch_size = len(field_values)
                break
        
        if batch_size == 0:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return {"continuous_features": torch.zeros(1, self.embedding_dim, device=device)}
        
        # Get sorted field names for consistency
        field_names = sorted(continuous_dict.keys())
        num_features = len(field_names)
        
        # Initialize MLP if needed
        if not self._mlp_initialized:
            self._initialize_mlp(num_features)
        
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Build batch tensor directly - this is more efficient and correct
        batch_features = []
        for i in range(batch_size):
            sample_values = []
            for field_name in field_names:
                if field_name in continuous_dict:
                    field_values = continuous_dict[field_name]
                    if isinstance(field_values, list) and i < len(field_values):
                        value = field_values[i]
                    else:
                        value = None
                else:
                    value = None
                
                # Handle None
                if value is None:
                    sample_values.append(0.0)
                else:
                    sample_values.append(float(value))
            
            batch_features.append(sample_values)
        
        # Convert to tensor: [batch_size, num_features]
        feature_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
        
        # Apply normalization (clamp and log transform for large values)
        if self.normalize:
            feature_tensor = torch.clamp(feature_tensor, min=-1e6, max=1e6)
            
            abs_features = torch.abs(feature_tensor)
            large_mask = abs_features > 1000
            if large_mask.any():
                sign = torch.sign(feature_tensor[large_mask])
                feature_tensor[large_mask] = sign * torch.log1p(abs_features[large_mask])
        
        # Simple standardization: scale to reasonable range
        # Don't use LayerNorm here - it destroys variation for low-dim inputs
        if self.normalize:
            # Standardize each feature independently
            # For batch_size=1, just center (subtract mean) without dividing by std
            # For batch_size>1, use full standardization
            mean = feature_tensor.mean(dim=0, keepdim=True)
            feature_tensor = feature_tensor - mean
            
            if batch_size > 1:
                # Only compute std if we have more than 1 sample
                std = feature_tensor.std(dim=0, keepdim=True, unbiased=False) + 1e-8
                feature_tensor = feature_tensor / std
            # For batch_size=1, std would be 0, so we skip division (already centered)
        
        # Pass through MLP
        embeddings = self.mlp(feature_tensor)  # [batch_size, embedding_dim]
        
        return {"continuous_features": embeddings}
    
    def get_output_dim(self) -> int:
        """Get the output embedding dimension"""
        return self.embedding_dim


def create_continuous_encoder(embedding_dim: int = 64,
                             hidden_dims: Optional[List[int]] = None,
                             hidden_dim: Optional[int] = None,  # Support both for compatibility
                             dropout: float = 0.1,
                             activation: str = 'relu',
                             normalize: bool = True) -> ContinuousEncoder:
    """
    Factory function to create a ContinuousEncoder
    
    Args:
        embedding_dim: Output embedding dimension
        hidden_dims: Hidden layer dimensions for MLP (takes priority)
        hidden_dim: Alternative way to specify hidden dimensions (backwards compatibility)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'tanh')
        normalize: Whether to normalize inputs
        
    Returns:
        Configured ContinuousEncoder instance
    """
    # Handle backwards compatibility
    if hidden_dims is None and hidden_dim is not None:
        hidden_dims = [hidden_dim]
    elif hidden_dims is None:
        hidden_dims = [128, 64]  # Default
    
    return ContinuousEncoder(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        normalize=normalize
    )


# Example usage and testing
# Utility function for feature scaling insights
def analyze_continuous_features(continuous_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze continuous features to understand their scale and distribution
    
    Args:
        continuous_dict: Dictionary of continuous features
        
    Returns:
        Analysis results
    """
    analysis = {}
    
    for feature_name, value in continuous_dict.items():
        if value is None:
            analysis[feature_name] = {'status': 'missing'}
            continue
        
        if isinstance(value, list):
            values = [v for v in value if v is not None]
            if values:
                analysis[feature_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'range': max(values) - min(values)
                }
        else:
            analysis[feature_name] = {
                'value': value,
                'magnitude': abs(value),
                'needs_scaling': abs(value) > 100  # Flag for large values
            }
    
    return analysis


if __name__ == "__main__":
    print("Testing Continuous Encoder...")
    
    # Test with realistic recommendation system data
    test_cases = [
        {'age': 25.5, 'income': 50000.0, 'rating': 4.5},
        {'age': 30.0, 'income': 75000.0, 'rating': 3.8},
        {'age': 45.0, 'income': 120000.0, 'rating': 4.9},
    ]
    
    encoder = create_continuous_encoder(embedding_dim=64, normalize=True)
    
    print("\n--- Single Sample Tests ---")
    for i, test_input in enumerate(test_cases):
        print(f"\nInput {i+1}: {test_input}")
        
        # Analyze features
        analysis = analyze_continuous_features(test_input)
        print(f"Analysis: {analysis}")
        
        with torch.no_grad():
            output = encoder(test_input)
            features = output['continuous_features']
            print(f"Output - Mean: {features.mean().item():.4f}, "
                  f"Std: {features.std().item():.4f}, "
                  f"Range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    
    print("\n--- Batch Test ---")
    batched_input = {
        'age': [25.5, 30.0, 45.0],
        'income': [50000.0, 75000.0, 120000.0],
        'rating': [4.5, 3.8, 4.9]
    }
    
    with torch.no_grad():
        output = encoder(batched_input)
        features = output['continuous_features']
        print(f"Batch shape: {features.shape}")
        print(f"Batch stats - Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
    
    print("\n✅ Continuous encoder tests completed!")