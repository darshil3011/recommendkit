"""
Factory function for creating continuous encoders.
"""

from typing import Dict, Any, List, Optional
from .mlp_encoder import MLPContinuousEncoder


def create_continuous_encoder(config: Dict[str, Any]) -> Any:
    """
    Factory function to create a continuous encoder based on configuration.
    
    Args:
        config: Configuration dictionary with:
            - embedding_dim: Output embedding dimension (default: 64)
            - hidden_dims: Hidden layer dimensions for MLP (default: [128, 64])
            - hidden_dim: Alternative way to specify hidden dimensions (backwards compatibility)
            - dropout: Dropout probability (default: 0.1)
            - activation: Activation function (default: 'relu')
            - normalize: Whether to normalize inputs (default: True)
    
    Returns:
        Configured continuous encoder instance
    """
    # Handle backwards compatibility
    hidden_dims = config.get('hidden_dims')
    if hidden_dims is None and 'hidden_dim' in config:
        hidden_dims = [config['hidden_dim']]
    elif hidden_dims is None:
        hidden_dims = [128, 64]
    
    return MLPContinuousEncoder(
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dims=hidden_dims,
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        normalize=config.get('normalize', True)
    )

