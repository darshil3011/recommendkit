"""
Factory function for creating categorical encoders.
"""

from typing import Dict, Any
from .hash_encoder import HashCategoricalEncoder


def create_categorical_encoder(config: Dict[str, Any]) -> Any:
    """
    Factory function to create a categorical encoder based on configuration.
    
    Args:
        config: Configuration dictionary with:
            - aggregation_strategy: 'separate_concat' or 'joint_embedding' (default: 'separate_concat')
            - hash_vocab_size: Size of hash space (default: 10000)
            - embedding_dim: Output embedding dimension (default: 64)
            - mlp_hidden_dims: Hidden layer dimensions for MLP (default: [64])
            - dropout: Dropout probability (default: 0.1)
            - activation: Activation function (default: 'relu')
            - hash_seed: Seed for reproducible hashing (default: 42)
            - num_categorical_fields: Number of categorical fields (default: 3)
    
    Returns:
        Configured categorical encoder instance
    """
    return HashCategoricalEncoder(
        aggregation_strategy=config.get('aggregation_strategy', 'separate_concat'),
        hash_vocab_size=config.get('hash_vocab_size', 10000),
        embedding_dim=config.get('embedding_dim', 64),
        mlp_hidden_dims=config.get('mlp_hidden_dims', [64]),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        hash_seed=config.get('hash_seed', 42),
        num_categorical_fields=config.get('num_categorical_fields', 3)
    )



