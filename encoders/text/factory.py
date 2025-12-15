"""
Factory function to create text encoders based on configuration.
"""

from typing import Dict, Any
from .transformer_encoder import TransformerTextEncoder, create_transformer_encoder
from .word2vec_encoder import Word2VecTextEncoder, create_word2vec_encoder


def _is_word2vec_model(model_name: str) -> bool:
    """Check if the model name indicates a Word2Vec model"""
    word2vec_indicators = [
        'word2vec', 'fasttext', 'glove', 'google-news', 'wikipedia'
    ]
    return any(indicator in model_name.lower() for indicator in word2vec_indicators)


def create_text_encoder(config: Dict[str, Any]) -> Any:
    """
    Factory function to create a text encoder based on configuration.
    
    Automatically detects whether to use transformer or Word2Vec encoder
    based on the model_name.
    
    Args:
        config: Configuration dictionary with:
            - model_name: Model name (required)
            - aggregation_strategy: 'separate_concat', 'joint_encoding', or 'mean' (default: 'separate_concat')
            - embedding_dim: Output embedding dimension (default: 256)
            - num_text_fields: Number of text fields (default: 2)
            - max_length: Max sequence length for transformers (default: 512)
            - freeze_bert: Freeze transformer params (default: False)
            - pooling_strategy: Pooling for transformers (default: 'cls')
    
    Returns:
        Text encoder instance (TransformerTextEncoder or Word2VecTextEncoder)
        
    Examples:
        # Transformer encoder
        encoder = create_text_encoder({
            'model_name': 'bert-base-uncased',
            'embedding_dim': 128
        })
        
        # Word2Vec encoder
        encoder = create_text_encoder({
            'model_name': 'word2vec-google-news-300',
            'embedding_dim': 128
        })
    """
    model_name = config.get('model_name', 'bert-base-uncased')
    
    # Determine encoder type
    if _is_word2vec_model(model_name):
        return create_word2vec_encoder(
            model_name=model_name,
            aggregation_strategy=config.get('aggregation_strategy', 'separate_concat'),
            embedding_dim=config.get('embedding_dim', 256),
            num_text_fields=config.get('num_text_fields', 2)
        )
    else:
        return create_transformer_encoder(
            model_name=model_name,
            aggregation_strategy=config.get('aggregation_strategy', 'separate_concat'),
            max_length=config.get('max_length', 512),
            embedding_dim=config.get('embedding_dim', 256),
            freeze_bert=config.get('freeze_bert', False),
            pooling_strategy=config.get('pooling_strategy', 'cls'),
            num_text_fields=config.get('num_text_fields', 2)
        )

