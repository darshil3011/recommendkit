"""
Factory function for creating temporal encoders.
"""

from typing import Dict, Any, List, Set
import torch.nn as nn

from .lstm_temporal_encoder import LSTMTemporalEncoder
from .base_temporal_encoder import ItemLookupInterface, ModalityType, TemporalAggregationStrategy


def create_temporal_encoder(item_lookup: ItemLookupInterface,
                           modality_encoders: Dict[str, nn.Module],
                           enabled_modalities: List[str] = None,
                           aggregation_strategy: str = "last_hidden",
                           lstm_hidden_dim: int = 128,
                           lstm_num_layers: int = 2,
                           lstm_dropout: float = 0.1,
                           bidirectional: bool = False,
                           output_dim: int = 256,
                           max_sequence_length: int = 50,
                           missing_item_strategy: str = 'zero') -> LSTMTemporalEncoder:
    """
    Factory function to create a temporal encoder based on configuration.
    
    Args:
        item_lookup: Interface for looking up item features
        modality_encoders: Dict mapping modality names to encoder modules
        enabled_modalities: List of modalities to use ('image', 'text', 'categorical', 'continuous')
        aggregation_strategy: 'last_hidden', 'mean_pooling', 'max_pooling', 'attention'
        lstm_hidden_dim: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout
        bidirectional: Whether to use bidirectional LSTM
        output_dim: Final output dimension
        max_sequence_length: Maximum sequence length
        missing_item_strategy: How to handle missing items ('zero', 'skip', 'previous')
        
    Returns:
        Configured LSTMTemporalEncoder instance
    """
    # Convert string keys to ModalityType enums
    enum_encoders = {}
    for mod_name, encoder in modality_encoders.items():
        if mod_name == 'image':
            enum_encoders[ModalityType.IMAGE] = encoder
        elif mod_name == 'text':
            enum_encoders[ModalityType.TEXT] = encoder
        elif mod_name == 'categorical':
            enum_encoders[ModalityType.CATEGORICAL] = encoder
        elif mod_name == 'continuous':
            enum_encoders[ModalityType.CONTINUOUS] = encoder
    
    # Convert enabled_modalities to enum set
    enabled_enum_modalities = None
    if enabled_modalities:
        enabled_enum_modalities = set()
        for mod_name in enabled_modalities:
            if mod_name == 'image':
                enabled_enum_modalities.add(ModalityType.IMAGE)
            elif mod_name == 'text':
                enabled_enum_modalities.add(ModalityType.TEXT)
            elif mod_name == 'categorical':
                enabled_enum_modalities.add(ModalityType.CATEGORICAL)
            elif mod_name == 'continuous':
                enabled_enum_modalities.add(ModalityType.CONTINUOUS)
    
    return LSTMTemporalEncoder(
        item_lookup=item_lookup,
        modality_encoders=enum_encoders,
        enabled_modalities=enabled_enum_modalities,
        aggregation_strategy=aggregation_strategy,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        bidirectional=bidirectional,
        embedding_dim=output_dim,
        max_sequence_length=max_sequence_length,
        missing_item_strategy=missing_item_strategy
    )



