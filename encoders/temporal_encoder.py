import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from enum import Enum
import warnings


class TemporalAggregationStrategy(Enum):
    """Supported aggregation strategies for LSTM outputs"""
    LAST_HIDDEN = "last_hidden"        # Use last hidden state
    MEAN_POOLING = "mean_pooling"      # Average all hidden states
    MAX_POOLING = "max_pooling"        # Max pool all hidden states
    ATTENTION = "attention"            # Attention over hidden states


class ModalityType(Enum):
    """Types of modalities to encode from items"""
    IMAGE = "image"
    TEXT = "text"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class ItemLookupInterface:
    """
    Abstract interface for looking up item features
    Users must implement this to provide item data
    """
    
    def get_item_features(self, item_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get features for a specific item
        
        Args:
            item_id: ID of the item to lookup
            
        Returns:
            Dictionary containing item features:
            {
                'image': {'main_image': '/path/to/image.jpg', ...},
                'text': {'title': 'Product Title', 'description': '...'},
                'categorical': {'category': 'electronics', 'brand': 'Apple'},
                'continuous': {'price': 99.99, 'rating': 4.5}
            }
        """
        raise NotImplementedError("Users must implement this method")
    
    def batch_get_item_features(self, item_ids: List[Union[str, int]]) -> List[Dict[str, Any]]:
        """
        Get features for multiple items (can be optimized for batch retrieval)
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            List of feature dictionaries in the same order as item_ids
        """
        return [self.get_item_features(item_id) for item_id in item_ids]


class AttentionPooling(nn.Module):
    """Attention mechanism for aggregating LSTM hidden states"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
            
        Returns:
            Attended output: (batch_size, hidden_dim)
        """
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights
        attended_output = torch.sum(
            attention_weights.unsqueeze(-1) * hidden_states, dim=1
        )  # (batch_size, hidden_dim)
        
        return attended_output


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for sequential item interactions
    
    Args:
        item_lookup: Interface for looking up item features
        modality_encoders: Dictionary mapping modality types to their encoders
        enabled_modalities: Set of modalities to use (subset of available encoders)
        aggregation_strategy: How to aggregate LSTM outputs
        lstm_hidden_dim: Hidden dimension of LSTM
        lstm_num_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout (applied if num_layers > 1)
        bidirectional: Whether to use bidirectional LSTM
        output_dim: Final output dimension
        max_sequence_length: Maximum sequence length (for padding/truncation)
        missing_item_strategy: How to handle missing items ('zero', 'skip', 'previous')
    """
    
    def __init__(self,
                 item_lookup: ItemLookupInterface,
                 modality_encoders: Dict[ModalityType, nn.Module],
                 enabled_modalities: Set[ModalityType] = None,
                 aggregation_strategy: Union[str, TemporalAggregationStrategy] = TemporalAggregationStrategy.LAST_HIDDEN,
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 bidirectional: bool = False,
                 output_dim: int = 256,
                 max_sequence_length: int = 50,
                 missing_item_strategy: str = 'zero'):
        
        super().__init__()
        
        # Convert string arguments to enums
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = TemporalAggregationStrategy(aggregation_strategy.lower())
        
        self.item_lookup = item_lookup
        self.modality_encoders = nn.ModuleDict()
        self.aggregation_strategy = aggregation_strategy
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        self.missing_item_strategy = missing_item_strategy
        
        # Validate and set enabled modalities
        available_modalities = set(modality_encoders.keys())
        if enabled_modalities is None:
            enabled_modalities = available_modalities
        else:
            # Check that enabled modalities are available
            invalid_modalities = enabled_modalities - available_modalities
            if invalid_modalities:
                raise ValueError(f"Enabled modalities {invalid_modalities} not available in encoders")
        
        self.enabled_modalities = enabled_modalities
        
        # Store only enabled modality encoders
        for modality_type, encoder in modality_encoders.items():
            if modality_type in enabled_modalities:
                self.modality_encoders[modality_type.value] = encoder
        
        # Calculate input dimension to LSTM
        self.item_embedding_dim = self._calculate_item_embedding_dim()
        
        # LSTM layer
        lstm_input_dim = self.item_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        # Aggregation layer
        if aggregation_strategy == TemporalAggregationStrategy.ATTENTION:
            self.attention_pooling = AttentionPooling(lstm_output_dim)
        
        # Final projection layer
        self.projection = nn.Linear(lstm_output_dim, output_dim)
        
        # For handling missing items
        self.register_buffer('default_item_embedding', torch.zeros(self.item_embedding_dim))
        
        # For padding sequences
        self.register_buffer('padding_embedding', torch.zeros(self.item_embedding_dim))
    
    def _calculate_item_embedding_dim(self) -> int:
        """Calculate the dimension of item embeddings based on enabled modalities"""
        total_dim = 0
        
        # We need to make assumptions about encoder output dimensions
        # In practice, users should ensure consistent dimensions or provide this info
        for modality_type in self.enabled_modalities:
            modality_key = modality_type.value
            
            # Skip if encoder is not available
            if modality_key not in self.modality_encoders:
                continue
                
            encoder = self.modality_encoders[modality_key]
            
            if modality_type == ModalityType.IMAGE:
                # Assume image encoder outputs a fixed dimension
                total_dim += getattr(encoder, 'embedding_dim', 256)
            elif modality_type == ModalityType.TEXT:
                # Assume text encoder outputs a fixed dimension
                total_dim += getattr(encoder, 'embedding_dim', 256)
            elif modality_type == ModalityType.CATEGORICAL:
                # For categorical encoders, always use embedding_dim as they project to fixed size
                total_dim += getattr(encoder, 'embedding_dim', 128)
            elif modality_type == ModalityType.CONTINUOUS:
                # Assume continuous encoder outputs a fixed dimension
                total_dim += getattr(encoder, 'output_dim', 64)
        
        return total_dim if total_dim > 0 else 128  # Fallback
    
    def _encode_single_item(self, item_features: Dict[str, Any]) -> torch.Tensor:
        """
        Encode a single item's features into an embedding
        
        Args:
            item_features: Dictionary containing item features
            
        Returns:
            Item embedding tensor of shape (item_embedding_dim,)
        """
        modality_embeddings = []
        device = next(self.parameters()).device
        
        for modality_type in self.enabled_modalities:
            modality_key = modality_type.value
            
            # Skip if encoder is not available
            if modality_key not in self.modality_encoders:
                continue
            
            if modality_key not in item_features or item_features[modality_key] is None:
                # Handle missing modality data
                encoder = self.modality_encoders[modality_key]
                if modality_type == ModalityType.IMAGE:
                    dim = getattr(encoder, 'embedding_dim', 256)
                elif modality_type == ModalityType.TEXT:
                    dim = getattr(encoder, 'embedding_dim', 256)
                elif modality_type == ModalityType.CATEGORICAL:
                    dim = getattr(encoder, 'embedding_dim', 128)
                else:  # CONTINUOUS
                    dim = getattr(encoder, 'output_dim', 64)
                
                # Use zero embedding for missing modality
                modality_embeddings.append(torch.zeros(dim, device=device))
                continue
            
            try:
                # Get modality data
                modality_data = item_features[modality_key]
                encoder = self.modality_encoders[modality_key]
                
                # Encode based on modality type
                if modality_type == ModalityType.IMAGE:
                    # For image encoder that expects paths dictionary
                    if hasattr(encoder, 'forward_from_paths'):
                        result = encoder.forward_from_paths(modality_data)
                        # Extract tensor from dict if needed
                        if isinstance(result, dict):
                            embedding = result["image_features"].squeeze(0)
                        else:
                            embedding = result.squeeze(0)
                    else:
                        result = encoder(modality_data)
                        # Extract tensor from dict if needed
                        if isinstance(result, dict):
                            embedding = result["image_features"].squeeze(0)
                        else:
                            embedding = result.squeeze(0)
                
                elif modality_type == ModalityType.TEXT:
                    # For text encoder that expects text dictionary
                    result = encoder(modality_data)
                    # Extract tensor from dict
                    if isinstance(result, dict):
                        embedding = result["text_features"].squeeze(0)
                    else:
                        embedding = result.squeeze(0)
                
                elif modality_type == ModalityType.CATEGORICAL:
                    # For categorical encoder
                    result = encoder(modality_data)
                    # Extract tensor from dict
                    if isinstance(result, dict):
                        embedding = result["categorical_features"].squeeze(0)
                    else:
                        embedding = result.squeeze(0)
                
                elif modality_type == ModalityType.CONTINUOUS:
                    # For continuous encoder (assume it exists)
                    result = encoder(modality_data)
                    # Extract tensor from dict if needed
                    if isinstance(result, dict):
                        embedding = result["continuous_features"].squeeze(0)
                    else:
                        embedding = result.squeeze(0)
                
                modality_embeddings.append(embedding)
                
            except Exception as e:
                warnings.warn(f"Error encoding {modality_key} for item: {e}")
                # Use zero embedding as fallback
                if modality_type == ModalityType.IMAGE:
                    dim = getattr(self.modality_encoders['image'], 'embedding_dim', 256)
                elif modality_type == ModalityType.TEXT:
                    dim = getattr(self.modality_encoders['text'], 'embedding_dim', 256)
                elif modality_type == ModalityType.CATEGORICAL:
                    cat_encoder = self.modality_encoders['categorical']
                    dim = cat_encoder.get_output_dim(num_fields=3) if hasattr(cat_encoder, 'get_output_dim') else 128
                else:  # CONTINUOUS
                    dim = getattr(self.modality_encoders.get('continuous'), 'output_dim', 64)
                modality_embeddings.append(torch.zeros(dim, device=device))
        
        # Concatenate all modality embeddings
        if modality_embeddings:
            item_embedding = torch.cat(modality_embeddings, dim=0)
        else:
            item_embedding = self.default_item_embedding
        
        return item_embedding
    
    def _encode_item_sequence(self, item_ids: List[Union[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a sequence of item IDs into embeddings
        
        Args:
            item_ids: List of item IDs in temporal order
            
        Returns:
            Tuple of (sequence_embeddings, sequence_mask)
            sequence_embeddings: (seq_len, item_embedding_dim)
            sequence_mask: (seq_len,) - 1 for valid items, 0 for padding/missing
        """
        device = next(self.parameters()).device
        
        # Handle empty sequence
        if not item_ids:
            seq_embeddings = torch.zeros(1, self.item_embedding_dim, device=device)
            seq_mask = torch.zeros(1, device=device)
            return seq_embeddings, seq_mask
        
        # Truncate or pad sequence
        if len(item_ids) > self.max_sequence_length:
            item_ids = item_ids[-self.max_sequence_length:]  # Keep most recent
        
        # Batch lookup item features
        try:
            item_features_list = self.item_lookup.batch_get_item_features(item_ids)
        except Exception as e:
            warnings.warn(f"Error in batch item lookup: {e}, falling back to individual lookups")
            item_features_list = []
            for item_id in item_ids:
                try:
                    features = self.item_lookup.get_item_features(item_id)
                    item_features_list.append(features)
                except Exception as e2:
                    warnings.warn(f"Error looking up item {item_id}: {e2}")
                    item_features_list.append(None)
        
        # Encode each item
        sequence_embeddings = []
        sequence_mask = []
        
        for i, (item_id, item_features) in enumerate(zip(item_ids, item_features_list)):
            if item_features is None:
                # Handle missing item based on strategy
                if self.missing_item_strategy == 'zero':
                    embedding = self.default_item_embedding
                    mask_value = 1  # Still include in sequence
                elif self.missing_item_strategy == 'skip':
                    continue  # Skip this item entirely
                elif self.missing_item_strategy == 'previous':
                    # Use previous embedding if available
                    if sequence_embeddings:
                        embedding = sequence_embeddings[-1]
                        mask_value = 1
                    else:
                        embedding = self.default_item_embedding
                        mask_value = 1
                else:
                    embedding = self.default_item_embedding
                    mask_value = 1
            else:
                # Encode item features
                embedding = self._encode_single_item(item_features)
                mask_value = 1
            
            sequence_embeddings.append(embedding)
            sequence_mask.append(mask_value)
        
        # Convert to tensors
        if sequence_embeddings:
            seq_embeddings = torch.stack(sequence_embeddings)  # (seq_len, item_embedding_dim)
            seq_mask = torch.tensor(sequence_mask, dtype=torch.float32, device=device)
        else:
            # Empty sequence after processing
            seq_embeddings = torch.zeros(1, self.item_embedding_dim, device=device)
            seq_mask = torch.zeros(1, device=device)
        
        return seq_embeddings, seq_mask
    
    def _apply_aggregation(self, lstm_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply aggregation strategy to LSTM outputs
        
        Args:
            lstm_outputs: (batch_size, seq_len, lstm_output_dim)
            mask: (batch_size, seq_len)
            
        Returns:
            Aggregated output: (batch_size, lstm_output_dim)
        """
        if self.aggregation_strategy == TemporalAggregationStrategy.LAST_HIDDEN:
            # Use the last valid hidden state
            batch_size, seq_len = mask.shape
            last_indices = mask.sum(dim=1).long() - 1  # Get last valid index for each sequence
            last_indices = torch.clamp(last_indices, 0, seq_len - 1)
            
            # Gather last hidden states
            batch_indices = torch.arange(batch_size, device=mask.device)
            last_outputs = lstm_outputs[batch_indices, last_indices]
            
            return last_outputs
        
        elif self.aggregation_strategy == TemporalAggregationStrategy.MEAN_POOLING:
            # Mean pooling with masking
            masked_outputs = lstm_outputs * mask.unsqueeze(-1)
            sum_outputs = masked_outputs.sum(dim=1)
            valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            return sum_outputs / valid_lengths
        
        elif self.aggregation_strategy == TemporalAggregationStrategy.MAX_POOLING:
            # Max pooling with masking
            masked_outputs = lstm_outputs * mask.unsqueeze(-1)
            masked_outputs = masked_outputs.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            max_outputs, _ = masked_outputs.max(dim=1)
            
            return max_outputs
        
        elif self.aggregation_strategy == TemporalAggregationStrategy.ATTENTION:
            # Attention pooling
            return self.attention_pooling(lstm_outputs, mask)
        
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
    
    def forward(self, temporal_dict: Dict[str, Union[List[Union[str, int]], List[List[Union[str, int]]]]]) -> torch.Tensor:
        """
        Forward pass for temporal encoding
        
        Args:
            temporal_dict: Dictionary mapping temporal field names to item ID lists
                          Single sample: {'prev_50_posts': [34, 56, 7646, 342]}
                          Batched: {'prev_50_posts': [[34, 56], [123, 456], [789, 101]]}
        
        Returns:
            Temporal embedding tensor of shape (batch_size, output_dim)
        """
        if not temporal_dict:
            # No temporal data, return zeros
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_dim, device=device)
        
        # Check if this is batched input by examining the first field
        first_field_name = next(iter(temporal_dict.keys()))
        first_field_value = temporal_dict[first_field_name]
        
        # Determine if this is batched input
        is_batched = isinstance(first_field_value, list) and len(first_field_value) > 0 and isinstance(first_field_value[0], list)
        
        if is_batched:
            return self._forward_batched(temporal_dict)
        else:
            return self._forward_single(temporal_dict)
    
    def _forward_single(self, temporal_dict: Dict[str, List[Union[str, int]]]) -> torch.Tensor:
        """Process a single sample"""
        # For now, process each temporal field separately and concatenate
        # In future versions, this could be made more sophisticated
        field_embeddings = []
        
        for field_name, item_ids in temporal_dict.items():
            if not item_ids:
                # Empty sequence
                device = next(self.parameters()).device
                field_embedding = torch.zeros(1, self.output_dim, device=device)
            else:
                # Encode item sequence
                seq_embeddings, seq_mask = self._encode_item_sequence(item_ids)
                
                # Add batch dimension and pass through LSTM
                seq_embeddings = seq_embeddings.unsqueeze(0)  # (1, seq_len, item_embedding_dim)
                seq_mask = seq_mask.unsqueeze(0)  # (1, seq_len)
                
                # LSTM forward pass
                lstm_outputs, _ = self.lstm(seq_embeddings)  # (1, seq_len, lstm_output_dim)
                
                # Apply aggregation
                aggregated = self._apply_aggregation(lstm_outputs, seq_mask)  # (1, lstm_output_dim)
                
                # Final projection
                field_embedding = self.projection(aggregated)  # (1, output_dim)
            
            field_embeddings.append(field_embedding)
        
        if len(field_embeddings) == 1:
            return field_embeddings[0]
        else:
            # Multiple temporal fields - take mean for now
            # Could be made configurable (concat, attention, etc.)
            stacked_embeddings = torch.stack(field_embeddings, dim=1)  # (1, num_fields, output_dim)
            return stacked_embeddings.mean(dim=1)  # (1, output_dim)
    
    def _forward_batched(self, temporal_dict: Dict[str, List[List[Union[str, int]]]]) -> torch.Tensor:
        """Process a batch of samples"""
        batch_size = len(next(iter(temporal_dict.values())))
        device = next(self.parameters()).device
        
        # Process each sample in the batch
        batch_embeddings = []
        for i in range(batch_size):
            # Extract single sample from batch
            sample_dict = {}
            for field_name, field_batch in temporal_dict.items():
                sample_dict[field_name] = field_batch[i] if i < len(field_batch) else []
            
            # Process single sample
            sample_embedding = self._forward_single(sample_dict)
            batch_embeddings.append(sample_embedding)
        
        # Stack all samples
        return torch.cat(batch_embeddings, dim=0)  # (batch_size, output_dim)
    
    def get_output_dim(self) -> int:
        """Get the output embedding dimension"""
        return self.output_dim


# Factory function
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
                           missing_item_strategy: str = 'zero') -> TemporalEncoder:
    """
    Factory function to create a TemporalEncoder
    
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
        Configured TemporalEncoder instance
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
    
    return TemporalEncoder(
        item_lookup=item_lookup,
        modality_encoders=enum_encoders,
        enabled_modalities=enabled_enum_modalities,
        aggregation_strategy=aggregation_strategy,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        bidirectional=bidirectional,
        output_dim=output_dim,
        max_sequence_length=max_sequence_length,
        missing_item_strategy=missing_item_strategy
    )


# Example ItemLookup implementation
class MockItemLookup(ItemLookupInterface):
    """Mock implementation for testing purposes"""
    
    def __init__(self):
        # Mock item database
        self.item_db = {
            34: {
                'image': {'main_image': '/images/post34.jpg'},
                'text': {'title': 'Tech Review', 'content': 'Great gadget review'},
                'categorical': {'category': 'tech', 'sentiment': 'positive'},
                'continuous': {'score': 4.5, 'engagement': 120}
            },
            56: {
                'image': {'main_image': '/images/post56.jpg'},
                'text': {'title': 'Travel Blog', 'content': 'Amazing vacation photos'},
                'categorical': {'category': 'travel', 'sentiment': 'positive'},
                'continuous': {'score': 4.8, 'engagement': 200}
            },
            7646: {
                'image': {'main_image': '/images/post7646.jpg'},
                'text': {'title': 'Food Recipe', 'content': 'Delicious pasta recipe'},
                'categorical': {'category': 'food', 'sentiment': 'neutral'},
                'continuous': {'score': 4.2, 'engagement': 95}
            }
        }
    
    def get_item_features(self, item_id: Union[str, int]) -> Dict[str, Any]:
        """Get features for a specific item"""
        item_id = int(item_id)
        return self.item_db.get(item_id, {
            'image': None,
            'text': None,
            'categorical': None,
            'continuous': None
        })


# Example usage and testing
if __name__ == "__main__":
    print("Testing Temporal Encoder...")
    
    # Create mock components for testing
    item_lookup = MockItemLookup()
    
    # Mock encoders (in practice, these would be your actual encoders)
    class MockImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_dim = 128
            self.linear = nn.Linear(1, 128)  # Dummy
        
        def forward_from_paths(self, paths_dict):
            return torch.randn(1, 128)
    
    class MockTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_dim = 256
        
        def forward(self, text_dict):
            return torch.randn(1, 256)
    
    class MockCategoricalEncoder(nn.Module):
        def forward(self, cat_dict, batch_size=1):
            return torch.randn(batch_size, 96)
        
        def get_output_dim(self, num_fields=3):
            return 96
    
    # Test configurations
    modality_encoders = {
        'image': MockImageEncoder(),
        'text': MockTextEncoder(),
        'categorical': MockCategoricalEncoder()
    }
    
    configs = [
        {
            'enabled_modalities': ['image', 'text'],
            'aggregation_strategy': 'last_hidden',
            'lstm_hidden_dim': 64,
            'output_dim': 128
        },
        {
            'enabled_modalities': ['categorical'],
            'aggregation_strategy': 'attention',
            'lstm_hidden_dim': 128,
            'output_dim': 256
        },
        {
            'enabled_modalities': ['image', 'text', 'categorical'],
            'aggregation_strategy': 'mean_pooling',
            'lstm_hidden_dim': 256,
            'bidirectional': True,
            'output_dim': 512
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        try:
            # Create temporal encoder
            temporal_encoder = create_temporal_encoder(
                item_lookup=item_lookup,
                modality_encoders=modality_encoders,
                **config
            )
            
            # Test with temporal data
            temporal_data = {
                'prev_50_posts': [34, 56, 7646, 342],  # 342 doesn't exist in mock DB
                'recent_purchases': [56, 34]
            }
            
            print(f"Enabled modalities: {config['enabled_modalities']}")
            print(f"Output dimension: {temporal_encoder.get_output_dim()}")
            
            # Forward pass
            with torch.no_grad():
                output = temporal_encoder(temporal_data)
                print(f"Input: {temporal_data}")
                print(f"Output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error with configuration {config}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTemporal encoder tests completed!")