import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import hashlib
import numpy as np


class CategoricalAggregationStrategy(Enum):
    """Supported aggregation strategies for multiple categorical fields"""
    SEPARATE_CONCAT = "separate_concat"  # Embed each field separately, then concatenate
    JOINT_EMBEDDING = "joint_embedding"  # Create joint categorical features, then embed


class MLPEmbedding(nn.Module):
    """
    MLP-based learnable embeddings for categorical features
    
    Args:
        vocab_size: Number of unique categories (hash space size)
        embedding_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions for MLP
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'tanh')
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            # Create a simple 2-layer MLP with reasonable hidden size
            hidden_size = max(32, min(256, vocab_size // 10))
            hidden_dims = [hidden_size]
        
        # Initial embedding layer (traditional lookup)
        initial_embed_dim = max(16, min(128, int(np.sqrt(vocab_size))))
        self.initial_embedding = nn.Embedding(vocab_size, initial_embed_dim)
        
        # Build MLP layers
        layers = []
        input_dim = initial_embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final projection to desired embedding dimension
        layers.append(nn.Linear(input_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
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
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            indices: Tensor of category indices, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Embeddings tensor, shape (batch_size, embedding_dim)
        """
        # Ensure indices are the right shape
        if indices.dim() > 1:
            indices = indices.squeeze(-1)
            
        # Clamp indices to valid range (safety against hash issues)
        indices = torch.clamp(indices, 0, self.vocab_size - 1)
        
        # Get initial embeddings
        initial_emb = self.initial_embedding(indices)  # (batch_size, initial_embed_dim)
        
        # Pass through MLP
        output = self.mlp(initial_emb)  # (batch_size, embedding_dim)
        
        return output


class CategoricalEncoder(nn.Module):
    """
    Hash-based categorical encoder that handles multiple categorical fields
    
    Args:
        aggregation_strategy: How to combine multiple categorical fields
        hash_vocab_size: Size of hash vocabulary for each field (or joint space)
        embedding_dim: Output embedding dimension per field (for separate) or total (for joint)
        mlp_hidden_dims: Hidden dimensions for MLP layers
        dropout: Dropout probability
        activation: Activation function
        hash_seed: Seed for reproducible hashing (important for consistency!)
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, CategoricalAggregationStrategy] = CategoricalAggregationStrategy.SEPARATE_CONCAT,
                 hash_vocab_size: int = 10000,
                 embedding_dim: int = 64,
                 mlp_hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 hash_seed: int = 42,
                 num_categorical_fields: int = 3):
        
        super().__init__()
        
        # Convert string arguments to enums
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = CategoricalAggregationStrategy(aggregation_strategy.lower())
        
        self.aggregation_strategy = aggregation_strategy
        self.hash_vocab_size = hash_vocab_size
        self.embedding_dim = embedding_dim
        # CRITICAL: Use 'is not None' check, not 'or', to preserve empty lists []
        self.mlp_hidden_dims = mlp_hidden_dims if mlp_hidden_dims is not None else [64]
        self.dropout = dropout
        self.activation = activation
        self.hash_seed = hash_seed
        self.num_categorical_fields = max(num_categorical_fields, 1)  # At least 1 field
        
        # Initialize embeddings based on strategy
        self.field_embeddings = nn.ModuleDict()
        
        # Track seen fields for separate concat strategy - INITIALIZE EARLY
        self._seen_fields = set()
        
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            # Pre-create embeddings for expected number of fields - NO LAZY INITIALIZATION
            self._field_embedding_template = {
                'vocab_size': hash_vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dims': mlp_hidden_dims,
                'dropout': dropout,
                'activation': activation
            }
            
            # Create embeddings for expected number of fields immediately
            for i in range(self.num_categorical_fields):
                field_name = f"field_{i}"  # Generic field names
                self._get_or_create_field_embedding(field_name)
            
            # Create projection layer immediately based on field count
            concat_input_dim = self.num_categorical_fields * embedding_dim
            self.concat_projection = nn.Linear(concat_input_dim, embedding_dim)
            self._projection_initialized = True
            
        elif self.aggregation_strategy == CategoricalAggregationStrategy.JOINT_EMBEDDING:
            # Create single joint embedding
            self.joint_embedding = MLPEmbedding(
                vocab_size=hash_vocab_size,
                embedding_dim=embedding_dim,
                hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                activation=activation
            )
        
        # For handling missing/empty inputs
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def _hash_category(self, field_name: str, category_value: Any) -> int:
        """
        Create deterministic hash for category value
        
        Args:
            field_name: Name of the categorical field
            category_value: Category value (will be converted to string)
            
        Returns:
            Hash ID in range [0, hash_vocab_size)
        """
        if category_value is None:
            # Use special token for None values
            hash_input = f"{field_name}:<NULL>"
        else:
            # Convert to string and create field-specific hash
            str_value = str(category_value).strip().lower()  # Normalize
            hash_input = f"{field_name}:{str_value}"
        
        # Use SHA-256 for better distribution and reproducibility
        hash_obj = hashlib.sha256(f"{self.hash_seed}:{hash_input}".encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Map to vocabulary range
        return hash_int % self.hash_vocab_size
    
    def _initialize_concat_projection(self, num_fields: int):
        """Initialize the projection layer for concatenated embeddings"""
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            input_dim = num_fields * self.embedding_dim
            self.concat_projection = nn.Linear(input_dim, self.embedding_dim)
            self._projection_initialized = True
    
    def _get_or_create_field_embedding(self, field_name: str) -> MLPEmbedding:
        """
        Get or create embedding for a field (for separate concat strategy)
        
        Args:
            field_name: Name of the categorical field
            
        Returns:
            MLPEmbedding instance for this field
        """
        if field_name not in self.field_embeddings:
            # Create new embedding for this field
            self.field_embeddings[field_name] = MLPEmbedding(**self._field_embedding_template)
            self._seen_fields.add(field_name)
        
        return self.field_embeddings[field_name]
    
    def _hash_joint_features(self, categorical_dict: Dict[str, Any]) -> int:
        """
        Create joint hash for all categorical features
        
        Args:
            categorical_dict: Dictionary of categorical values
            
        Returns:
            Joint hash ID
        """
        # Sort fields for consistent ordering
        sorted_items = sorted(categorical_dict.items())
        
        # Create combined string
        combined_parts = []
        for field_name, category_value in sorted_items:
            if category_value is None:
                combined_parts.append(f"{field_name}:<NULL>")
            else:
                str_value = str(category_value).strip().lower()
                combined_parts.append(f"{field_name}:{str_value}")
        
        combined_string = "|".join(combined_parts)
        
        # Hash the combined string
        hash_obj = hashlib.sha256(f"{self.hash_seed}:{combined_string}".encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        
        return hash_int % self.hash_vocab_size
    
    def _separate_concat_forward(self, categorical_dict: Dict[str, Any], batch_size: int = 1) -> torch.Tensor:
        """
        Separate embedding + concatenation strategy with hashing
        Pads or truncates to match expected num_categorical_fields
        
        Args:
            categorical_dict: Dictionary of categorical values
            batch_size: Batch size for tensor creation
            
        Returns:
            Concatenated embeddings tensor
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        field_embeddings = []
        available_fields = list(categorical_dict.keys())
        
        # Process up to num_categorical_fields (pad with zeros if needed, truncate if too many)
        for i in range(self.num_categorical_fields):
            if i < len(available_fields):
                field_name = available_fields[i]
                category_value = categorical_dict[field_name]
                
                # Hash the category
                hash_id = self._hash_category(field_name, category_value)
                
                # Get field embedding (use generic field name)
                generic_field_name = f"field_{i}"
                field_embedding_layer = self.field_embeddings[generic_field_name]
                
                # Convert to tensor
                hash_tensor = torch.tensor([hash_id] * batch_size, dtype=torch.long, device=device)
                
                # Get embedding
                field_emb = field_embedding_layer(hash_tensor)  # (batch_size, embedding_dim)
                field_embeddings.append(field_emb)
            else:
                # Missing field -> zero embedding
                zero_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
                field_embeddings.append(zero_emb)
        
        # Concatenate all field embeddings (always num_categorical_fields * embedding_dim)
        concatenated = torch.cat(field_embeddings, dim=1)  # (batch_size, num_fields * embedding_dim)
        
        # Project to target embedding dimension (projection already initialized!)
        projected = self.concat_projection(concatenated)  # (batch_size, embedding_dim)
        return projected
    
    def _joint_embedding_forward(self, categorical_dict: Dict[str, Any], batch_size: int = 1) -> torch.Tensor:
        """
        Joint embedding strategy with hashing
        
        Args:
            categorical_dict: Dictionary of categorical values
            batch_size: Batch size for tensor creation
            
        Returns:
            Joint embedding tensor
        """
        # Hash all features together
        joint_hash_id = self._hash_joint_features(categorical_dict)
        
        # Convert to tensor and get embedding
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        hash_tensor = torch.tensor([joint_hash_id] * batch_size, dtype=torch.long, device=device)
        
        joint_emb = self.joint_embedding(hash_tensor)  # (batch_size, embedding_dim)
        return joint_emb
    
    def forward(self, categorical_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for categorical encoding that handles both single and batched inputs
        
        Args:
            categorical_dict: Dictionary mapping field names to categorical values
                             Single: {'country': 'USA', 'gender': 'male'}
                             Batched: {'country': ['USA', 'Canada'], 'gender': ['male', 'female']}
        
        Returns:
            Dictionary with categorical features: {"categorical_features": torch.Tensor}
        """
        if not categorical_dict:
            # Empty input, return default embedding
            if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
                default_tensor = torch.zeros(1, self.embedding_dim, device=next(self.parameters()).device)
            else:
                default_tensor = self.default_embedding.unsqueeze(0)
            return {"categorical_features": default_tensor}
        
        # Determine if this is batched input by checking if any value is a list
        first_value = next(iter(categorical_dict.values()))
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            return self._forward_batched(categorical_dict)
        else:
            # Single sample
            result = self._forward_single(categorical_dict)
            return {"categorical_features": result}
    
    def _forward_single(self, categorical_dict: Dict[str, Any]) -> torch.Tensor:
        """Process a single sample"""
        # Apply aggregation strategy
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            return self._separate_concat_forward(categorical_dict, batch_size=1)
        elif self.aggregation_strategy == CategoricalAggregationStrategy.JOINT_EMBEDDING:
            return self._joint_embedding_forward(categorical_dict, batch_size=1)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
    
    def _forward_batched(self, categorical_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a batch of samples"""
        # Get batch size from first non-empty field
        batch_size = 0
        for field_values in categorical_dict.values():
            if isinstance(field_values, list) and len(field_values) > 0:
                batch_size = len(field_values)
                break
        
        if batch_size == 0:
            if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
                default_tensor = torch.zeros(1, self.embedding_dim, device=next(self.parameters()).device)
            else:
                default_tensor = self.default_embedding.unsqueeze(0)
            return {"categorical_features": default_tensor}
        
        # Process each sample in the batch
        batch_embeddings = []
        for i in range(batch_size):
            # Extract single sample from batch
            sample_dict = {}
            for field_name, field_values in categorical_dict.items():
                if isinstance(field_values, list) and i < len(field_values):
                    sample_dict[field_name] = field_values[i]
                else:
                    sample_dict[field_name] = None
            
            # Process single sample
            sample_embedding = self._forward_single(sample_dict)
            batch_embeddings.append(sample_embedding)
        
        # Stack into batch
        batch_result = torch.cat(batch_embeddings, dim=0)  # (batch_size, embedding_dim)
        return {"categorical_features": batch_result}
    
    def get_output_dim(self, num_fields: Optional[int] = None) -> int:
        """
        Get the output embedding dimension
        
        Args:
            num_fields: Number of categorical fields (only needed for separate_concat estimation)
            
        Returns:
            Output dimension
        """
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            if num_fields is not None:
                return num_fields * self.embedding_dim
            elif self._seen_fields:
                return len(self._seen_fields) * self.embedding_dim
            else:
                # Return per-field dimension if number unknown
                return self.embedding_dim
        else:  # JOINT_EMBEDDING
            return self.embedding_dim
    
    def get_hash_info(self, categorical_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get hash information for debugging and analysis
        
        Args:
            categorical_dict: Dictionary of categorical values
            
        Returns:
            Dictionary with hash information
        """
        hash_info = {}
        
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            hash_info['field_hashes'] = {}
            for field_name, category_value in categorical_dict.items():
                hash_id = self._hash_category(field_name, category_value)
                hash_info['field_hashes'][field_name] = {
                    'original_value': category_value,
                    'hash_id': hash_id,
                    'hash_ratio': hash_id / self.hash_vocab_size
                }
        else:  # JOINT_EMBEDDING
            joint_hash_id = self._hash_joint_features(categorical_dict)
            hash_info['joint_hash'] = {
                'original_dict': categorical_dict,
                'hash_id': joint_hash_id,
                'hash_ratio': joint_hash_id / self.hash_vocab_size
            }
        
        return hash_info
    
    def estimate_collision_probability(self, num_unique_categories_per_field: Dict[str, int]) -> Dict[str, float]:
        """
        Estimate hash collision probability for analysis
        
        Args:
            num_unique_categories_per_field: Estimated number of unique categories per field
            
        Returns:
            Dictionary with collision probability estimates
        """
        collision_probs = {}
        
        for field_name, num_unique in num_unique_categories_per_field.items():
            if num_unique <= self.hash_vocab_size:
                # Birthday paradox approximation
                prob = 1 - np.exp(-num_unique * (num_unique - 1) / (2 * self.hash_vocab_size))
            else:
                # Guarantee collision if more categories than hash space
                prob = 1.0
            
            collision_probs[field_name] = prob
        
        return collision_probs


# Factory function
def create_categorical_encoder(aggregation_strategy: str = "separate_concat",
                              hash_vocab_size: int = 10000,
                              embedding_dim: int = 64,
                              mlp_hidden_dims: List[int] = None,
                              dropout: float = 0.1,
                              activation: str = 'relu',
                              hash_seed: int = 42,
                              num_categorical_fields: int = 3) -> CategoricalEncoder:
    """
    Factory function to create a hash-based CategoricalEncoder
    
    Args:
        aggregation_strategy: 'separate_concat' or 'joint_embedding'
        hash_vocab_size: Size of hash space (larger = fewer collisions)
        embedding_dim: Output embedding dimension (per field for separate, total for joint)
        mlp_hidden_dims: Hidden layer dimensions for MLP
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'tanh')
        hash_seed: Seed for reproducible hashing (important!)
        num_categorical_fields: Number of categorical fields to expect (eliminates lazy initialization)
        
    Returns:
        Configured CategoricalEncoder instance
    """
    return CategoricalEncoder(
        aggregation_strategy=aggregation_strategy,
        hash_vocab_size=hash_vocab_size,
        embedding_dim=embedding_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        activation=activation,
        hash_seed=hash_seed,
        num_categorical_fields=num_categorical_fields
    )


# Removed unnecessary utility functions - the encoder handles everything internally


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hash-Based Categorical Encoders...")
    
    # Test configurations
    configs = [
        {
            "aggregation_strategy": "separate_concat",
            "hash_vocab_size": 5000,
            "embedding_dim": 32,
            "mlp_hidden_dims": [64, 32],
            "dropout": 0.1,
            "activation": "relu",
            "hash_seed": 42
        },
        {
            "aggregation_strategy": "joint_embedding", 
            "hash_vocab_size": 10000,
            "embedding_dim": 128,
            "mlp_hidden_dims": [128],
            "dropout": 0.2,
            "activation": "gelu",
            "hash_seed": 42
        }
    ]
    
    # Sample test data
    test_cases = [
        {'country': 'USA', 'gender': 'male', 'state': 'texas'},
        {'country': 'UK', 'gender': 'female', 'state': 'london'},
        {'country': 'Unknown_Country', 'gender': 'other', 'state': None},  # Test unknown/None
        {'country': 'USA', 'gender': 'male', 'state': 'texas'},  # Same as first (should get same hash)
        {'new_field': 'value', 'another_field': 123},  # Completely different fields
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config['aggregation_strategy']} ---")
        
        try:
            # Create encoder
            encoder = create_categorical_encoder(**config)
            print(f"Hash vocab size: {config['hash_vocab_size']}")
            print(f"Embedding dim: {config['embedding_dim']}")
            
            # Test with different inputs
            for j, test_input in enumerate(test_cases):
                print(f"\nTest case {j+1}: {test_input}")
                
                # Validate input
                validation = validate_hash_encoder_input(test_input)
                print(f"Validation: {validation}")
                
                # Get hash info
                hash_info = encoder.get_hash_info(test_input)
                print(f"Hash info: {hash_info}")
                
                # Forward pass
                with torch.no_grad():
                    output = encoder(test_input, batch_size=2)
                    print(f"Output shape: {output.shape}")
                    print(f"Output dimension: {encoder.get_output_dim(len(test_input))}")
            
            # Test collision estimation
            estimated_categories = {'country': 200, 'gender': 5, 'state': 50}
            collision_probs = encoder.estimate_collision_probability(estimated_categories)
            print(f"\nEstimated collision probabilities: {collision_probs}")
            
            # Test recommended hash size
            total_categories = sum(estimated_categories.values())
            recommended_size = recommend_hash_vocab_size(total_categories)
            print(f"Recommended hash vocab size for {total_categories} categories: {recommended_size}")
                
        except Exception as e:
            print(f"Error with configuration {config}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nHash-based categorical encoder tests completed!")