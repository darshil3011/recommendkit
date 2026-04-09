"""
Hash-based categorical encoder with MLP embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any

from .base_categorical_encoder import BaseCategoricalEncoder, CategoricalAggregationStrategy


class MLPEmbedding(nn.Module):
    """MLP-based learnable embeddings for categorical features"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        if hidden_dims is None:
            hidden_size = max(32, min(256, vocab_size // 10))
            hidden_dims = [hidden_size]
        
        initial_embed_dim = max(16, min(128, int(np.sqrt(vocab_size))))
        self.initial_embedding = nn.Embedding(vocab_size, initial_embed_dim)
        
        layers = []
        input_dim = initial_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() > 1:
            indices = indices.squeeze(-1)
        indices = torch.clamp(indices, 0, self.vocab_size - 1)
        initial_emb = self.initial_embedding(indices)
        return self.mlp(initial_emb)


class HashCategoricalEncoder(BaseCategoricalEncoder):
    """
    Hash-based categorical encoder that handles multiple categorical fields.
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
        """
        Initialize hash-based categorical encoder.
        
        Args:
            aggregation_strategy: 'separate_concat' or 'joint_embedding'
            hash_vocab_size: Size of hash vocabulary
            embedding_dim: Output embedding dimension
            mlp_hidden_dims: Hidden dimensions for MLP layers
            dropout: Dropout probability
            activation: Activation function
            hash_seed: Seed for reproducible hashing
            num_categorical_fields: Number of categorical fields to expect
        """
        super().__init__(aggregation_strategy, embedding_dim, hash_vocab_size, hash_seed, num_categorical_fields)
        
        self.mlp_hidden_dims = mlp_hidden_dims if mlp_hidden_dims is not None else [64]
        self.dropout = dropout
        self.activation = activation
        
        # Initialize embeddings based on strategy
        self.field_embeddings = nn.ModuleDict()
        
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            # Pre-create embeddings for expected fields
            template = {
                'vocab_size': hash_vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dims': self.mlp_hidden_dims,
                'dropout': dropout,
                'activation': activation
            }
            for i in range(self.num_categorical_fields):
                field_name = f"field_{i}"
                self.field_embeddings[field_name] = MLPEmbedding(**template)
            
            # Projection for concatenated embeddings
            concat_input_dim = self.num_categorical_fields * embedding_dim
            self.concat_projection = nn.Linear(concat_input_dim, embedding_dim)
        else:  # JOINT_EMBEDDING
            self.joint_embedding = MLPEmbedding(
                vocab_size=hash_vocab_size,
                embedding_dim=embedding_dim,
                hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                activation=activation
            )
    
    def forward(self, categorical_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for categorical encoding.
        
        Args:
            categorical_dict: Dictionary mapping field names to categorical values
                             Single: {'country': 'USA', 'gender': 'male'}
                             Batched: {'country': ['USA', 'Canada'], 'gender': ['male', 'female']}
        
        Returns:
            Dictionary with categorical features: {"categorical_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        if not categorical_dict:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
                return {"categorical_features": torch.zeros(1, self.embedding_dim, device=device)}
            else:
                return {"categorical_features": self.default_embedding.unsqueeze(0)}
        
        # Check if batched
        first_value = next(iter(categorical_dict.values()))
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            # Process batch
            batch_size = len(first_value) if first_value else 1
            batch_embeddings = []
            for i in range(batch_size):
                sample = {k: (v[i] if isinstance(v, list) and i < len(v) else None)
                         for k, v in categorical_dict.items()}
                batch_embeddings.append(self._process_sample(sample))
            result = torch.cat(batch_embeddings, dim=0)
        else:
            result = self._process_sample(categorical_dict)
        
        return {"categorical_features": result}
    
    def _process_sample(self, categorical_dict: Dict[str, Any]) -> torch.Tensor:
        """Process a single sample"""
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if self.aggregation_strategy == CategoricalAggregationStrategy.SEPARATE_CONCAT:
            # Separate embedding + concatenation
            field_embeddings = []
            available_fields = list(categorical_dict.keys())[:self.num_categorical_fields]
            
            for i in range(self.num_categorical_fields):
                if i < len(available_fields):
                    field_name = available_fields[i]
                    category_value = categorical_dict[field_name]
                    hash_id = self._hash_category(field_name, category_value)
                    generic_field_name = f"field_{i}"
                    field_embedding_layer = self.field_embeddings[generic_field_name]
                    hash_tensor = torch.tensor([hash_id], dtype=torch.long, device=device)
                    field_emb = field_embedding_layer(hash_tensor)
                    field_embeddings.append(field_emb)
                else:
                    zero_emb = torch.zeros(1, self.embedding_dim, device=device)
                    field_embeddings.append(zero_emb)
            
            concatenated = torch.cat(field_embeddings, dim=1)
            return self.concat_projection(concatenated)
        
        else:  # JOINT_EMBEDDING
            joint_hash_id = self._hash_joint(categorical_dict)
            hash_tensor = torch.tensor([joint_hash_id], dtype=torch.long, device=device)
            return self.joint_embedding(hash_tensor)



