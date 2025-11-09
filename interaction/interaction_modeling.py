"""
interaction_modeling.py
Generates interaction embeddings from combined user and item embeddings
Using simple concatenation + MLP (attention-based code commented out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple

# Import helper components (kept for backward compatibility)
from .attention_utils import (
    MultiHeadAttention, 
    TransformerBlock, 
    InteractionType,
    create_attention_mask
)


class UserItemInteractionLayer(nn.Module):
    """
    User-item interaction layer
    Supports both simple concatenation + MLP and attention-based interaction
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 interaction_strategy: str = "bidirectional",
                 use_cls_token: bool = True,
                 use_simple_fusion: bool = True):
        """
        Args:
            embedding_dim: Dimension of user and item embeddings (should be same)
            num_attention_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate for MLP or transformer
            interaction_strategy: How user and item should interact
            use_cls_token: Whether to use CLS token
            use_simple_fusion: If True, use simple MLP; if False, use attention-based interaction
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_simple_fusion = use_simple_fusion
        
        if use_simple_fusion:
            # Simple MLP for interaction
            self.interaction_mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim * 2),  # Concatenate user + item
                nn.LayerNorm(embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            # Attention-based interaction
            self.num_attention_layers = num_attention_layers
            self.interaction_strategy = interaction_strategy
            self.use_cls_token = use_cls_token
            
            # Learnable CLS token for final interaction representation
            if use_cls_token:
                self.interaction_cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            
            # Position embeddings for interaction types (user vs item)
            self.interaction_type_embeddings = nn.Embedding(len(InteractionType), embedding_dim)
            
            # Stack of transformer blocks for interaction modeling
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(embedding_dim, num_heads, dropout=dropout)
                for _ in range(num_attention_layers)
            ])
            
            # Final projection layer
            self.final_projection = nn.Linear(embedding_dim, embedding_dim)
            
            # Optional: Additional learnable parameters for different interaction strategies
            if interaction_strategy != "bidirectional":
                self.strategy_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                user_embedding: torch.Tensor,
                item_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_embedding: [batch_size, embedding_dim] - Combined user representation
            item_embedding: [batch_size, embedding_dim] - Combined item representation
        
        Returns:
            interaction_embedding: [batch_size, embedding_dim] - Final user-item interaction representation
        """
        # Validate input dimensions
        assert user_embedding.shape == item_embedding.shape, \
            f"User and item embeddings must have same shape, got {user_embedding.shape} and {item_embedding.shape}"
        
        if self.use_simple_fusion:
            # Simple concatenation + MLP
            concatenated = torch.cat([user_embedding, item_embedding], dim=1)  # [batch_size, 2*embedding_dim]
            interaction_embedding = self.interaction_mlp(concatenated)  # [batch_size, embedding_dim]
        else:
            # Attention-based interaction
            batch_size = user_embedding.size(0)
            device = user_embedding.device
            
            # Expand embeddings to sequence format
            user_seq = user_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            item_seq = item_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            
            # Add interaction type embeddings
            user_type_idx = torch.tensor([0], device=device)  # USER = 0
            item_type_idx = torch.tensor([1], device=device)  # ITEM = 1
            
            user_type_embedding = self.interaction_type_embeddings(user_type_idx)  # [1, embedding_dim]
            item_type_embedding = self.interaction_type_embeddings(item_type_idx)  # [1, embedding_dim]
            
            user_type_embedding = user_type_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 1, embedding_dim]
            item_type_embedding = item_type_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 1, embedding_dim]
            
            # Add type embeddings
            user_seq = user_seq + user_type_embedding
            item_seq = item_seq + item_type_embedding
            
            # Construct interaction sequence
            if self.use_cls_token:
                cls_tokens = self.interaction_cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embedding_dim]
                interaction_sequence = torch.cat([cls_tokens, user_seq, item_seq], dim=1)  # [batch_size, 3, embedding_dim]
            else:
                interaction_sequence = torch.cat([user_seq, item_seq], dim=1)  # [batch_size, 2, embedding_dim]
            
            # Create attention mask
            seq_len = interaction_sequence.size(1)
            attention_mask = create_attention_mask(batch_size, seq_len, device=device)
            
            # Apply transformer layers
            hidden_states = interaction_sequence
            for layer in self.transformer_layers:
                hidden_states = layer(hidden_states, attention_mask)
            
            # Extract interaction representation
            if self.use_cls_token:
                interaction_embedding = hidden_states[:, 0, :]  # [batch_size, embedding_dim]
            else:
                interaction_embedding = hidden_states.mean(dim=1)  # [batch_size, embedding_dim]
            
            # Apply final projection
            interaction_embedding = self.final_projection(interaction_embedding)
        
        return interaction_embedding


class MultipleInteractionStrategies(nn.Module):
    """
    Combines multiple interaction strategies and learns to weight them
    """
    
    def __init__(self,
                 embedding_dim: int,
                 strategies: List[str] = ["bidirectional"],
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            embedding_dim: Embedding dimension
            strategies: List of interaction strategies to use
            num_attention_layers: Number of attention layers per strategy
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.strategies = strategies
        
        # Create separate interaction layers for each strategy
        self.interaction_layers = nn.ModuleDict()
        for strategy in strategies:
            self.interaction_layers[strategy] = UserItemInteractionLayer(
                embedding_dim=embedding_dim,
                num_attention_layers=num_attention_layers,
                num_heads=num_heads,
                dropout=dropout,
                interaction_strategy=strategy,
                use_cls_token=True
            )
        
        # If multiple strategies, learn to combine them
        if len(strategies) > 1:
            self.strategy_weights = nn.Linear(len(strategies) * embedding_dim, embedding_dim)
            self.strategy_gate = nn.Linear(len(strategies) * embedding_dim, len(strategies))
    
    def forward(self,
                user_embedding: torch.Tensor,
                item_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_embedding: [batch_size, embedding_dim]
            item_embedding: [batch_size, embedding_dim]
        
        Returns:
            interaction_embedding: [batch_size, embedding_dim]
        """
        if len(self.strategies) == 1:
            # Single strategy
            strategy = self.strategies[0]
            return self.interaction_layers[strategy](user_embedding, item_embedding)
        
        # Multiple strategies
        strategy_outputs = []
        for strategy in self.strategies:
            output = self.interaction_layers[strategy](user_embedding, item_embedding)
            strategy_outputs.append(output)
        
        # Concatenate all strategy outputs
        combined = torch.cat(strategy_outputs, dim=-1)  # [batch_size, len(strategies) * embedding_dim]
        
        # Learn strategy weights
        strategy_weights = F.softmax(self.strategy_gate(combined), dim=-1)  # [batch_size, len(strategies)]
        
        # Weighted combination
        weighted_outputs = []
        for i, output in enumerate(strategy_outputs):
            weight = strategy_weights[:, i:i+1]  # [batch_size, 1]
            weighted_outputs.append(weight * output)
        
        # Sum weighted outputs
        final_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)  # [batch_size, embedding_dim]
        
        return final_output


class InteractionEmbeddingGenerator(nn.Module):
    """
    Complete interaction embedding generator that can be easily integrated
    """
    
    def __init__(self,
                 embedding_dim: int,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 interaction_strategy: str = "bidirectional",
                 use_multiple_strategies: bool = False,
                 use_simple_fusion: bool = True):
        """
        Args:
            embedding_dim: Embedding dimension
            num_attention_layers: Number of attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            interaction_strategy: Single strategy to use
            use_multiple_strategies: Whether to use multiple strategies
            use_simple_fusion: If True, use simple MLP; if False, use attention-based interaction
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        if use_multiple_strategies:
            # Use multiple strategies
            strategies = ["bidirectional", "user_to_item", "item_to_user"]
            self.interaction_model = MultipleInteractionStrategies(
                embedding_dim=embedding_dim,
                strategies=strategies,
                num_attention_layers=num_attention_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # Use single strategy
            self.interaction_model = UserItemInteractionLayer(
                embedding_dim=embedding_dim,
                num_attention_layers=num_attention_layers,
                num_heads=num_heads,
                dropout=dropout,
                interaction_strategy=interaction_strategy,
                use_cls_token=True,
                use_simple_fusion=use_simple_fusion
            )
    
    def forward(self,
                combined_user_embedding: torch.Tensor,
                combined_item_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate interaction embedding from combined user and item embeddings
        
        Args:
            combined_user_embedding: [batch_size, embedding_dim] - From feature_fusion.py
            combined_item_embedding: [batch_size, embedding_dim] - From feature_fusion.py
        
        Returns:
            interaction_embedding: [batch_size, embedding_dim] - Ready for MLP + classification
        """
        return self.interaction_model(combined_user_embedding, combined_item_embedding)


# Removed unnecessary factory function - use direct class instantiation instead


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    embedding_dim = 256
    batch_size = 32
    
    # Simulate combined embeddings from feature_fusion.py
    combined_user_embedding = torch.randn(batch_size, embedding_dim)
    combined_item_embedding = torch.randn(batch_size, embedding_dim)
    
    print("=== Testing Direct Class Instantiation ===")
    
    # Method 1: Direct instantiation with clear parameters
    interaction_generator = InteractionEmbeddingGenerator(
        embedding_dim=embedding_dim,
        num_attention_layers=2,
        num_heads=8,  # Must divide embedding_dim
        dropout=0.1,
        interaction_strategy="bidirectional"
    )
    
    interaction_embedding = interaction_generator(combined_user_embedding, combined_item_embedding)
    
    print(f"Single strategy interaction embedding shape: {interaction_embedding.shape}")  # [32, 256]
    
    print("\n=== Testing Multiple Strategy Interaction ===")
    
    # Method 2: Multiple strategies
    multi_interaction_generator = InteractionEmbeddingGenerator(
        embedding_dim=embedding_dim,
        num_attention_layers=2,
        num_heads=8,
        dropout=0.1,
        use_multiple_strategies=True
    )
    
    multi_interaction_embedding = multi_interaction_generator(combined_user_embedding, combined_item_embedding)
    
    print(f"Multi-strategy interaction embedding shape: {multi_interaction_embedding.shape}")  # [32, 256]
    
    print("\n=== Testing Different Configurations ===")
    
    # Test different configurations with direct instantiation
    configs = [
        {"num_attention_layers": 1, "num_heads": 4, "dropout": 0.1},
        {"num_attention_layers": 2, "num_heads": 8, "dropout": 0.1}, 
        {"num_attention_layers": 3, "num_heads": 16, "dropout": 0.15}
    ]
    
    for i, config in enumerate(configs):
        generator = InteractionEmbeddingGenerator(
            embedding_dim=embedding_dim,
            interaction_strategy="bidirectional",
            **config
        )
        output = generator(combined_user_embedding, combined_item_embedding)
        print(f"Config {i+1} interaction embedding shape: {output.shape}")
    
    print("✅ Interaction modeling completed successfully!")