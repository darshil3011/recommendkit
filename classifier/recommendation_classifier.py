"""
recommendation_classifier.py
Main recommendation classifier that combines MLP + Classification head
Handles training and inference for recommendation systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple

# Import helper components
from .classification_utils import (
    MLPTower,
    BinaryClassificationHead,
    LossManager,
    LossType,
    create_optimizer,
    create_scheduler
)


class RecommendationClassifier(nn.Module):
    """
    Complete recommendation classifier with MLP + Classification head
    Supports multiple loss functions for training engagement prediction
    """
    
    def __init__(self,
                 embedding_dim: int,
                 mlp_hidden_dims: List[int] = [512, 256, 128],
                 mlp_activation: str = "relu",
                 mlp_dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 loss_type: str = "bce",
                 loss_weights: Optional[Dict[str, float]] = None,
                 contrastive_margin: float = 1.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Args:
            embedding_dim: Dimension of interaction embeddings
            mlp_hidden_dims: List of hidden dimensions for MLP
            mlp_activation: Activation function for MLP
            mlp_dropout: Dropout rate for MLP
            use_batch_norm: Whether to use batch normalization
            loss_type: Type of loss ("bce", "contrastive", "combined", "focal", "triplet")
            loss_weights: Weights for combining multiple losses
            contrastive_margin: Margin for contrastive loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        
        # Set default loss weights
        if loss_weights is None:
            loss_weights = {"bce": 1.0, "contrastive": 1.0, "focal": 1.0}
        
        # MLP tower
        mlp_output_dim = mlp_hidden_dims[-1] if mlp_hidden_dims else embedding_dim
        self.mlp = MLPTower(
            input_dim=embedding_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=mlp_output_dim,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_batch_norm=use_batch_norm,
            final_activation=None  # No activation on final MLP layer
        )
        
        # Classification head
        self.classification_head = BinaryClassificationHead(mlp_output_dim)
        
        # Loss manager
        self.loss_manager = LossManager(
            loss_type=loss_type,
            bce_weight=loss_weights.get("bce", 1.0),
            contrastive_weight=loss_weights.get("contrastive", 1.0),
            focal_weight=loss_weights.get("focal", 1.0),
            contrastive_margin=contrastive_margin,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
        
    def forward(self, 
            interaction_embeddings: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through MLP + Classification head"""
    
        # Pass through MLP
        mlp_embeddings = self.mlp(interaction_embeddings)
        
        # Pass through classification head
        logits = self.classification_head(mlp_embeddings)
        
        if labels is not None:
            loss = self.loss_manager.compute_loss(logits, mlp_embeddings, labels)
            if return_embeddings:
                return loss, logits, mlp_embeddings
            else:
                return loss, logits
        else:
            if return_embeddings:
                return logits, mlp_embeddings
            else:
                return logits
        
    def predict_proba(self, interaction_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get engagement probabilities (for inference)
        
        Args:
            interaction_embeddings: [batch_size, embedding_dim]
            
        Returns:
            probabilities: [batch_size, 1] - Engagement probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(interaction_embeddings)
            return torch.sigmoid(logits)
    
    def predict(self, 
               interaction_embeddings: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions (for inference)
        
        Args:
            interaction_embeddings: [batch_size, embedding_dim]
            threshold: Classification threshold
            
        Returns:
            predictions: [batch_size, 1] - Binary predictions
        """
        probabilities = self.predict_proba(interaction_embeddings)
        return (probabilities > threshold).float()
    
    def get_embeddings(self, interaction_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get MLP embeddings (useful for similarity search, clustering, etc.)
        
        Args:
            interaction_embeddings: [batch_size, embedding_dim]
            
        Returns:
            mlp_embeddings: [batch_size, mlp_output_dim]
        """
        self.eval()
        with torch.no_grad():
            return self.mlp(interaction_embeddings)


# Training will be handled in separate trainer module


# Removed unnecessary factory function - use direct class instantiation instead


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    embedding_dim = 256
    batch_size = 32
    
    # Simulate interaction embeddings (from interaction_modeling.py)
    interaction_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Simulate binary engagement labels (1 = engaged, 0 = not engaged)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    print("=== Testing Recommendation Classifier ===")
    
    # Test different configurations with direct instantiation
    configs = [
        {"hidden_dims": [128], "dropout": 0.1, "loss_type": "bce"},
        {"hidden_dims": [512, 256], "dropout": 0.1, "loss_type": "bce"},
        {"hidden_dims": [512, 256, 128, 64], "dropout": 0.2, "loss_type": "contrastive"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Testing Config {i+1}: {config['hidden_dims']} + {config['loss_type'].upper()} ---")
        
        classifier = RecommendationClassifier(
            embedding_dim=embedding_dim,
            **config
        )
        
        # Training mode
        loss, logits = classifier(interaction_embeddings, labels)
        probabilities = torch.sigmoid(logits)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits shape: {logits.shape}")
        print(f"Probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        
        # Inference mode
        inference_probs = classifier.predict_proba(interaction_embeddings)
        predictions = classifier.predict(interaction_embeddings)
        embeddings = classifier.get_embeddings(interaction_embeddings)
        
        print(f"Inference - Probs: {inference_probs.shape}, Preds: {predictions.shape}, Embeddings: {embeddings.shape}")
    
    print("✅ Recommendation classifier completed successfully!")