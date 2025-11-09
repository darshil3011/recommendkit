"""
classification_utils.py
Helper functions and reusable components for classification head
Includes MLP blocks, loss functions, and utility components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Literal
from enum import Enum
import math


class LossType(Enum):
    """Supported loss functions for training"""
    BCE = "bce"  # Binary Cross Entropy
    CONTRASTIVE = "contrastive"  # Contrastive Loss
    COMBINED = "combined"  # Combination of BCE and Contrastive


class MLPBlock(nn.Module):
    """
    A single MLP block with configurable activation and dropout
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ("relu", "gelu", "leaky_relu", "swish")
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        if use_batch_norm:  # Using the flag but implementing LayerNorm
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None
            
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "swish":
            self.activation = nn.SiLU()  # SiLU is equivalent to Swish
        else:
            self.activation = nn.ReLU()  # Default to ReLU
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        x = self.linear(x)
        
        if self.norm is not None:
            x = self.norm(x)
            
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class MLPTower(nn.Module):
    """
    Multi-layer perceptron with configurable architecture
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 final_activation: Optional[str] = None):
        """
        Args:
            input_dim: Input dimension (interaction embedding size)
            hidden_dims: List of hidden layer dimensions
            output_dim: Final output dimension
            activation: Activation function for hidden layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            final_activation: Activation for final layer (None means no activation)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(
                input_dim=current_dim,
                output_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            ))
            current_dim = hidden_dim
            
        # Final layer (no activation by default)
        final_layer = nn.Linear(current_dim, output_dim)
        layers.append(final_layer)
        
        # Final activation if specified
        if final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif final_activation == "tanh":
            layers.append(nn.Tanh())
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        return self.mlp(x)


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head for engagement prediction
    """
    
    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Input dimension from MLP
        """
        super().__init__()
        
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            logits: [batch_size, 1] - Raw logits (apply sigmoid for probabilities)
        """
        return self.classifier(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning representations
    Pulls positive pairs closer, pushes negative pairs apart
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Args:
            margin: Margin for negative pairs
            temperature: Temperature scaling for similarities
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embedding_dim] - Interaction embeddings
            labels: [batch_size] - Binary labels (1 for positive, 0 for negative)
            
        Returns:
            loss: Scalar contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        batch_size = embeddings.size(0)
        distances = torch.cdist(embeddings, embeddings, p=2)  # [batch_size, batch_size]
        
        # Create pairwise labels (1 if both samples have same label, 0 otherwise)
        labels_expanded = labels.unsqueeze(1).expand(-1, batch_size)  # [batch_size, batch_size]
        pairwise_labels = (labels_expanded == labels_expanded.T).float()
        
        # Mask out diagonal (distance to self)
        mask = torch.eye(batch_size, device=embeddings.device)
        distances = distances * (1 - mask) + mask * 1e6  # Set diagonal to large value
        pairwise_labels = pairwise_labels * (1 - mask)
        
        # Contrastive loss calculation
        positive_loss = pairwise_labels * distances.pow(2)
        negative_loss = (1 - pairwise_labels) * F.relu(self.margin - distances).pow(2)
        
        # Average over valid pairs
        num_valid_pairs = (1 - mask).sum()
        loss = (positive_loss.sum() + negative_loss.sum()) / num_valid_pairs
        
        return loss


class TripletLoss(nn.Module):
    """
    Alternative triplet loss for learning representations
    Uses anchor, positive, and negative samples
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin between positive and negative pairs
        """
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
    def forward(self, 
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] - Binary labels
            
        Returns:
            loss: Scalar triplet loss
        """
        batch_size = embeddings.size(0)
        
        # Split into positive and negative samples
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            # Cannot compute triplet loss without both positive and negative samples
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]
        
        # Create triplets (anchor, positive, negative)
        num_pos = pos_embeddings.size(0)
        num_neg = neg_embeddings.size(0)
        
        # Use all positive samples as anchors
        anchors = pos_embeddings
        
        # For each anchor, find a positive and negative
        # Simple strategy: cycle through available samples
        positives = pos_embeddings[torch.arange(num_pos) % num_pos]
        negatives = neg_embeddings[torch.arange(num_pos) % num_neg]
        
        return self.triplet_loss(anchors, positives, negatives)


class FocalLoss(nn.Module):
    """
    Focal loss for handling imbalanced datasets
    Focuses learning on hard examples
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, 
                logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, 1] - Raw logits
            labels: [batch_size] - Binary labels
            
        Returns:
            loss: Scalar focal loss
        """
        # Convert to probabilities
        probs = torch.sigmoid(logits.squeeze(-1))
        labels = labels.float()
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy(probs, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class LossManager(nn.Module):
    """
    Manages multiple loss functions and combines them
    """
    
    def __init__(self,
                 loss_type: str = "bce",
                 bce_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 contrastive_margin: float = 1.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Args:
            loss_type: Primary loss type ("bce", "contrastive", "combined", "focal", "triplet")
            bce_weight: Weight for BCE loss in combinations
            contrastive_weight: Weight for contrastive loss in combinations
            focal_weight: Weight for focal loss in combinations
            contrastive_margin: Margin for contrastive loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.loss_type = LossType(loss_type) if loss_type in [lt.value for lt in LossType] else loss_type
        self.bce_weight = bce_weight
        self.contrastive_weight = contrastive_weight
        self.focal_weight = focal_weight
        
        # Initialize loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.triplet_loss = TripletLoss(margin=contrastive_margin)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    def compute_loss(self,
                    logits: torch.Tensor,
                    embeddings: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on specified loss type
        
        Args:
            logits: [batch_size, 1] - Classification logits
            embeddings: [batch_size, embedding_dim] - MLP embeddings
            labels: [batch_size] - Binary labels
            
        Returns:
            loss: Scalar loss value
        """
        labels = labels.float()  # Ensure float type
        
        if self.loss_type == LossType.BCE or self.loss_type == "bce":
            return self.bce_loss(logits.squeeze(-1), labels)
            
        elif self.loss_type == LossType.CONTRASTIVE or self.loss_type == "contrastive":
            return self.contrastive_loss(embeddings, labels)
            
        elif self.loss_type == LossType.COMBINED or self.loss_type == "combined":
            bce_loss = self.bce_loss(logits.squeeze(-1), labels)
            contrastive_loss = self.contrastive_loss(embeddings, labels)
            return self.bce_weight * bce_loss + self.contrastive_weight * contrastive_loss
            
        elif self.loss_type == "focal":
            return self.focal_loss(logits, labels)
            
        elif self.loss_type == "triplet":
            return self.triplet_loss(embeddings, labels)
            
        else:
            # Default to BCE
            return self.bce_loss(logits.squeeze(-1), labels)


# Removed unnecessary MLP config function - use direct parameter specification instead


# Removed unnecessary validation function - PyTorch will handle invalid loss configurations


def create_optimizer(model_parameters,
                    optimizer_type: str = "adam",
                    learning_rate: float = 0.001,
                    weight_decay: float = 0.01,
                    **kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer for model training
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: "adam", "adamw", "sgd", "rmsprop"
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer arguments
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model_parameters, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model_parameters, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model_parameters, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    elif optimizer_type.lower() == "rmsprop":
        return torch.optim.RMSprop(
            model_parameters, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        # Default to Adam
        return torch.optim.Adam(
            model_parameters, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            **kwargs
        )


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = "cosine",
                    num_epochs: int = 100,
                    **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: "cosine", "step", "exponential", "plateau", None
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        PyTorch scheduler or None
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None
        
    if scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            **kwargs
        )
    elif scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            **{k: v for k, v in kwargs.items() if k not in ['step_size', 'gamma']}
        )
    elif scheduler_type.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            **{k: v for k, v in kwargs.items() if k != 'gamma'}
        )
    elif scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=kwargs.get('patience', 10),
            factor=kwargs.get('factor', 0.5),
            **{k: v for k, v in kwargs.items() if k not in ['patience', 'factor']}
        )
    else:
        return None