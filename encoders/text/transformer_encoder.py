"""
Transformer-based text encoder (BERT, etc.) using HuggingFace transformers.
"""

import torch
import torch.nn as nn
import warnings
import re
from typing import Dict, Optional, Union, List

from .base_text_encoder import BaseTextEncoder, TextAggregationStrategy


class TransformerTextEncoder(BaseTextEncoder):
    """
    Text encoder using transformer models (BERT, DistilBERT, etc.) from HuggingFace.
    
    Supports any model from HuggingFace transformers library via AutoModel.
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 aggregation_strategy: str = "separate_concat",
                 max_length: int = 512,
                 embedding_dim: int = 256,
                 freeze_bert: bool = False,
                 pooling_strategy: str = "cls",
                 num_text_fields: int = 2):
        """
        Initialize transformer text encoder.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased', 'distilbert-base-uncased')
            aggregation_strategy: 'separate_concat', 'joint_encoding', or 'mean'
            max_length: Maximum input sequence length
            embedding_dim: Output embedding dimension
            freeze_bert: Whether to freeze transformer parameters
            pooling_strategy: How to pool outputs ('cls', 'mean', 'max')
            num_text_fields: Number of text fields to expect
        """
        super().__init__(aggregation_strategy, embedding_dim, num_text_fields)
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy.lower()
        self.freeze_bert = freeze_bert
        
        # Validate pooling strategy
        if self.pooling_strategy not in ['cls', 'mean', 'max']:
            raise ValueError("pooling_strategy must be one of: 'cls', 'mean', 'max'")
        
        # Initialize transformer model
        self._initialize_model()
        
        # Calculate projection input dimension
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            projection_input_dim = self.bert_hidden_dim
        elif self.aggregation_strategy == TextAggregationStrategy.MEAN:
            projection_input_dim = self.bert_hidden_dim  # Average produces same dim as single field
        else:  # SEPARATE_CONCAT
            projection_input_dim = self.num_text_fields * self.bert_hidden_dim
        
        # Initialize projection layer
        self.projection = nn.Linear(projection_input_dim, embedding_dim)
    
    def _initialize_model(self):
        """Initialize transformer model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.bert = AutoModel.from_pretrained(self.model_name)
            self.bert_hidden_dim = self.bert.config.hidden_size
            
            # Freeze if requested
            if self.freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            raise ValueError(f"Failed to load transformer model '{self.model_name}': {e}")
    
    def forward(self, text_dict: Dict[str, Union[str, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text encoding.
        
        Args:
            text_dict: Dictionary mapping text field names to content
                      Single: {'bio': 'text', 'summary': 'text'}
                      Batched: {'bio': ['text1', 'text2'], 'summary': ['text3', 'text4']}
        
        Returns:
            Dictionary with text features: {"text_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        if not text_dict:
            return {"text_features": self.default_embedding.unsqueeze(0)}
        
        # Check if batched
        first_value = next(iter(text_dict.values()))
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            # Process batch
            batch_size = len(first_value) if first_value else 0
            if batch_size == 0:
                return {"text_features": self.default_embedding.unsqueeze(0)}
            
            batch_embeddings = []
            for i in range(batch_size):
                sample = {k: (v[i] if isinstance(v, list) and i < len(v) else None) 
                         for k, v in text_dict.items()}
                batch_embeddings.append(self._process_sample(sample))
            
            return {"text_features": torch.cat(batch_embeddings, dim=0)}
        else:
            # Process single sample
            result = self._process_sample(text_dict)
            return {"text_features": result}
    
    def _process_sample(self, text_dict: Dict[str, Optional[str]]) -> torch.Tensor:
        """Process a single sample"""
        if not text_dict:
            return self.default_embedding.unsqueeze(0)
        
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            # Combine all fields and encode together
            text_parts = []
            for field_name, text_content in text_dict.items():
                if text_content:
                    cleaned = re.sub(r'\s+', ' ', str(text_content).strip())
                    if cleaned:
                        text_parts.append(f"[{field_name}]: {cleaned}")
            
            if not text_parts:
                return self.default_embedding.unsqueeze(0)
            
            combined = " [SEP] ".join(text_parts)
            embedding = self._encode(combined)
            return self.projection(embedding)
        
        elif self.aggregation_strategy == TextAggregationStrategy.MEAN:
            # Encode each field separately, then average
            field_embeddings = []
            for field_name, text_content in text_dict.items():
                if text_content:
                    cleaned = re.sub(r'\s+', ' ', str(text_content).strip())
                    if cleaned:
                        embedding = self._encode(cleaned)
                        field_embeddings.append(embedding)
            
            if not field_embeddings:
                return self.default_embedding.unsqueeze(0)
            
            # Average all field embeddings
            stacked = torch.cat(field_embeddings, dim=0)  # (num_fields, hidden_dim)
            averaged = stacked.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            return self.projection(averaged)
        
        else:  # SEPARATE_CONCAT
            # Encode each field separately, then concatenate
            field_embeddings = []
            fields = list(text_dict.keys())[:self.num_text_fields]
            
            for i in range(self.num_text_fields):
                if i < len(fields) and text_dict[fields[i]]:
                    cleaned = re.sub(r'\s+', ' ', str(text_dict[fields[i]]).strip())
                    embedding = self._encode(cleaned) if cleaned else None
                else:
                    embedding = None
                
                if embedding is None:
                    device = next(self.parameters()).device
                    embedding = torch.zeros(1, self.bert_hidden_dim, device=device)
                
                field_embeddings.append(embedding)
            
            concatenated = torch.cat(field_embeddings, dim=1)
            return self.projection(concatenated)
    
    def _encode(self, text: str) -> torch.Tensor:
        """Encode text using transformer"""
        if not text or not text.strip():
            device = next(self.parameters()).device
            return torch.zeros(1, self.bert_hidden_dim, device=device)
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            device = next(self.bert.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad() if self.freeze_bert else torch.enable_grad():
                outputs = self.bert(**inputs)
            
            # Apply pooling
            if self.pooling_strategy == "cls":
                return outputs.last_hidden_state[:, 0, :]
            elif self.pooling_strategy == "mean":
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                return masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            elif self.pooling_strategy == "max":
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                masked_embeddings = masked_embeddings.masked_fill(attention_mask == 0, -1e9)
                return masked_embeddings.max(dim=1)[0]
            
        except Exception as e:
            warnings.warn(f"Error encoding text: {e}")
            device = next(self.parameters()).device
            return torch.zeros(1, self.bert_hidden_dim, device=device)


def create_transformer_encoder(model_name: str = "bert-base-uncased",
                               aggregation_strategy: str = "separate_concat",
                               max_length: int = 512,
                               embedding_dim: int = 256,
                               freeze_bert: bool = False,
                               pooling_strategy: str = "cls",
                               num_text_fields: int = 2) -> TransformerTextEncoder:
    """
    Factory function to create a TransformerTextEncoder.
    
    Args:
        model_name: HuggingFace model name
        aggregation_strategy: 'separate_concat' or 'joint_encoding'
        max_length: Maximum input sequence length
        embedding_dim: Output embedding dimension
        freeze_bert: Whether to freeze transformer parameters
        pooling_strategy: How to pool outputs ('cls', 'mean', 'max')
        num_text_fields: Number of text fields to expect
        
    Returns:
        Configured TransformerTextEncoder instance
    """
    return TransformerTextEncoder(
        model_name=model_name,
        aggregation_strategy=aggregation_strategy,
        max_length=max_length,
        embedding_dim=embedding_dim,
        freeze_bert=freeze_bert,
        pooling_strategy=pooling_strategy,
        num_text_fields=num_text_fields
    )
