"""
Word2Vec/FastText text encoder using gensim.
"""

import torch
import torch.nn as nn
import re
import numpy as np
import warnings
from typing import Dict, Optional, Union, List

from .base_text_encoder import BaseTextEncoder, TextAggregationStrategy


class Word2VecTextEncoder(BaseTextEncoder):
    """
    Text encoder using Word2Vec or FastText models from gensim.
    
    Lightweight alternative to transformer models for faster inference.
    """
    
    def __init__(self,
                 model_name: str = "word2vec-google-news-300",
                 aggregation_strategy: str = "separate_concat",
                 embedding_dim: int = 256,
                 num_text_fields: int = 2):
        """
        Initialize Word2Vec text encoder.
        
        Args:
            model_name: Word2Vec model name or path
                       - 'word2vec-google-news-300' (via gensim downloader)
                       - 'fasttext-wiki-news-300d-1m' (via gensim downloader)
                       - Path to local .bin, .vec, or .txt file
            aggregation_strategy: 'separate_concat', 'joint_encoding', or 'mean'
            embedding_dim: Output embedding dimension
            num_text_fields: Number of text fields to expect
        """
        super().__init__(aggregation_strategy, embedding_dim, num_text_fields)
        
        self.model_name = model_name
        
        # Initialize Word2Vec model
        self._initialize_model()
        
        # Calculate projection input dimension
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            projection_input_dim = self.word2vec_hidden_dim
        elif self.aggregation_strategy == TextAggregationStrategy.MEAN:
            projection_input_dim = self.word2vec_hidden_dim  # Average produces same dim as single field
        else:  # SEPARATE_CONCAT
            projection_input_dim = self.num_text_fields * self.word2vec_hidden_dim
        
        # Initialize projection layer
        self.projection = nn.Linear(projection_input_dim, embedding_dim)
    
    def _initialize_model(self):
        """Initialize Word2Vec model"""
        try:
            import gensim
            from gensim.models import KeyedVectors
            import gensim.downloader as api
            
            # Try to load using gensim downloader first
            try:
                self.word2vec_model = api.load(self.model_name)
                self.word2vec_hidden_dim = self.word2vec_model.vector_size
                return
            except Exception as e:
                pass
            
            # Handle specific hardcoded models
            if self.model_name == "word2vec-google-news-300":
                self.word2vec_model = KeyedVectors.load_word2vec_format(
                    'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM',
                    binary=True
                )
            elif self.model_name == "fasttext-wiki-news-300d-1m":
                self.word2vec_model = KeyedVectors.load_word2vec_format(
                    'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                    binary=False
                )
            else:
                self.word2vec_model = KeyedVectors.load_word2vec_format(self.model_name)
            
            self.word2vec_hidden_dim = self.word2vec_model.vector_size
            
        except ImportError:
            raise ImportError(
                "gensim library is required for Word2Vec. Install with: pip install gensim"
            )
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec model '{self.model_name}': {e}")
    
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
                    embedding = torch.zeros(1, self.word2vec_hidden_dim, device=device)
                
                field_embeddings.append(embedding)
            
            concatenated = torch.cat(field_embeddings, dim=1)
            return self.projection(concatenated)
    
    def _encode(self, text: str) -> torch.Tensor:
        """Encode text using Word2Vec averaging"""
        if not text or not text.strip():
            device = next(self.parameters()).device
            return torch.zeros(1, self.word2vec_hidden_dim, device=device)
        
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            
            if not words:
                device = next(self.parameters()).device
                return torch.zeros(1, self.word2vec_hidden_dim, device=device)
            
            word_embeddings = []
            for word in words:
                if word in self.word2vec_model:
                    word_embeddings.append(self.word2vec_model[word])
            
            if not word_embeddings:
                device = next(self.parameters()).device
                return torch.zeros(1, self.word2vec_hidden_dim, device=device)
            
            # Average word embeddings
            word_embeddings = np.array(word_embeddings)
            averaged_embedding = np.mean(word_embeddings, axis=0)
            
            device = next(self.parameters()).device
            return torch.tensor(averaged_embedding, dtype=torch.float32, device=device).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Error encoding text: {e}")
            device = next(self.parameters()).device
            return torch.zeros(1, self.word2vec_hidden_dim, device=device)


def create_word2vec_encoder(model_name: str = "word2vec-google-news-300",
                           aggregation_strategy: str = "separate_concat",
                           embedding_dim: int = 256,
                           num_text_fields: int = 2) -> Word2VecTextEncoder:
    """
    Factory function to create a Word2VecTextEncoder.
    
    Args:
        model_name: Word2Vec model name or path
        aggregation_strategy: 'separate_concat' or 'joint_encoding'
        embedding_dim: Output embedding dimension
        num_text_fields: Number of text fields to expect
        
    Returns:
        Configured Word2VecTextEncoder instance
    """
    return Word2VecTextEncoder(
        model_name=model_name,
        aggregation_strategy=aggregation_strategy,
        embedding_dim=embedding_dim,
        num_text_fields=num_text_fields
    )
