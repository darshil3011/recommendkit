import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
import warnings
import re
import numpy as np


class TextAggregationStrategy(Enum):
    """Supported aggregation strategies for multiple text fields"""
    SEPARATE_CONCAT = "separate_concat"  # Encode each field separately, then concatenate
    JOINT_ENCODING = "joint_encoding"    # Concatenate raw text, then encode jointly


class TextEncoder(nn.Module):
    """
    Text encoder that handles multiple text fields with different aggregation strategies
    
    Args:
        aggregation_strategy: How to combine multiple text fields
        model_name: BERT model to use (e.g., 'bert-base-uncased', 'distilbert-base-uncased')
        max_length: Maximum input sequence length
        embedding_dim: Output embedding dimension
        freeze_bert: Whether to freeze BERT weights during training
        pooling_strategy: How to pool BERT outputs ('cls', 'mean', 'max')
    """
    
    def __init__(self,
                 aggregation_strategy: Union[str, TextAggregationStrategy] = TextAggregationStrategy.SEPARATE_CONCAT,
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 embedding_dim: int = 256,
                 freeze_bert: bool = False,
                 pooling_strategy: str = "cls",
                 num_text_fields: int = 2):
        
        super().__init__()
        
        # Convert string arguments to enums
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = TextAggregationStrategy(aggregation_strategy.lower())
            
        self.aggregation_strategy = aggregation_strategy
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy.lower()
        self.num_text_fields = max(num_text_fields, 1)  # At least 1 field
        
        # Validate pooling strategy
        if self.pooling_strategy not in ['cls', 'mean', 'max']:
            raise ValueError("pooling_strategy must be one of: 'cls', 'mean', 'max'")
        
        # Initialize BERT model and tokenizer
        self._initialize_bert()
        
        # Get hidden dimension (from BERT or Word2Vec)
        if hasattr(self, 'bert') and self.bert is not None:
            self.bert_hidden_dim = self.bert.config.hidden_size
            # Freeze BERT if requested
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            # Word2Vec case - bert_hidden_dim is set in _initialize_word2vec
            pass
        
        # Calculate projection layer input dimension based on aggregation strategy and field count
        if aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            # Single BERT encoding
            projection_input_dim = self.bert_hidden_dim
        else:  # SEPARATE_CONCAT
            # Use field count - NO MORE LAZY INITIALIZATION!
            projection_input_dim = self.num_text_fields * self.bert_hidden_dim
        
        # Initialize projection layer immediately
        self.projection = nn.Linear(projection_input_dim, embedding_dim)
        self._projection_initialized = True
        
        # Special tokens for joint encoding
        self.field_separator = " [SEP] "
        self.field_prefix_template = "[{}]: "
        
        # For handling missing text
        self.register_buffer('default_embedding', torch.zeros(embedding_dim))
    
    def _initialize_bert(self):
        """Initialize BERT model and tokenizer, or Word2Vec model"""
        # Initialize attributes to None first
        self.bert = None
        self.tokenizer = None
        self.word2vec_model = None
        self.is_word2vec = False
        
        # Check if this is a Word2Vec model
        if self._is_word2vec_model(self.model_name):
            self._initialize_word2vec()
        else:
            self._initialize_transformer_model()
    
    def _is_word2vec_model(self, model_name: str) -> bool:
        """Check if the model name indicates a Word2Vec model"""
        word2vec_indicators = [
            'word2vec', 'fasttext', 'glove', 'google-news', 'wikipedia'
        ]
        return any(indicator in model_name.lower() for indicator in word2vec_indicators)
    
    def _initialize_word2vec(self):
        """Initialize Word2Vec model"""
        try:
            import gensim
            from gensim.models import KeyedVectors
            import gensim.downloader as api
            
            print(f"Loading Word2Vec model: {self.model_name}")
            
            # Try to load using gensim downloader first
            try:
                self.word2vec_model = api.load(self.model_name)
                self.bert_hidden_dim = self.word2vec_model.vector_size
                self.is_word2vec = True
                print(f"Successfully loaded Word2Vec model with {self.bert_hidden_dim} dimensions")
                return
            except Exception as e:
                print(f"Gensim downloader failed: {e}")
                # Fall back to hardcoded models
                pass
            
            # Handle specific hardcoded models
            if self.model_name == "word2vec-google-news-300":
                # Download and load Google News Word2Vec
                self.word2vec_model = KeyedVectors.load_word2vec_format(
                    'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM', 
                    binary=True
                )
            elif self.model_name == "fasttext-wiki-news-300d-1m":
                # FastText model
                self.word2vec_model = KeyedVectors.load_word2vec_format(
                    'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                    binary=False
                )
            else:
                # Try to load as a local file or URL
                self.word2vec_model = KeyedVectors.load_word2vec_format(self.model_name)
            
            # Get embedding dimension from the model
            self.bert_hidden_dim = self.word2vec_model.vector_size
            self.is_word2vec = True
            print(f"Successfully loaded Word2Vec model with {self.bert_hidden_dim} dimensions")
            
        except ImportError:
            raise ImportError(
                "gensim library is required for Word2Vec. Install with: pip install gensim"
            )
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec model '{self.model_name}': {e}")
    
    def _initialize_transformer_model(self):
        """Initialize transformer model (BERT, etc.)"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            print(f"Loading transformer model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.bert = AutoModel.from_pretrained(self.model_name)
            self.bert_hidden_dim = self.bert.config.hidden_size
            self.is_word2vec = False
            print(f"Successfully loaded transformer model with {self.bert_hidden_dim} dimensions")
            
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            raise ValueError(f"Failed to load transformer model '{self.model_name}': {e}")
    
    def _initialize_projection_layer(self, input_dim: int):
        """Initialize projection layer with known input dimension"""
        if not self._projection_initialized:
            self.projection = nn.Linear(input_dim, self.embedding_dim)
            self._projection_initialized = True
    
    def initialize_projection_for_loading(self, num_text_fields: int):
        """
        Initialize projection layer for loading state dict
        
        Args:
            num_text_fields: Number of text fields that will be processed
        """
        if self._projection_initialized:
            return  # Already initialized
            
        # Calculate the expected input dimension based on aggregation strategy
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            input_dim = self.bert_hidden_dim
        else:  # SEPARATE_CONCAT
            input_dim = num_text_fields * self.bert_hidden_dim
            
        self._initialize_projection_layer(input_dim)
    
    def _clean_text(self, text: Union[str, List[str], None]) -> Union[str, List[str]]:
        """Clean and preprocess text, handling both single strings and lists"""
        if text is None:
            return ""
        
        if isinstance(text, list):
            # Handle batched text data
            cleaned_texts = []
            for t in text:
                if t is None:
                    cleaned_texts.append("")
                else:
                    # Remove excessive whitespace
                    cleaned = re.sub(r'\s+', ' ', t.strip())
                    cleaned_texts.append(cleaned if cleaned else "")
            return cleaned_texts
        
        # Handle single text string
        if isinstance(text, str):
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            # Handle empty text
            if not text:
                return ""
            return text
        
        # Fallback for unexpected types
        return str(text) if text else ""
    
    def _encode_single_text(self, text: str, field_name: str = None) -> torch.Tensor:
        """
        Encode a single text string using BERT or Word2Vec
        
        Args:
            text: Input text string
            field_name: Name of the text field (for debugging)
            
        Returns:
            Text embedding tensor of shape (1, hidden_dim)
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text:
            # Return zero embedding for empty text
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, self.bert_hidden_dim, device=device)
        
        try:
            if self.is_word2vec:
                return self._encode_with_word2vec(cleaned_text, field_name)
            else:
                return self._encode_with_bert(cleaned_text, field_name)
                
        except Exception as e:
            warnings.warn(f"Error encoding text for field '{field_name}': {e}")
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, self.bert_hidden_dim, device=device)
    
    def _encode_with_word2vec(self, text: str, field_name: str = None) -> torch.Tensor:
        """Encode text using Word2Vec averaging"""
        # Simple tokenization (split on whitespace and punctuation)
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, self.bert_hidden_dim, device=device)
        
        # Get word embeddings
        word_embeddings = []
        for word in words:
            if word in self.word2vec_model:
                word_embeddings.append(self.word2vec_model[word])
        
        if not word_embeddings:
            # No words found in vocabulary
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, self.bert_hidden_dim, device=device)
        
        # Average word embeddings
        word_embeddings = np.array(word_embeddings)
        averaged_embedding = np.mean(word_embeddings, axis=0)
        
        # Convert to torch tensor
        device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
        return torch.tensor(averaged_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    
    def _encode_with_bert(self, text: str, field_name: str = None) -> torch.Tensor:
        """Encode text using BERT"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT outputs
        with torch.no_grad() if hasattr(self, 'freeze_bert') and self.freeze_bert else torch.enable_grad():
            outputs = self.bert(**inputs)
            
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence length
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask
            pooled_output = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling_strategy == "max":
            # Max pooling over sequence length
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask
            # Set padded positions to large negative value before max pooling
            masked_embeddings = masked_embeddings.masked_fill(attention_mask == 0, -1e9)
            pooled_output = masked_embeddings.max(dim=1)[0]
            
        return pooled_output
    
    def _joint_encoding_forward(self, text_dict: Dict[str, Optional[str]]) -> torch.Tensor:
        """
        Joint encoding: concatenate all text fields and encode together
        
        Args:
            text_dict: Dictionary mapping field names to text content
            
        Returns:
            Combined embedding tensor of shape (1, embedding_dim)
        """
        # Combine all text fields into one string
        text_parts = []
        
        for field_name, text_content in text_dict.items():
            cleaned_text = self._clean_text(text_content)
            if cleaned_text:
                # Add field prefix for better understanding
                prefixed_text = self.field_prefix_template.format(field_name) + cleaned_text
                text_parts.append(prefixed_text)
        
        if not text_parts:
            # No text content, return default embedding
            return self.default_embedding.unsqueeze(0)
        
        # Join all text parts
        combined_text = self.field_separator.join(text_parts)
        
        # Encode the combined text
        bert_embedding = self._encode_single_text(combined_text, "combined")
        
        # Initialize projection layer if needed
        if not self._projection_initialized:
            self._initialize_projection_layer(self.bert_hidden_dim)
        
        # Project to desired embedding dimension
        final_embedding = self.projection(bert_embedding)
        return final_embedding
    
    def _separate_concat_forward(self, text_dict: Dict[str, Optional[str]]) -> torch.Tensor:
        """
        Separate encoding: encode each field separately, then concatenate
        Pads or truncates to match expected num_text_fields
        
        Args:
            text_dict: Dictionary mapping field names to text content
            
        Returns:
            Combined embedding tensor of shape (1, embedding_dim)
        """
        field_embeddings = []
        available_fields = list(text_dict.keys())
        
        # Process up to num_text_fields (pad with zeros if needed, truncate if too many)
        for i in range(self.num_text_fields):
            if i < len(available_fields):
                field_name = available_fields[i]
                text_content = text_dict[field_name]
                if text_content:
                    bert_embedding = self._encode_single_text(text_content, field_name)
                else:
                    # Empty text -> zero embedding
                    bert_embedding = torch.zeros(1, self.bert_hidden_dim, device=next(self.parameters()).device)
            else:
                # Missing field -> zero embedding
                bert_embedding = torch.zeros(1, self.bert_hidden_dim, device=next(self.parameters()).device)
            
            field_embeddings.append(bert_embedding)
        
        # Concatenate all field embeddings (always num_text_fields * bert_hidden_dim)
        concatenated_embeddings = torch.cat(field_embeddings, dim=1)
        
        # Project to desired embedding dimension (projection already initialized!)
        final_embedding = self.projection(concatenated_embeddings)
        return final_embedding
    
    def forward(self, text_dict: Dict[str, Union[str, List[str], None]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text encoding that handles both single and batched inputs
        
        Args:
            text_dict: Dictionary mapping text field names to content
                      Single sample: {'bio': 'I am a college guy', 'summary': 'some text'}
                      Batched: {'bio': ['text1', 'text2'], 'summary': ['text3', 'text4']}
        
        Returns:
            Dictionary with text features: {"text_features": torch.Tensor}
            Shape: (batch_size, embedding_dim)
        """
        if not text_dict:
            return {"text_features": self.default_embedding.unsqueeze(0)}
        
        # Determine if this is batched input
        first_value = next(iter(text_dict.values()))
        is_batched = isinstance(first_value, list)
        
        if is_batched:
            return self._forward_batched(text_dict)
        else:
            # Single sample - convert to batch format for consistency
            result = self._forward_single(text_dict)
            return {"text_features": result}
    
    def _forward_single(self, text_dict: Dict[str, Optional[str]]) -> torch.Tensor:
        """Process a single sample"""
        if not text_dict:
            return self.default_embedding.unsqueeze(0)
        
        # Apply aggregation strategy
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            return self._joint_encoding_forward(text_dict)
        elif self.aggregation_strategy == TextAggregationStrategy.SEPARATE_CONCAT:
            return self._separate_concat_forward(text_dict)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
    
    def _forward_batched(self, text_dict: Dict[str, List[Optional[str]]]) -> Dict[str, torch.Tensor]:
        """Process a batch of samples"""
        # Get batch size from first non-empty field
        batch_size = 0
        for field_values in text_dict.values():
            if isinstance(field_values, list) and len(field_values) > 0:
                batch_size = len(field_values)
                break
        
        if batch_size == 0:
            return {"text_features": self.default_embedding.unsqueeze(0)}
        
        # Process each sample in the batch
        batch_embeddings = []
        for i in range(batch_size):
            # Extract single sample from batch
            sample_dict = {}
            for field_name, field_values in text_dict.items():
                if isinstance(field_values, list) and i < len(field_values):
                    sample_dict[field_name] = field_values[i]
                else:
                    sample_dict[field_name] = None
            
            # Process single sample
            sample_embedding = self._forward_single(sample_dict)
            batch_embeddings.append(sample_embedding)
        
        # Stack into batch
        batch_result = torch.cat(batch_embeddings, dim=0)  # (batch_size, embedding_dim)
        return {"text_features": batch_result}
    
    def get_output_dim(self, num_text_fields: int = None) -> int:
        """
        Get the output embedding dimension
        
        Args:
            num_text_fields: Number of text fields (only needed for debugging/planning)
            
        Returns:
            Output embedding dimension
        """
        return self.embedding_dim
    
    def get_projection_input_dim(self, num_text_fields: int) -> int:
        """
        Get the expected input dimension for the projection layer
        
        Args:
            num_text_fields: Number of text fields
            
        Returns:
            Expected projection layer input dimension
        """
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            return self.bert_hidden_dim
        else:  # SEPARATE_CONCAT
            return num_text_fields * self.bert_hidden_dim
    
    def set_default_embedding(self, embedding: torch.Tensor):
        """Set the default embedding for missing/empty text"""
        if embedding.size(0) != self.embedding_dim:
            raise ValueError(f"Default embedding dimension {embedding.size(0)} doesn't match expected {self.embedding_dim}")
        self.default_embedding = embedding
    
    def get_bert_hidden_dim(self) -> int:
        """Get BERT model's hidden dimension"""
        return self.bert_hidden_dim
    
    def estimate_max_tokens(self, text_dict: Dict[str, str]) -> int:
        """
        Estimate the number of tokens for input validation
        
        Args:
            text_dict: Dictionary of text fields
            
        Returns:
            Estimated token count
        """
        if self.aggregation_strategy == TextAggregationStrategy.JOINT_ENCODING:
            # Combine all text and estimate tokens
            combined_text = ""
            for field_name, text_content in text_dict.items():
                if text_content:
                    combined_text += f"[{field_name}]: {text_content} [SEP] "
            
            # Rough estimation: ~1.3 tokens per word
            word_count = len(combined_text.split())
            return int(word_count * 1.3)
        else:
            # For separate encoding, return max tokens among all fields
            max_tokens = 0
            for text_content in text_dict.values():
                if text_content:
                    word_count = len(text_content.split())
                    tokens = int(word_count * 1.3)
                    max_tokens = max(max_tokens, tokens)
            return max_tokens


# Factory function
def create_text_encoder(aggregation_strategy: str = "separate_concat",
                       model_name: str = "bert-base-uncased",
                       max_length: int = 512,
                       embedding_dim: int = 256,
                       freeze_bert: bool = False,
                       pooling_strategy: str = "cls",
                       num_text_fields: int = 2) -> TextEncoder:
    """
    Factory function to create a TextEncoder
    
    Args:
        aggregation_strategy: 'separate_concat' or 'joint_encoding'
        model_name: Model name. Supports:
                   - BERT models: 'bert-base-uncased', 'distilbert-base-uncased'
                   - Word2Vec models: 'word2vec-google-news-300', 'fasttext-wiki-news-300d-1m'
                   - Custom Word2Vec: path to .bin, .vec, or .txt file
        max_length: Maximum input sequence length (ignored for Word2Vec)
        embedding_dim: Output embedding dimension
        freeze_bert: Whether to freeze BERT parameters (ignored for Word2Vec)
        pooling_strategy: How to pool BERT outputs ('cls', 'mean', 'max') - ignored for Word2Vec
        num_text_fields: Number of text fields to expect (eliminates lazy initialization)
        
    Returns:
        Configured TextEncoder instance
        
    Examples:
        # BERT encoder
        encoder = create_text_encoder(model_name="bert-base-uncased")
        
        # Word2Vec encoder (lightweight, fast)
        encoder = create_text_encoder(model_name="word2vec-google-news-300")
        
        # FastText encoder
        encoder = create_text_encoder(model_name="fasttext-wiki-news-300d-1m")
    """
    return TextEncoder(
        aggregation_strategy=aggregation_strategy,
        model_name=model_name,
        max_length=max_length,
        embedding_dim=embedding_dim,
        freeze_bert=freeze_bert,
        pooling_strategy=pooling_strategy,
        num_text_fields=num_text_fields
    )


# Removed unnecessary validation utility function - BERT tokenizer handles validation internally


# Example usage and testing
if __name__ == "__main__":
    print("Testing Text Encoders...")
    
    # Test configurations
    configs = [
        {
            "aggregation_strategy": "separate_concat",
            "model_name": "distilbert-base-uncased",  # Smaller BERT model for testing
            "max_length": 256,
            "embedding_dim": 128,
            "freeze_bert": True,
            "pooling_strategy": "cls"
        },
        {
            "aggregation_strategy": "joint_encoding", 
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "embedding_dim": 256,
            "freeze_bert": False,
            "pooling_strategy": "mean"
        },
        {
            "aggregation_strategy": "separate_concat",
            "model_name": "word2vec-google-news-300",  # Word2Vec model (lightweight)
            "max_length": 512,  # Ignored for Word2Vec
            "embedding_dim": 128,
            "freeze_bert": False,  # Ignored for Word2Vec
            "pooling_strategy": "cls"  # Ignored for Word2Vec
        }
    ]
    
    # Sample text data
    sample_text_data = {
        'bio': 'I am a college guy in Texas studying computer science',
        'summary': 'Passionate about machine learning and AI',
        'description': 'Looking for internship opportunities in tech',
        'interests': None  # Missing field
    }
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}: {config['aggregation_strategy']} ---")
        
        try:
            # Create encoder
            encoder = create_text_encoder(**config)
            
            # Test encoder properties
            print(f"Model type: {'Word2Vec' if encoder.is_word2vec else 'Transformer'}")
            print(f"Hidden dim: {encoder.bert_hidden_dim}")
            
            # Test forward pass
            with torch.no_grad():
                output = encoder(sample_text_data)
                print(f"Output shape: {output.shape}")
                print(f"Expected output dim: {encoder.get_output_dim()}")
                
                # Show projection input dimension
                num_fields = len([v for v in sample_text_data.values() if v is not None])
                proj_input_dim = encoder.get_projection_input_dim(num_fields)
                print(f"Projection input dim for {num_fields} fields: {proj_input_dim}")
                
        except Exception as e:
            print(f"Error with configuration {config}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nText encoder tests completed!")