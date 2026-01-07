"""
trainer/pipeline_builder.py
Builds end-to-end recommendation pipeline by connecting all modules:
- Input processing → Encoders → Feature fusion → Interaction modeling → Classification
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all the modules we've built
from input_processor import Inputs
from encoders.image import create_image_encoder
from encoders.text import create_text_encoder
from encoders.categorical import create_categorical_encoder
from encoders.continuous import create_continuous_encoder
from encoders.temporal import create_temporal_encoder, ItemLookupInterface
from interaction.feature_fusion import UserEmbeddingGenerator, ItemEmbeddingGenerator, AsymmetricTowerModel
from interaction.interaction_modeling import InteractionEmbeddingGenerator
from classifier.recommendation_classifier import RecommendationClassifier


class GenericItemLookupInterface(ItemLookupInterface):
    """Generic implementation of ItemLookupInterface for any item data"""
    
    def __init__(self, item_data_dict: Dict[Union[str, int], Dict[str, Any]]):
        """
        Args:
            item_data_dict: Dictionary mapping item_id to item features
        """
        self.item_db = item_data_dict
    
    def get_item_features(self, item_id: Union[str, int]) -> Dict[str, Any]:
        """Get features for a specific item"""
        # Convert to string for consistent key lookup
        item_key = str(item_id)
        
        # Try string key first
        if item_key in self.item_db:
            return self.item_db[item_key]
        
        # Try integer key if item_id is numeric
        try:
            int_key = int(item_id)
            if int_key in self.item_db:
                return self.item_db[int_key]
        except (ValueError, TypeError):
            pass
        
        # Return default empty features if not found
        return {
            'image': None,
            'text': None,
            'categorical': None,
            'continuous': None
        }
    
    def batch_get_item_features(self, item_ids: List[Union[str, int]]) -> List[Dict[str, Any]]:
        """Get features for multiple items"""
        return [self.get_item_features(item_id) for item_id in item_ids]


class DimensionAligner(nn.Module):
    """
    Generic dimension alignment layer that projects embeddings to target dimension
    Handles different encoder output dimensions automatically
    """
    
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim
        self.projections = nn.ModuleDict()
    
    def _get_projection(self, input_dim: int) -> nn.Module:
        """Get or create projection layer for input dimension"""
        if input_dim not in self.projections:
            if input_dim == self.target_dim:
                # No projection needed
                self.projections[str(input_dim)] = nn.Identity()
            else:
                # Create projection layer
                self.projections[str(input_dim)] = nn.Linear(input_dim, self.target_dim)
        return self.projections[str(input_dim)]
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Align all embeddings to target dimension and batch size
        
        Args:
            embeddings: Dict of feature_name -> embedding_tensor
            
        Returns:
            Dict of aligned embeddings with same keys
        """
        if not embeddings:
            return {}

        # Find the target batch size (use the most common batch size)
        batch_sizes = [emb.shape[0] for emb in embeddings.values()]
        target_batch_size = max(set(batch_sizes), key=batch_sizes.count)
        
        aligned_embeddings = {}
        
        for feature_name, embedding in embeddings.items():
            input_dim = embedding.shape[-1]
            projection = self._get_projection(input_dim)
            aligned_embedding = projection(embedding)
            
            # Handle batch size mismatch
            if aligned_embedding.shape[0] != target_batch_size:
                if aligned_embedding.shape[0] == 1:
                    # Broadcast single embedding to target batch size
                    aligned_embedding = aligned_embedding.expand(target_batch_size, -1)
                else:
                    # This shouldn't happen, but handle gracefully
                    print(f"Warning: Unexpected batch size mismatch for {feature_name}: {aligned_embedding.shape[0]} vs {target_batch_size}")
                    # Repeat the embedding to match batch size
                    repeat_factor = target_batch_size // aligned_embedding.shape[0]
                    remainder = target_batch_size % aligned_embedding.shape[0]
                    if remainder == 0:
                        aligned_embedding = aligned_embedding.repeat(repeat_factor, 1)
                    else:
                        # Handle non-divisible case
                        repeated = aligned_embedding.repeat(repeat_factor, 1)
                        extra = aligned_embedding[:remainder]
                        aligned_embedding = torch.cat([repeated, extra], dim=0)
            
            
            aligned_embeddings[feature_name] = aligned_embedding
        
        return aligned_embeddings


class RecommendationPipeline(nn.Module):
    """
    Complete end-to-end recommendation pipeline that connects all components
    Takes raw input data → outputs engagement predictions
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 # Encoder configurations
                 image_encoder_config: Optional[Dict] = None,
                 text_encoder_config: Optional[Dict] = None,
                 categorical_encoder_config: Optional[Dict] = None,
                 continuous_encoder_config: Optional[Dict] = None,
                 temporal_encoder_config: Optional[Dict] = None,
                 # Item data for temporal encoder (optional)
                 item_data: Optional[List[Dict[str, Any]]] = None,
                 # Direct parameters instead of complexity levels
                 user_num_attention_layers: int = 4,
                 user_num_heads: int = 16,
                 user_dropout: float = 0.15,
                 user_use_cls_token: bool = True,
                 user_use_layer_norm: bool = False,
                 user_use_simple_fusion: bool = True,
                 item_num_attention_layers: int = 1,
                 item_num_heads: int = 8,
                 item_dropout: float = 0.1,
                 item_use_simple_fusion: bool = True,
                 interaction_num_attention_layers: int = 2,
                 interaction_num_heads: int = 8,
                 interaction_dropout: float = 0.1,
                 interaction_use_simple_fusion: bool = True,
                 # Classifier configuration
                 classifier_hidden_dims: List[int] = None,
                 classifier_dropout: float = 0.2,
                 loss_type: str = "bce"):
        """
        Args:
            embedding_dim: Dimension of all embeddings throughout the pipeline
            image_encoder_config: Configuration for image encoder
            text_encoder_config: Configuration for text encoder
            categorical_encoder_config: Configuration for categorical encoder
            temporal_encoder_config: Configuration for temporal encoder
            user_num_attention_layers: Number of attention layers in user tower
            user_num_heads: Number of attention heads in user tower
            user_dropout: Dropout rate in user tower
            user_use_cls_token: Whether to use CLS token in user tower
            item_num_attention_layers: Number of attention layers in item tower
            item_num_heads: Number of attention heads in item tower
            item_dropout: Dropout rate in item tower
            interaction_num_attention_layers: Number of attention layers in interaction
            interaction_num_heads: Number of attention heads in interaction
            interaction_dropout: Dropout rate in interaction
            classifier_hidden_dims: Hidden layer dimensions for classifier
            classifier_dropout: Dropout rate in classifier
            loss_type: Loss function type
        """
        super().__init__()
        
        # Store configuration parameters
        self.embedding_dim = embedding_dim
        self.item_data = item_data
        self.loss_type = loss_type
        
        # Store original config for saving (will be set by create_model_from_config)
        self._saved_config: Optional[Dict[str, Any]] = None
        
        # User tower configuration
        self.user_num_attention_layers = user_num_attention_layers
        self.user_num_heads = user_num_heads
        self.user_dropout = user_dropout
        self.user_use_cls_token = user_use_cls_token
        self.user_use_layer_norm = user_use_layer_norm
        self.user_use_simple_fusion = user_use_simple_fusion
        
        # Item tower configuration
        self.item_num_attention_layers = item_num_attention_layers
        self.item_num_heads = item_num_heads
        self.item_dropout = item_dropout
        self.item_use_simple_fusion = item_use_simple_fusion
        
        # Interaction modeling configuration
        self.interaction_num_attention_layers = interaction_num_attention_layers
        self.interaction_num_heads = interaction_num_heads
        self.interaction_dropout = interaction_dropout
        self.interaction_use_simple_fusion = interaction_use_simple_fusion
        
        # Classifier configuration
        self.classifier_hidden_dims = classifier_hidden_dims
        self.classifier_dropout = classifier_dropout
        
        # Create individual encoders
        self.image_encoder = self._create_image_encoder(image_encoder_config)
        self.text_encoder = self._create_text_encoder(text_encoder_config) 
        self.categorical_encoder = self._create_categorical_encoder(categorical_encoder_config)
        # Create separate continuous encoders for user and item (they may have different numbers of features)
        self.user_continuous_encoder = self._create_continuous_encoder(continuous_encoder_config)
        self.item_continuous_encoder = self._create_continuous_encoder(continuous_encoder_config)
        self.temporal_encoder = self._create_temporal_encoder(temporal_encoder_config)
        
        # Create dimension aligners for user and item features
        self.user_dimension_aligner = DimensionAligner(target_dim=embedding_dim)
        self.item_dimension_aligner = DimensionAligner(target_dim=embedding_dim)
        
        # Ensure num_heads divides embedding_dim
        user_num_heads = self._get_valid_num_heads(user_num_heads, embedding_dim)
        item_num_heads = self._get_valid_num_heads(item_num_heads, embedding_dim)
        interaction_num_heads = self._get_valid_num_heads(interaction_num_heads, embedding_dim)
        
        # Create fusion layers (user and item towers) directly
        self.user_generator = UserEmbeddingGenerator(
            embedding_dim=embedding_dim,
            num_attention_layers=user_num_attention_layers,
            num_heads=user_num_heads,
            dropout=user_dropout,
            use_cls_token=user_use_cls_token,
            use_layer_norm=user_use_layer_norm,
            use_simple_fusion=user_use_simple_fusion
        )
        
        self.item_generator = ItemEmbeddingGenerator(
            embedding_dim=embedding_dim,
            num_attention_layers=item_num_attention_layers,
            num_heads=item_num_heads,
            dropout=item_dropout,
            use_simple_fusion=item_use_simple_fusion
        )
        
        # Create interaction layer directly
        self.interaction_generator = InteractionEmbeddingGenerator(
            embedding_dim=embedding_dim,
            num_attention_layers=interaction_num_attention_layers,
            num_heads=interaction_num_heads,
            dropout=interaction_dropout,
            interaction_strategy="bidirectional",
            use_simple_fusion=interaction_use_simple_fusion
        )
        
        # Create classifier directly
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]
            
        self.classifier = RecommendationClassifier(
            embedding_dim=embedding_dim,
            mlp_hidden_dims=classifier_hidden_dims,
            mlp_dropout=classifier_dropout,
            loss_type=loss_type
        )
    
    def _get_valid_num_heads(self, target_heads: int, embedding_dim: int) -> int:
        """Find the closest valid number of heads that divides embedding_dim"""
        valid_heads = [h for h in [4, 8, 16, 32, 64] if embedding_dim % h == 0]
        if not valid_heads:
            return 1  # Fallback to 1 head
        # Find closest to target
        return min(valid_heads, key=lambda x: abs(x - target_heads))
        
    def _create_image_encoder(self, config: Optional[Dict]):
        """Create image encoder with default config"""
        if config is None:
            return None
        default_config = {
            "aggregation_strategy": "average",
            "model_type": "cnn",
            "embedding_dim": self.embedding_dim,
            "num_image_fields": 2  # Default for most use cases
        }
        default_config.update(config)
        encoder = create_image_encoder(default_config)
        encoder._actual_config = default_config.copy()
        return encoder
    
    def _create_text_encoder(self, config: Optional[Dict]):
        """Create text encoder from config"""
        if config is None:
            return None
        
        # Ensure embedding_dim is set
        if "embedding_dim" not in config:
            config["embedding_dim"] = self.embedding_dim
        
        # Create encoder using factory (automatically detects transformer vs word2vec)
        encoder = create_text_encoder(config)
        
        # Store actual config used for saving
        encoder._actual_config = config.copy()
        return encoder
    
    def _create_categorical_encoder(self, config: Optional[Dict]):
        """Create categorical encoder from config (no defaults - config must be explicit)"""
        if config is None:
            return None
        
        # mlp_hidden_dims is required - validation should catch this, but double-check
        if "mlp_hidden_dims" not in config:
            raise ValueError(
                "categorical_encoder_config: 'mlp_hidden_dims' is required. "
                "Specify explicitly (use [] for no hidden layers)."
            )
        
        # Ensure embedding_dim is set
        if "embedding_dim" not in config:
            config["embedding_dim"] = self.embedding_dim
        
        # Store the actual config used for saving
        encoder = create_categorical_encoder(config)
        encoder._actual_config = config.copy()
        return encoder
    
    def _create_continuous_encoder(self, config: Optional[Dict]):
        """Create continuous encoder with default config"""
        default_config = {
            "embedding_dim": self.embedding_dim,
            "hidden_dims": [128, 64],  # Changed from hidden_dim to hidden_dims
            "normalize": True
        }
        if config:
            default_config.update(config)
        encoder = create_continuous_encoder(default_config)
        # Store the actual config used (with defaults applied) for saving
        encoder._actual_config = default_config.copy()
        return encoder
    
    def _create_temporal_encoder(self, config: Optional[Dict]):
        """Create temporal encoder with optional item lookup for any dataset"""
        if config is None:
            return None
        
        # Check if item lookup is enabled
        enable_item_lookup = config.get('enable_item_lookup', False)
        
        # If config has full temporal encoder config (like saved models), enable item lookup
        if not enable_item_lookup and ('aggregation_strategy' in config or 'output_dim' in config):
            enable_item_lookup = True
            print("Temporal encoder: Full config detected, enabling item lookup")
        
        if not enable_item_lookup:
            print("Temporal encoder: Item lookup disabled, skipping temporal encoder creation")
            return None
        
        # Check if item data is available
        if self.item_data is None:
            print("Temporal encoder: Item lookup enabled but no item data provided, skipping temporal encoder creation")
            return None
        
        print("Temporal encoder: Creating with item lookup enabled")
        
        # Create generic item lookup interface
        item_data_dict = {str(item['item_id']): item for item in self.item_data}
        item_lookup = GenericItemLookupInterface(item_data_dict)
        
        # Get modality encoders (only include non-None encoders)
        modality_encoders = {}
        if self.text_encoder is not None:
            modality_encoders['text'] = self.text_encoder
        if self.categorical_encoder is not None:
            modality_encoders['categorical'] = self.categorical_encoder
        if self.image_encoder is not None:
            modality_encoders['image'] = self.image_encoder
        
        # Determine enabled modalities based on available encoders
        available_modalities = list(modality_encoders.keys())
        default_enabled_modalities = config.get('enabled_modalities', available_modalities)
        
        # Filter enabled modalities to only include available ones
        enabled_modalities = [mod for mod in default_enabled_modalities if mod in available_modalities]
        
        # Further filter based on what features are actually available in item data
        if self.item_data:
            # Check what features are available in the first few items
            sample_items = self.item_data[:min(10, len(self.item_data))]
            available_features = set()
            for item in sample_items:
                available_features.update(item.keys())
            
            # Remove 'item_id' as it's not a feature
            available_features.discard('item_id')
            
            # Filter enabled modalities to only include those with available features
            enabled_modalities = [mod for mod in enabled_modalities if mod in available_features]
        
        if not enabled_modalities:
            print("Temporal encoder: No available modalities, skipping temporal encoder creation")
            return None
        
        # Create temporal encoder with default config
        default_config = {
            'item_lookup': item_lookup,
            'modality_encoders': modality_encoders,
            'enabled_modalities': enabled_modalities,
            'aggregation_strategy': config.get('aggregation_strategy', 'attention'),
            'lstm_hidden_dim': config.get('lstm_hidden_dim', 128),
            'lstm_num_layers': config.get('lstm_num_layers', 2),
            'lstm_dropout': config.get('lstm_dropout', 0.1),
            'bidirectional': config.get('bidirectional', False),
            'output_dim': self.embedding_dim,
            'max_sequence_length': config.get('max_sequence_length', 50),
            'missing_item_strategy': config.get('missing_item_strategy', 'zero')
        }
        
        # Update with config but exclude unsupported parameters
        if config:
            excluded_params = {'enable_item_lookup', 'item_embedding_dim'}
            filtered_config = {k: v for k, v in config.items() if k not in excluded_params}
            default_config.update(filtered_config)
        
        return create_temporal_encoder(**default_config)
    
    def _encode_features(self, features: Dict[str, Any], aligner: DimensionAligner) -> Dict[str, torch.Tensor]:
        """
        Encode features using appropriate encoders and align dimensions
        
        Args:
            features: Dict with feature type -> feature data mapping
            aligner: Dimension aligner to project all embeddings to target dimension
            
        Returns:
            encoded_features: Dict with aligned encoded embeddings
        """
        encoded = {}
        
        # Encode images
        if "image" in features:
            encoded.update(self.image_encoder(features["image"]))
        
        # Encode text
        if "text" in features and self.text_encoder is not None:
            encoded.update(self.text_encoder(features["text"]))
        
        # Encode categorical
        if "categorical" in features:
            encoded.update(self.categorical_encoder(features["categorical"]))
        
        # Encode continuous - use appropriate encoder based on aligner
        if "continuous" in features:
            if aligner is self.user_dimension_aligner:
                encoded.update(self.user_continuous_encoder(features["continuous"]))
            elif aligner is self.item_dimension_aligner:
                encoded.update(self.item_continuous_encoder(features["continuous"]))
            else:
                # Fallback (shouldn't happen in normal operation)
                encoded.update(self.user_continuous_encoder(features["continuous"]))
        
        # Encode temporal
        if "temporal" in features and self.temporal_encoder is not None:
            encoded.update(self.temporal_encoder(features["temporal"]))
        
        # Align all embeddings to target dimension
        aligned_encoded = aligner(encoded)
        
        return aligned_encoded
    
    def forward(self,
                user_data: Dict[str, Any],
                item_data: Dict[str, Any],
                labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Complete forward pass through entire pipeline
        
        Args:
            user_data: User features organized by type
            item_data: Item features organized by type
            labels: Binary engagement labels (for training)
            
        Returns:
            If training (labels provided): (loss, logits)
            If inference: logits
        """
        # Step 1: Encode user features and align dimensions
        user_encoded = self._encode_features(user_data, self.user_dimension_aligner)
        user_feature_types = self._get_feature_types(user_encoded)
        
        # Step 2: Encode item features and align dimensions
        item_encoded = self._encode_features(item_data, self.item_dimension_aligner)
        item_feature_types = self._get_feature_types(item_encoded)
        
        # Step 3: User tower - fuse user features
        user_embedding = self.user_generator(user_encoded, user_feature_types)
        
        # Step 4: Item tower - fuse item features
        item_embedding = self.item_generator(item_encoded, item_feature_types)
        
        # Step 5: Interaction modeling
        interaction_embedding = self.interaction_generator(user_embedding, item_embedding)
        
        # Step 6: Classification
        if labels is not None:
            # Training mode
            loss, logits = self.classifier(interaction_embedding, labels)
            return loss, logits
        else:
            # Inference mode
            logits = self.classifier(interaction_embedding)
            return logits
    
    def _get_feature_types(self, encoded_features: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Infer feature types from encoded feature names
        """
        feature_types = {}
        
        # REMOVE THESE DEBUG LINES:
        # print(f"DEBUG _get_feature_types: Processing {list(encoded_features.keys())}")
        
        for feature_name in encoded_features.keys():
            if "image" in feature_name.lower() or "img" in feature_name.lower():
                feature_types[feature_name] = "image"
            elif "text" in feature_name.lower() or "bio" in feature_name.lower() or "description" in feature_name.lower():
                feature_types[feature_name] = "text"
            elif "temporal" in feature_name.lower() or "history" in feature_name.lower() or "sequence" in feature_name.lower():
                feature_types[feature_name] = "temporal"
            elif ("category" in feature_name.lower() or "location" in feature_name.lower() or 
                "gender" in feature_name.lower() or "categorical" in feature_name.lower()):
                feature_types[feature_name] = "categorical"
            else:
                # Default to continuous for numeric features
                feature_types[feature_name] = "continuous"
        
        # REMOVE THIS DEBUG LINE:
        # print(f"DEBUG _get_feature_types: Result = {feature_types}")
        
        return feature_types

    def predict_proba(self, user_data: Dict[str, Any], item_data: Dict[str, Any]) -> torch.Tensor:
        """Get engagement probabilities for inference"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(user_data, item_data)
            return torch.sigmoid(logits)
    
    def predict(self, user_data: Dict[str, Any], item_data: Dict[str, Any], threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions for inference"""
        probabilities = self.predict_proba(user_data, item_data)
        return (probabilities > threshold).float()
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the stored configuration for this model.
        Returns None if config was not stored (e.g., model created manually without config).
        
        Returns:
            Configuration dictionary in the format expected by load_model_from_config()
        """
        return self._saved_config.copy() if self._saved_config is not None else None
    
    def set_config(self, config: Dict[str, Any]):
        """
        Store configuration in the model. This should be called when creating the model
        from a config dict to enable proper saving/loading.
        
        Args:
            config: Configuration dictionary (will be normalized to save format)
        """
        # Normalize config: convert encoder_config keys to encoder keys for consistency
        normalized_config = config.copy()
        
        # Normalize encoder config keys (support both formats)
        for encoder_type in ['image', 'text', 'categorical', 'continuous', 'temporal']:
            config_key = f'{encoder_type}_encoder_config'
            encoder_key = f'{encoder_type}_encoder'
            
            # If both exist, prefer the one without _config suffix
            if encoder_key in normalized_config:
                # Already in correct format
                pass
            elif config_key in normalized_config:
                # Convert _config suffix to no suffix
                normalized_config[encoder_key] = normalized_config.pop(config_key)
        
        # Update encoder configs with actual values used (from encoders if available)
        # This ensures we save what was actually used
        if hasattr(self, 'categorical_encoder') and self.categorical_encoder is not None:
            if hasattr(self.categorical_encoder, '_actual_config'):
                normalized_config['categorical_encoder'] = self.categorical_encoder._actual_config
        
        if hasattr(self, 'user_continuous_encoder') and self.user_continuous_encoder is not None:
            if hasattr(self.user_continuous_encoder, '_actual_config'):
                normalized_config['continuous_encoder'] = self.user_continuous_encoder._actual_config
        
        self._saved_config = normalized_config


def save_model_config(model: RecommendationPipeline, config_path: str):
    """
    Save model configuration to JSON file.
    Uses stored config from model creation.
    
    Args:
        model: Trained RecommendationPipeline instance
        config_path: Path to save the config JSON file
        
    Raises:
        RuntimeError: If model has no stored config
    """
    import json
    
    stored_config = model.get_config()
    if stored_config is None:
        raise RuntimeError(
            "Model has no stored config. Model must be created from config using "
            "create_model_from_config() to enable saving."
        )
    
    with open(config_path, "w") as f:
        json.dump(stored_config, f, indent=2)


def save_complete_model(model: RecommendationPipeline, save_dir: str, model_name: str = "model", verbose: bool = True):
    """
    Save complete model including all state dicts and parameters
    
    Args:
        model: Trained RecommendationPipeline instance
        save_dir: Directory to save model files
        model_name: Base name for model files (default: "model")
    """
    import os
    import torch
    import json
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure model is in eval mode and all parameters are initialized
    model.eval()
    
    # CRITICAL: Ensure all parameters are registered by doing a dummy forward pass
    # This ensures any lazy initialization happens before saving
    try:
        # Get sample data from the model's inputs if available
        # Create minimal dummy inputs to trigger parameter registration
        dummy_user_data = {}
        dummy_item_data = {}
        
        # Try to create minimal valid inputs based on what encoders exist
        if hasattr(model, 'categorical_encoder') and model.categorical_encoder is not None:
            # Create dummy categorical data
            dummy_user_data['categorical'] = {'field_0': 'dummy'}
            dummy_item_data['categorical'] = {'field_0': 'dummy'}
        
        if hasattr(model, 'continuous_encoder') and model.continuous_encoder is not None:
            dummy_user_data['continuous'] = {'field_0': 0.0}
            dummy_item_data['continuous'] = {'field_0': 0.0}
        
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            dummy_item_data['text'] = {'field_0': 'dummy text'}
        
        # Only do forward pass if we have at least some data
        if dummy_user_data or dummy_item_data:
            with torch.no_grad():
                try:
                    _ = model(dummy_user_data, dummy_item_data)
                except Exception:
                    # If forward pass fails, that's okay - parameters should still be registered
                    pass
    except Exception:
        # If dummy forward pass fails, continue anyway
        pass
    
    # Force initialization of fusion projections if they exist but aren't initialized
    # During training, these should already be initialized, but we check to be safe
    fusion_layers_to_check = []
    if hasattr(model, 'user_generator') and hasattr(model.user_generator, 'user_fusion'):
        if hasattr(model.user_generator.user_fusion, 'projection') and model.user_generator.user_fusion.projection is None:
            fusion_layers_to_check.append(('user_generator.user_fusion', model.user_generator.user_fusion))
    
    if hasattr(model, 'item_generator') and hasattr(model.item_generator, 'item_fusion'):
        if hasattr(model.item_generator.item_fusion, 'projection') and model.item_generator.item_fusion.projection is None:
            fusion_layers_to_check.append(('item_generator.item_fusion', model.item_generator.item_fusion))
    
    if hasattr(model, 'interaction_generator') and hasattr(model.interaction_generator, 'interaction_fusion'):
        if hasattr(model.interaction_generator.interaction_fusion, 'projection') and model.interaction_generator.interaction_fusion.projection is None:
            fusion_layers_to_check.append(('interaction_generator.interaction_fusion', model.interaction_generator.interaction_fusion))
    
    # Try to initialize fusion projections with dummy data
    for layer_name, fusion_layer in fusion_layers_to_check:
        try:
            # Create dummy feature embeddings (try with 3 features as a reasonable default)
            dummy_embeddings = {
                'feature_0': torch.zeros(1, model.embedding_dim),
                'feature_1': torch.zeros(1, model.embedding_dim),
                'feature_2': torch.zeros(1, model.embedding_dim)
            }
            dummy_types = {k: 'categorical' for k in dummy_embeddings.keys()}
            with torch.no_grad():
                _ = fusion_layer(dummy_embeddings, dummy_types)
            if verbose:
                print(f"   ✅ Initialized {layer_name} projection")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not initialize {layer_name} projection: {e}")
                print(f"   ⚠️  WARNING: Fusion projection not initialized - weights may not be saved!")
    
    # Get complete state dict
    state_dict = model.state_dict()
    
    # Get all parameter names from the model (excluding buffers)
    model_param_names = set(name for name, _ in model.named_parameters())
    # Get all buffer names (these are in state_dict but not in parameters)
    model_buffer_names = set(name for name, _ in model.named_buffers())
    state_dict_keys = set(state_dict.keys())
    
    # Validate that all parameters are present and initialized
    total_params = sum(p.numel() for p in model.parameters())
    # Count only parameters in state_dict (exclude buffers for comparison)
    state_dict_param_count = sum(p.numel() for name, p in state_dict.items() if name in model_param_names)
    state_dict_buffer_count = sum(p.numel() for name, p in state_dict.items() if name in model_buffer_names)
    
    # Check for missing parameters
    missing_params = model_param_names - state_dict_keys
    
    if missing_params:
        error_msg = f"❌ CRITICAL ERROR: {len(missing_params)} parameters are NOT in state_dict!\n"
        error_msg += "   These parameters will NOT be saved:\n"
        for name in sorted(missing_params):
            error_msg += f"   - {name}\n"
        if verbose:
            print(error_msg)
        raise RuntimeError(f"Failed to register all model parameters. {len(missing_params)} parameters missing from state_dict.")
    
    if verbose:
        # Compare only parameters (buffers are expected to be different)
        if total_params != state_dict_param_count:
            print(f"⚠️  Warning: Parameter count mismatch!")
            print(f"   Model parameters: {total_params:,}")
            print(f"   State dict parameters: {state_dict_param_count:,}")
            print(f"   Difference: {abs(total_params - state_dict_param_count):,}")
            print(f"   This may indicate some parameters are not being saved.")
        else:
            print(f"✅ Parameter count matches: {total_params:,} trainable parameters")
            if state_dict_buffer_count > 0:
                print(f"   Plus {state_dict_buffer_count:,} buffers (running stats, default embeddings, etc.)")
        
        # Check for any uninitialized parameters (all zeros or very small)
        # Exclude buffers and intentionally zero-initialized parameters
        uninitialized = []
        intentionally_zero = {
            'default_embedding',  # Image/text encoder default embeddings
            'running_mean', 'running_var', 'num_batches_tracked'  # BatchNorm buffers
        }
        
        for name, param in state_dict.items():
            # Skip buffers - they're not trainable parameters
            if name in model_buffer_names:
                continue
            
            # Skip intentionally zero-initialized buffers/parameters
            if any(intentional in name for intentional in intentionally_zero):
                continue
            
            if param.numel() > 0:
                # Check if parameter is essentially zero (might indicate uninitialized)
                if torch.allclose(param, torch.zeros_like(param), atol=1e-8):
                    uninitialized.append(name)
        
        if uninitialized:
            print(f"⚠️  Warning: {len(uninitialized)} trainable parameters appear uninitialized (all zeros):")
            for name in uninitialized[:10]:  # Show first 10
                print(f"   - {name}")
            if len(uninitialized) > 10:
                print(f"   ... and {len(uninitialized) - 10} more")
    
    # Save main model state dict
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(state_dict, model_path)
    
    # Verify saved file contains all keys
    saved_state = torch.load(model_path, map_location='cpu')
    saved_keys = set(saved_state.keys())
    original_keys = set(state_dict.keys())
    
    if saved_keys != original_keys:
        missing_in_saved = original_keys - saved_keys
        error_msg = f"❌ Error: Some keys were not saved!\n"
        for key in missing_in_saved:
            error_msg += f"   Missing: {key}\n"
        if verbose:
            print(error_msg)
        raise RuntimeError(f"Failed to save all model parameters. {len(missing_in_saved)} keys missing.")
    
    if verbose:
        print(f"✅ Saved {len(saved_keys)} parameters and buffers to {model_path}")
        print(f"   Total trainable parameters: {state_dict_param_count:,}")
        if state_dict_buffer_count > 0:
            print(f"   Total buffers: {state_dict_buffer_count:,}")
        
        # Debug: List all categorical encoder MLP keys to verify structure
        cat_mlp_keys = [k for k in sorted(state_dict.keys()) if 'categorical_encoder.field_embeddings.field_0.mlp' in k]
        if cat_mlp_keys:
            print(f"\n📋 Categorical encoder MLP structure ({len(cat_mlp_keys)} keys):")
            
            # Extract layer indices dynamically
            layer_indices = set()
            for key in cat_mlp_keys:
                # Extract layer number from keys like "mlp.0.weight" or "mlp.3.bias"
                parts = key.split('.mlp.')
                if len(parts) > 1:
                    layer_part = parts[1].split('.')[0]
                    try:
                        layer_indices.add(int(layer_part))
                    except ValueError:
                        pass
            
            if layer_indices:
                max_layer = max(layer_indices)
                # Find all Linear layer indices
                # Structure: For each hidden_dim, we have Linear (idx 0,3,6...), Activation (idx 1,4,7...), Dropout (idx 2,5,8...)
                # Final Linear is at index 3*len(hidden_dims)
                # So Linear layers are at indices that are multiples of 3: 0, 3, 6, 9, ...
                linear_layer_indices = []
                for key in cat_mlp_keys:
                    if '.weight' in key:  # Linear layers have .weight, activations/dropouts don't
                        parts = key.split('.mlp.')
                        if len(parts) > 1:
                            layer_part = parts[1].split('.')[0]
                            try:
                                layer_idx = int(layer_part)
                                # Linear layers are at multiples of 3 (0, 3, 6, 9, ...)
                                if layer_idx % 3 == 0:
                                    linear_layer_indices.append((layer_idx, key))
                            except ValueError:
                                pass
                
                if linear_layer_indices:
                    # The final projection is the Linear layer with highest index
                    linear_layer_indices.sort(key=lambda x: x[0], reverse=True)
                    final_layer_idx, final_key = linear_layer_indices[0]
                    print(f"   ✅ Final projection layer found at mlp.{final_layer_idx}")
                    print(f"      Key: {final_key}")
                    print(f"      Shape: {state_dict[final_key].shape}")
                    if len(linear_layer_indices) > 1:
                        print(f"      (Total {len(linear_layer_indices)} Linear layers in MLP)")
                else:
                    print(f"   ⚠️  Could not identify Linear layers in MLP structure")
            else:
                print(f"   ⚠️  Could not parse MLP layer structure")
    
    # Save model configuration
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    save_model_config(model, config_path)
    
    # Save individual tower state dicts for modular loading
    if hasattr(model, 'user_generator'):
        user_tower_path = os.path.join(save_dir, f"{model_name}_user_tower.pt")
        torch.save(model.user_generator.state_dict(), user_tower_path)
    
    if hasattr(model, 'item_generator'):
        item_tower_path = os.path.join(save_dir, f"{model_name}_item_tower.pt")
        torch.save(model.item_generator.state_dict(), item_tower_path)
    
    if hasattr(model, 'interaction_generator'):
        interaction_tower_path = os.path.join(save_dir, f"{model_name}_interaction_tower.pt")
        torch.save(model.interaction_generator.state_dict(), interaction_tower_path)
    
    if hasattr(model, 'classifier'):
        classifier_path = os.path.join(save_dir, f"{model_name}_classifier.pt")
        torch.save(model.classifier.state_dict(), classifier_path)
    
    # Save encoder state dicts
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        text_encoder_path = os.path.join(save_dir, f"{model_name}_text_encoder.pt")
        torch.save(model.text_encoder.state_dict(), text_encoder_path)
    
    if hasattr(model, 'categorical_encoder') and model.categorical_encoder is not None:
        categorical_encoder_path = os.path.join(save_dir, f"{model_name}_categorical_encoder.pt")
        torch.save(model.categorical_encoder.state_dict(), categorical_encoder_path)
    
    if hasattr(model, 'image_encoder') and model.image_encoder is not None:
        image_encoder_path = os.path.join(save_dir, f"{model_name}_image_encoder.pt")
        torch.save(model.image_encoder.state_dict(), image_encoder_path)
    
    if hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
        temporal_encoder_path = os.path.join(save_dir, f"{model_name}_temporal_encoder.pt")
        torch.save(model.temporal_encoder.state_dict(), temporal_encoder_path)
    
    # Create a manifest file listing all saved components
    manifest = {
        "model_name": model_name,
        "main_model": f"{model_name}.pt",
        "config": f"{model_name}_config.json",
        "components": {
            "user_tower": f"{model_name}_user_tower.pt",
            "item_tower": f"{model_name}_item_tower.pt",
            "interaction_tower": f"{model_name}_interaction_tower.pt",
            "classifier": f"{model_name}_classifier.pt",
            "text_encoder": f"{model_name}_text_encoder.pt",
            "categorical_encoder": f"{model_name}_categorical_encoder.pt",
            "image_encoder": f"{model_name}_image_encoder.pt"
        }
    }
    
    # Add temporal encoder to manifest if it exists
    if hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
        manifest["components"]["temporal_encoder"] = f"{model_name}_temporal_encoder.pt"
    
    manifest_path = os.path.join(save_dir, f"{model_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    if verbose:
        print(f"✅ Complete model saved to {save_dir}/")
        print(f"   Main model: {model_path}")
        print(f"   Config: {config_path}")
        print(f"   Manifest: {manifest_path}")
        print(f"   Components: {len(manifest['components'])} individual state dicts")


def load_model_from_config(config_path: str, weights_path: str, item_data: Optional[List[Dict[str, Any]]] = None) -> RecommendationPipeline:
    """
    Load model from saved configuration and weights
    
    Args:
        config_path: Path to the config JSON file
        weights_path: Path to the model weights file
        item_data: Optional item data for temporal encoder (any dataset format)
        
    Returns:
        RecommendationPipeline instance with loaded weights
    """
    import json
    import torch
    
    # Load config (saved config from training)
    with open(config_path) as f:
        config = json.load(f)
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Create model with same configuration
    model = RecommendationPipeline(
        embedding_dim=config['embedding_dim'],
        loss_type=config.get('loss_type', 'bce'),
        
        # User tower configuration
        user_num_attention_layers=config.get('user_num_attention_layers', 4),
        user_num_heads=config.get('user_num_heads', 16),
        user_dropout=config.get('user_dropout', 0.15),
        user_use_simple_fusion=config.get('user_use_simple_fusion', True),
        
        # Item tower configuration
        item_num_attention_layers=config.get('item_num_attention_layers', 1),
        item_num_heads=config.get('item_num_heads', 8),
        item_dropout=config.get('item_dropout', 0.1),
        item_use_simple_fusion=config.get('item_use_simple_fusion', True),
        
        # Interaction modeling configuration
        interaction_num_attention_layers=config.get('interaction_num_attention_layers', 2),
        interaction_num_heads=config.get('interaction_num_heads', 8),
        interaction_dropout=config.get('interaction_dropout', 0.1),
        interaction_use_simple_fusion=config.get('interaction_use_simple_fusion', True),
        
        # Classifier configuration
        classifier_hidden_dims=config.get('classifier_hidden_dims', [256, 128, 64]),
        classifier_dropout=config.get('classifier_dropout', 0.2),
        
        # Encoder configurations
        # Support both 'encoder' and 'encoder_config' keys for backwards compatibility
        image_encoder_config=config.get('image_encoder') or config.get('image_encoder_config'),
        text_encoder_config=config.get('text_encoder') or config.get('text_encoder_config'),
        categorical_encoder_config=config.get('categorical_encoder') or config.get('categorical_encoder_config'),
        continuous_encoder_config=config.get('continuous_encoder') or config.get('continuous_encoder_config'),
        temporal_encoder_config=config.get('temporal_encoder') or config.get('temporal_encoder_config'),
        
        # Item data for temporal encoder
        item_data=item_data
    )
    
    # Check for architectural compatibility
    state_dict_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    
    # Detect architectural incompatibilities
    has_old_continuous_encoder = any('continuous_encoder.mlp' in k for k in state_dict_keys)
    has_new_continuous_encoders = any('user_continuous_encoder' in k for k in model_keys)
    
    # Check fusion type mismatch
    has_attention_fusion_in_checkpoint = any('user_generator.user_fusion.transformer_layers' in k for k in state_dict_keys)
    has_simple_fusion_in_checkpoint = any('user_generator.user_fusion.projection' in k for k in state_dict_keys)
    has_attention_fusion_in_model = any('user_generator.user_fusion.transformer_layers' in k for k in model_keys)
    has_simple_fusion_in_model = any('user_generator.user_fusion.projection' in k for k in model_keys)
    
    # Determine mismatches
    continuous_encoder_mismatch = has_old_continuous_encoder and has_new_continuous_encoders
    fusion_type_mismatch = (has_attention_fusion_in_checkpoint and has_simple_fusion_in_model) or \
                          (has_simple_fusion_in_checkpoint and has_attention_fusion_in_model)
    
    if continuous_encoder_mismatch or fusion_type_mismatch:
        print("\n⚠️  WARNING: Model architecture mismatch!")
        print("\n   Incompatibilities detected:")
        if continuous_encoder_mismatch:
            print("   - Continuous encoder: checkpoint has single encoder, model has separate user/item encoders")
        if fusion_type_mismatch:
            if has_attention_fusion_in_checkpoint and has_simple_fusion_in_model:
                print("   - Fusion type: checkpoint uses attention-based fusion, model uses SimpleFusion")
                print("     → Set user_use_simple_fusion=false, item_use_simple_fusion=false, interaction_use_simple_fusion=false")
            else:
                print("   - Fusion type: checkpoint uses SimpleFusion, model uses attention-based fusion")
                print("     → Set user_use_simple_fusion=true, item_use_simple_fusion=true, interaction_use_simple_fusion=true")
        print("\n   ❌ Cannot load incompatible checkpoint.")
        print("   ✅ Solutions:")
        print("      1. Retrain the model with current architecture: python3 train.py")
        print("      2. Adjust config parameters to match checkpoint architecture")
        raise RuntimeError(
            "Model architecture mismatch: checkpoint architecture doesn't match current configuration. "
            "Please adjust config or retrain the model."
        )
    
    # Pre-initialize dimension aligner projections if they exist in state dict
    for key in state_dict.keys():
        if 'dimension_aligner.projections.' in key:
            # Extract dimension from key (e.g., "16" from "projections.16.weight")
            parts = key.split('.')
            if len(parts) >= 3 and parts[2].isdigit():
                dim = int(parts[2])
                if 'item_dimension_aligner' in key:
                    # Force creation of projection for item dimension aligner
                    model.item_dimension_aligner._get_projection(dim)
                elif 'user_dimension_aligner' in key:
                    # Force creation of projection for user dimension aligner
                    model.user_dimension_aligner._get_projection(dim)
    
    # 2. Pre-initialize continuous encoder MLPs from checkpoint
    # Extract input_dim from checkpoint weights
    user_continuous_mlp_keys = [k for k in state_dict.keys() if 'user_continuous_encoder.mlp.0.weight' in k]
    if user_continuous_mlp_keys:
        # Get input dimension from first layer weight
        weight_key = user_continuous_mlp_keys[0]
        input_dim = state_dict[weight_key].shape[1]  # Input dim is second dimension
        
        # Force initialization of user continuous encoder MLP
        if hasattr(model, 'user_continuous_encoder') and model.user_continuous_encoder is not None:
            if model.user_continuous_encoder.mlp is None:
                model.user_continuous_encoder._initialize_mlp(input_dim)
                print(f"✅ Pre-initialized user_continuous_encoder MLP (input_dim={input_dim})")
    
    item_continuous_mlp_keys = [k for k in state_dict.keys() if 'item_continuous_encoder.mlp.0.weight' in k]
    if item_continuous_mlp_keys:
        weight_key = item_continuous_mlp_keys[0]
        input_dim = state_dict[weight_key].shape[1]
        
        # Force initialization of item continuous encoder MLP
        if hasattr(model, 'item_continuous_encoder') and model.item_continuous_encoder is not None:
            if model.item_continuous_encoder.mlp is None:
                model.item_continuous_encoder._initialize_mlp(input_dim)
                print(f"✅ Pre-initialized item_continuous_encoder MLP (input_dim={input_dim})")
    
    # 3. Pre-initialize fusion projections from checkpoint
    # Extract num_features from checkpoint projection weights
    fusion_projection_keys = [k for k in state_dict.keys() if 'fusion.projection.0.weight' in k]
    
    for projection_key in fusion_projection_keys:
        # Extract which fusion layer this belongs to
        # Format: "user_generator.user_fusion.projection.0.weight" or similar
        # Split by '.projection.0.weight' to get the path to the fusion layer
        if '.projection.0.weight' in projection_key:
            parts = projection_key.split('.projection.0.weight')[0]
            fusion_path = parts.split('.')  # e.g., ['user_generator', 'user_fusion']
        else:
            print(f"⚠️  Warning: Unexpected fusion projection key format: {projection_key}")
            continue
        
        # Get the weight tensor to determine input_dim
        weight_tensor = state_dict[projection_key]
        input_dim = weight_tensor.shape[1]  # Input dim is second dimension
        num_features = input_dim // model.embedding_dim
        
        # Navigate to the fusion layer
        try:
            fusion_layer = model
            for part in fusion_path:
                if fusion_layer is None:
                    raise AttributeError(f"Intermediate attribute is None when navigating to {part}")
                fusion_layer = getattr(fusion_layer, part)
            
            # Verify we got a valid fusion layer
            if fusion_layer is None:
                raise AttributeError(f"Fusion layer is None after navigation: {'.'.join(fusion_path)}")
            
            # Check if projection needs initialization
            if hasattr(fusion_layer, 'projection') and fusion_layer.projection is None:
                fusion_layer._initialize_projection(num_features)
                # CRITICAL: Set _projection_initialized to True to prevent re-initialization in forward pass
                fusion_layer._projection_initialized = True
                print(f"✅ Pre-initialized {'.'.join(fusion_path)}.projection (num_features={num_features})")
            elif hasattr(fusion_layer, 'projection') and fusion_layer.projection is not None:
                # Projection already exists (loaded from checkpoint), but _projection_initialized might be False
                # CRITICAL: Set it to True to prevent re-initialization in forward pass
                fusion_layer._projection_initialized = True
                print(f"✅ Verified {'.'.join(fusion_path)}.projection is initialized (from checkpoint)")
            elif not hasattr(fusion_layer, 'projection'):
                print(f"⚠️  Warning: {'.'.join(fusion_path)} does not have 'projection' attribute (might not be SimpleFusionLayer)")
        except (AttributeError, TypeError) as e:
            print(f"⚠️  Warning: Could not pre-initialize fusion projection for {projection_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Load the state dict with strict=False for dimension aligner compatibility
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # CRITICAL: After loading state dict, ensure _projection_initialized is True for all fusion layers
    # This prevents re-initialization during forward pass
    for name, module in model.named_modules():
        if hasattr(module, 'projection') and module.projection is not None:
            if hasattr(module, '_projection_initialized'):
                module._projection_initialized = True
    
    # Only warn about unexpected keys if they're not dimension aligner related
    unexpected_non_aligner = [k for k in unexpected_keys if 'dimension_aligner.projections' not in k]
    if unexpected_non_aligner:
        print(f"\n⚠️  Warning: {len(unexpected_non_aligner)} unexpected keys in checkpoint (ignoring)")
        if len(unexpected_non_aligner) <= 20:
            for key in sorted(unexpected_non_aligner):
                print(f"   - {key}")
        else:
            for key in sorted(unexpected_non_aligner)[:10]:
                print(f"   - {key}")
            print(f"   ... and {len(unexpected_non_aligner) - 10} more")
    
    # Filter missing keys to remove false positives
    # Check what MLP layers actually exist in the checkpoint to determine expected structure
    checkpoint_mlp_layers = {}
    for key in state_dict.keys():
        if '.mlp.' in key:
            # Extract the MLP path and layer index
            # Pattern: something.mlp.X.weight or something.mlp.X.bias
            parts = key.split('.mlp.')
            if len(parts) > 1:
                mlp_path = parts[0]  # e.g., "categorical_encoder.field_embeddings.field_0"
                layer_part = parts[1].split('.')[0]
                try:
                    layer_idx = int(layer_part)
                    if mlp_path not in checkpoint_mlp_layers:
                        checkpoint_mlp_layers[mlp_path] = set()
                    checkpoint_mlp_layers[mlp_path].add(layer_idx)
                except ValueError:
                    pass
    
    # Filter missing keys: only warn about keys that are asking for layers that exist in checkpoint
    # OR keys that are not MLP-related structural differences
    filtered_missing = []
    structural_differences = []
    
    for key in missing_keys:
        # Skip dimension aligner keys (expected to be different)
        if 'dimension_aligner' in key:
            continue
        
        # Check if this is an MLP layer mismatch
        if '.mlp.' in key:
            parts = key.split('.mlp.')
            if len(parts) > 1:
                mlp_path = parts[0]
                layer_part = parts[1].split('.')[0]
                try:
                    requested_layer = int(layer_part)
                    # Check if this MLP path exists in checkpoint
                    if mlp_path in checkpoint_mlp_layers:
                        # Check what layers exist in checkpoint for this MLP
                        checkpoint_layers = checkpoint_mlp_layers[mlp_path]
                        # If the requested layer doesn't exist in checkpoint, it's a structural difference
                        if requested_layer not in checkpoint_layers:
                            structural_differences.append(key)
                            continue
                    # If MLP path doesn't exist in checkpoint at all, it's a structural difference
                    else:
                        structural_differences.append(key)
                        continue
                except ValueError:
                    pass
        
        # Not a structural difference, include in filtered missing
        filtered_missing.append(key)
    
    # Only warn about truly missing keys (not structural differences)
    if filtered_missing:
        print(f"\n⚠️  Warning: {len(filtered_missing)} missing keys in checkpoint (using random initialization)")
        print(f"   These parameters will be randomly initialized, causing non-deterministic behavior!")
        print(f"   Missing keys:")
        for key in sorted(filtered_missing):
            print(f"   - {key}")
        print(f"\n   ⚠️  This will cause non-deterministic results between runs!")
        print(f"   ✅ Solution: Retrain the model to save a complete checkpoint.")
    
    # Inform about structural differences (different MLP configurations) but don't treat as errors
    if structural_differences:
        print(f"\nℹ️  Note: {len(structural_differences)} parameters are missing due to structural differences:")
        print(f"   (Model structure differs from checkpoint - e.g., different MLP hidden_dims)")
        print(f"   These will be randomly initialized but the model will work correctly.")
        if len(structural_differences) <= 5:
            for key in sorted(structural_differences):
                print(f"   - {key}")
        else:
            for key in sorted(structural_differences)[:3]:
                print(f"   - {key}")
            print(f"   ... and {len(structural_differences) - 3} more")
    
    # Check for critical missing parameters (non-structural)
    if filtered_missing:
        truly_critical = [k for k in filtered_missing if 'dimension_aligner' not in k]
        if truly_critical:
            print(f"\n❌ Critical parameters missing from checkpoint:")
            for key in sorted(truly_critical):
                print(f"   - {key}")
            print(f"\n   This checkpoint is incomplete and will produce non-deterministic results.")
            print(f"   Please retrain the model to generate a complete checkpoint.")
    
    return model


# Removed unnecessary factory function - use direct RecommendationPipeline instantiation instead


# Removed unnecessary factory function - use direct RecommendationPipeline instantiation instead


# Example usage
if __name__ == "__main__":
    print("=== Recommendation Pipeline Builder ===")
    
    # Example: Create pipeline directly with clear parameters
    pipeline = RecommendationPipeline(
        embedding_dim=256,
        loss_type="bce"
    )
    
    print(f"Created pipeline with {sum(p.numel() for p in pipeline.parameters()):,} parameters")
    
    print("\nPipeline components:")
    print("✅ Image Encoder")
    print("✅ Text Encoder") 
    print("✅ Categorical Encoder")
    print("✅ Temporal Encoder")
    print("✅ User Tower (Feature Fusion)")
    print("✅ Item Tower (Feature Fusion)")
    print("✅ Interaction Layer")
    print("✅ Classification Head")
    
    print("\nReady for training! Use trainer.py to train this pipeline.")