"""
Config validation utilities for recommendation system.
Validates configuration before model creation to ensure architecture compatibility.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of config validation"""
    errors: List[str]
    warnings: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0
    
    def __str__(self) -> str:
        if self.is_valid:
            msg = "✅ Config validation passed"
            if self.warnings:
                msg += f" (with {len(self.warnings)} warnings)"
            return msg
        else:
            msg = f"❌ Config validation failed with {len(self.errors)} error(s):\n"
            for i, error in enumerate(self.errors, 1):
                msg += f"  {i}. {error}\n"
            if self.warnings:
                msg += f"\n⚠️  {len(self.warnings)} warning(s):\n"
                for i, warning in enumerate(self.warnings, 1):
                    msg += f"  {i}. {warning}\n"
            return msg


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration dictionary for recommendation model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []
    
    # Required top-level fields
    if 'embedding_dim' not in config:
        errors.append("'embedding_dim' is required")
    elif not isinstance(config['embedding_dim'], int) or config['embedding_dim'] <= 0:
        errors.append("'embedding_dim' must be a positive integer")
    
    # Validate encoder configs if present
    for encoder_type in ['image', 'text', 'categorical', 'continuous', 'temporal']:
        config_key = f'{encoder_type}_encoder_config'
        encoder_key = f'{encoder_type}_encoder'
        
        # Check both naming conventions
        encoder_config = config.get(config_key) or config.get(encoder_key)
        
        if encoder_config is not None:
            if encoder_type == 'categorical':
                validate_categorical_encoder_config(encoder_config, errors, warnings)
            elif encoder_type == 'text':
                validate_text_encoder_config(encoder_config, errors, warnings)
            elif encoder_type == 'image':
                validate_image_encoder_config(encoder_config, errors, warnings)
            elif encoder_type == 'continuous':
                validate_continuous_encoder_config(encoder_config, errors, warnings)
            elif encoder_type == 'temporal':
                validate_temporal_encoder_config(encoder_config, errors, warnings)
    
    # Validate architecture compatibility
    embedding_dim = config.get('embedding_dim')
    if embedding_dim:
        # Check num_heads divisibility
        for tower in ['user', 'item', 'interaction']:
            num_heads = config.get(f'{tower}_num_heads')
            if num_heads and embedding_dim % num_heads != 0:
                errors.append(
                    f"'{tower}_num_heads' ({num_heads}) must divide 'embedding_dim' ({embedding_dim})"
                )
    
    return ValidationResult(errors, warnings)


def validate_categorical_encoder_config(config: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Validate categorical encoder configuration"""
    # mlp_hidden_dims is REQUIRED for architecture verification
    if 'mlp_hidden_dims' not in config:
        errors.append(
            "categorical_encoder_config: 'mlp_hidden_dims' is required for architecture verification. "
            "Specify explicitly (use [] for no hidden layers)."
        )
    elif not isinstance(config['mlp_hidden_dims'], list):
        errors.append("categorical_encoder_config: 'mlp_hidden_dims' must be a list")
    
    # Other required fields
    if 'embedding_dim' not in config:
        errors.append("categorical_encoder_config: 'embedding_dim' is required")
    
    if 'aggregation_strategy' not in config:
        errors.append("categorical_encoder_config: 'aggregation_strategy' is required")


def validate_text_encoder_config(config: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Validate text encoder configuration"""
    if 'model_name' not in config:
        errors.append("text_encoder_config: 'model_name' is required")
    
    if 'embedding_dim' not in config:
        errors.append("text_encoder_config: 'embedding_dim' is required")


def validate_image_encoder_config(config: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Validate image encoder configuration"""
    if 'model_type' not in config:
        errors.append("image_encoder_config: 'model_type' is required")
    
    if 'embedding_dim' not in config:
        errors.append("image_encoder_config: 'embedding_dim' is required")


def validate_continuous_encoder_config(config: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Validate continuous encoder configuration"""
    if 'embedding_dim' not in config:
        errors.append("continuous_encoder_config: 'embedding_dim' is required")
    
    # hidden_dims is optional but recommended
    if 'hidden_dims' not in config and 'hidden_dim' not in config:
        warnings.append("continuous_encoder_config: 'hidden_dims' not specified, will use default")


def validate_temporal_encoder_config(config: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Validate temporal encoder configuration"""
    if 'output_dim' not in config:
        errors.append("temporal_encoder_config: 'output_dim' is required")


def validate_config_file(config_path: str) -> ValidationResult:
    """
    Validate configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        ValidationResult
    """
    import json
    import os
    
    if not os.path.exists(config_path):
        return ValidationResult(
            errors=[f"Config file not found: {config_path}"],
            warnings=[]
        )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return validate_config(config)
    except json.JSONDecodeError as e:
        return ValidationResult(
            errors=[f"Invalid JSON in config file: {e}"],
            warnings=[]
        )
    except Exception as e:
        return ValidationResult(
            errors=[f"Error reading config file: {e}"],
            warnings=[]
        )


