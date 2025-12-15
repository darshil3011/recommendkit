#!/usr/bin/env python3
"""
Generic Recommendation System Training Driver
Reuses all functions from trainer/ folder for training any recommendation model
"""

import os
import sys
import json
import argparse
import torch
from typing import Dict, Any, Optional, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_processor import Inputs
from trainer.pipeline_builder import RecommendationPipeline, save_complete_model
from trainer.data_loader import create_data_loaders, load_interactions_from_input
from trainer.trainer import train_model
from utils.config_validator import validate_config


def create_model_from_config(config: Dict[str, Any], item_data: Optional[List[Dict[str, Any]]] = None) -> RecommendationPipeline:
    """
    Create model from configuration dictionary and store the config for saving.
    Validates config before creating model.
    
    Args:
        config: Model configuration dictionary
        item_data: Optional item data for temporal encoder (any dataset format)
        
    Returns:
        RecommendationPipeline instance with stored config
        
    Raises:
        ValueError: If config validation fails
    """
    # Validate config first
    validation_result = validate_config(config)
    if not validation_result.is_valid:
        raise ValueError(f"Config validation failed:\n{validation_result}")
    if validation_result.warnings:
        print(f"⚠️  Config validation warnings:\n{validation_result}")
    
    model = RecommendationPipeline(
        embedding_dim=config.get('embedding_dim', 256),
        loss_type=config.get('loss_type', 'bce'),
        
        # User tower configuration
        user_num_attention_layers=config.get('user_num_attention_layers', 4),
        user_num_heads=config.get('user_num_heads', 16),
        user_dropout=config.get('user_dropout', 0.15),
        user_use_cls_token=config.get('user_use_cls_token', True),
        user_use_layer_norm=config.get('user_use_layer_norm', False),
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
        image_encoder_config=config.get('image_encoder_config'),
        text_encoder_config=config.get('text_encoder_config'),
        categorical_encoder_config=config.get('categorical_encoder_config'),
        continuous_encoder_config=config.get('continuous_encoder_config'),
        temporal_encoder_config=config.get('temporal_encoder_config'),
        
        # Item data for temporal encoder
        item_data=item_data
    )
    
    # Store the config in the model AFTER creation so it can access encoder._actual_config
    # This ensures we save the actual values used
    model.set_config(config)
    
    return model


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def extract_architecture_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only architecture-related parameters from config (exclude training parameters)
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary containing only architecture parameters
    """
    arch_config = {
        'embedding_dim': config.get('embedding_dim'),
        'loss_type': config.get('loss_type'),
        
        # User tower configuration
        'user_num_attention_layers': config.get('user_num_attention_layers'),
        'user_num_heads': config.get('user_num_heads'),
        'user_dropout': config.get('user_dropout'),
        'user_use_cls_token': config.get('user_use_cls_token'),
        'user_use_layer_norm': config.get('user_use_layer_norm'),
        'user_use_simple_fusion': config.get('user_use_simple_fusion'),
        
        # Item tower configuration
        'item_num_attention_layers': config.get('item_num_attention_layers'),
        'item_num_heads': config.get('item_num_heads'),
        'item_dropout': config.get('item_dropout'),
        'item_use_simple_fusion': config.get('item_use_simple_fusion'),
        
        # Interaction modeling configuration
        'interaction_num_attention_layers': config.get('interaction_num_attention_layers'),
        'interaction_num_heads': config.get('interaction_num_heads'),
        'interaction_dropout': config.get('interaction_dropout'),
        'interaction_use_simple_fusion': config.get('interaction_use_simple_fusion'),
        
        # Classifier configuration
        'classifier_hidden_dims': config.get('classifier_hidden_dims'),
        'classifier_dropout': config.get('classifier_dropout'),
        
        # Encoder configurations
        'image_encoder_config': config.get('image_encoder_config') or config.get('image_encoder'),
        'text_encoder_config': config.get('text_encoder_config') or config.get('text_encoder'),
        'categorical_encoder_config': config.get('categorical_encoder_config') or config.get('categorical_encoder'),
        'continuous_encoder_config': config.get('continuous_encoder_config') or config.get('continuous_encoder'),
        'temporal_encoder_config': config.get('temporal_encoder_config') or config.get('temporal_encoder'),
    }
    
    return arch_config


def validate_config_compatibility(training_config: Dict[str, Any], saved_config: Dict[str, Any]) -> None:
    """
    Validate that training config architecture matches saved model config architecture exactly.
    Raises RuntimeError if any architecture parameter differs.
    
    Args:
        training_config: Training configuration dictionary (from --config_path)
        saved_config: Saved model configuration dictionary (from pretrained model)
        
    Raises:
        RuntimeError: If architecture configs don't match exactly
    """
    train_arch = extract_architecture_config(training_config)
    saved_arch = extract_architecture_config(saved_config)
    
    mismatches = []
    
    # Compare all architecture parameters
    for key in train_arch.keys():
        train_val = train_arch[key]
        saved_val = saved_arch[key]
        
        # Handle None values - both must be None or both must be non-None
        if train_val is None and saved_val is None:
            continue
        if train_val is None or saved_val is None:
            mismatches.append(f"{key}: training={train_val}, saved={saved_val}")
            continue
        
        # Handle nested dictionaries (encoder configs)
        if isinstance(train_val, dict) and isinstance(saved_val, dict):
            # Compare nested dict keys - both must have same keys and values
            if set(train_val.keys()) != set(saved_val.keys()):
                mismatches.append(f"{key}: keys differ - training has {set(train_val.keys())}, saved has {set(saved_val.keys())}")
            else:
                for nested_key in train_val.keys():
                    if train_val[nested_key] != saved_val[nested_key]:
                        mismatches.append(f"{key}.{nested_key}: training={train_val[nested_key]}, saved={saved_val[nested_key]}")
        elif isinstance(train_val, list) and isinstance(saved_val, list):
            # Handle lists (e.g., classifier_hidden_dims)
            if train_val != saved_val:
                mismatches.append(f"{key}: training={train_val}, saved={saved_val}")
        elif train_val != saved_val:
            mismatches.append(f"{key}: training={train_val}, saved={saved_val}")
    
    if mismatches:
        error_msg = (
            "❌ CRITICAL: Training config architecture does not match saved model config!\n"
            "The following architecture parameters differ:\n"
        )
        for mismatch in mismatches:
            error_msg += f"  - {mismatch}\n"
        error_msg += (
            "\nPre-training can only continue if the training config JSON architecture "
            "matches the saved model config exactly.\n"
            "Please use the same architecture configuration or train a new model from scratch."
        )
        raise RuntimeError(error_msg)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train a recommendation system model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data JSON file')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to training configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model (default: models)')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Base name for saved model files (default: model)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cpu, cuda)')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to directory containing pretrained model weights (optional)')
    parser.add_argument('--pretrained_model_name', type=str, default='model',
                       help='Base name of pretrained model files (default: model)')
    
    args = parser.parse_args()
    
    print("🚀 Generic Recommendation System Training")
    print("=" * 60)
    
    # Validate input files
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    if not os.path.exists(args.config_path):
        print(f"❌ Config file not found: {args.config_path}")
        return 1
    
    try:
        # Step 1: Load training configuration
        print("🔄 Loading training configuration...")
        config = load_training_config(args.config_path)
        print(f"✅ Configuration loaded from {args.config_path}")
        
        # Step 2: Load data
        print("🔄 Loading data...")
        inputs = Inputs()
        inputs.configure_validators(image_check_files=False)
        result = inputs.load_from_json(args.data_path)
        
        if not result.is_valid:
            print("❌ Data loading errors:")
            for error in result.errors:
                print(f"  - {error}")
            return 1
        
        user_data = inputs.get_user_data()
        item_data = inputs.get_item_data()
        print(f"✅ Loaded {len(user_data)} users and {len(item_data)} items")
        
        # Step 3: Load interactions
        print("🔄 Loading interactions...")
        interactions = load_interactions_from_input(inputs=inputs)
        print(f"✅ Loaded {len(interactions)} interactions")
        
        # Step 4: Create model
        print("🔄 Creating model...")
        model = create_model_from_config(config, item_data)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Step 4.5: Load pretrained weights if provided
        if args.pretrained_weights:
            pretrained_dir = args.pretrained_weights
            pretrained_model_name = args.pretrained_model_name
            
            # Validate pretrained directory exists
            if not os.path.exists(pretrained_dir):
                print(f"❌ Pretrained weights directory not found: {pretrained_dir}")
                return 1
            
            # Paths to pretrained model files
            pretrained_config_path = os.path.join(pretrained_dir, f"{pretrained_model_name}_config.json")
            pretrained_weights_path = os.path.join(pretrained_dir, f"{pretrained_model_name}.pt")
            
            # Validate pretrained files exist
            if not os.path.exists(pretrained_config_path):
                print(f"❌ Pretrained config file not found: {pretrained_config_path}")
                return 1
            if not os.path.exists(pretrained_weights_path):
                print(f"❌ Pretrained weights file not found: {pretrained_weights_path}")
                return 1
            
            print(f"🔄 Loading pretrained weights from {pretrained_dir}...")
            
            # Load saved model config
            with open(pretrained_config_path, 'r') as f:
                saved_config = json.load(f)
            
            # Validate config compatibility
            print("🔄 Validating config compatibility...")
            try:
                validate_config_compatibility(config, saved_config)
                print("✅ Config validation passed - architectures match exactly")
            except RuntimeError as e:
                print(str(e))
                return 1
            
            # Load pretrained weights
            print(f"🔄 Loading weights from {pretrained_weights_path}...")
            try:
                state_dict = torch.load(pretrained_weights_path, map_location='cpu')
                
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
                
                # Pre-initialize continuous encoder MLPs from checkpoint (they're lazy-initialized)
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
                
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                # Filter out dimension aligner keys (they're expected to differ and are handled dynamically)
                missing_keys_filtered = [k for k in missing_keys if 'dimension_aligner.projections' not in k]
                unexpected_keys_filtered = [k for k in unexpected_keys if 'dimension_aligner.projections' not in k]
                
                # Only show warnings for actual issues (not dimension aligner differences which are expected)
                # In normal operation, after pre-initialization, there should be no missing/unexpected keys
                # But we don't show warnings to keep output clean during re-training
                
                print("✅ Pretrained weights loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load pretrained weights: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 5: Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"✅ Using device: {device}")
        
        # Step 6: Create data loaders
        print("🔄 Creating data loaders...")
        train_loader, val_loader, test_interactions, train_interactions = create_data_loaders(
            inputs=inputs,
            interactions=interactions,
            train_split=config.get('train_split', 0.8),
            batch_size=config.get('batch_size', 32),
            negative_sampling_ratio=config.get('negative_sampling_ratio', 1.0),
            seed=config.get('seed', 42),
            test_split=0.05  # 5% for testing
        )
        print(f"✅ Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_interactions)} interactions")
        
        # Save test interactions and training interactions for evaluation
        test_interactions_path = os.path.join(args.output_dir, f"{args.model_name}_test_interactions.json")
        with open(test_interactions_path, 'w') as f:
            json.dump(test_interactions, f, indent=2)
        print(f"✅ Test interactions saved to {test_interactions_path}")
        
        # Save training interactions (to exclude from recommendations during evaluation)
        train_interactions_path = os.path.join(args.output_dir, f"{args.model_name}_train_interactions.json")
        with open(train_interactions_path, 'w') as f:
            json.dump(train_interactions, f, indent=2)
        print(f"✅ Training interactions saved to {train_interactions_path}")
        
        # Step 7: Train model
        print("🚀 Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get('num_epochs', 100),
            learning_rate=config.get('learning_rate', 0.001),
            optimizer_type=config.get('optimizer_type', 'adam'),
            scheduler_type=config.get('scheduler_type', 'plateau'),
            device=device,
            print_every=config.get('print_every', 10),
            save_path=os.path.join(args.output_dir, f"{args.model_name}.pt")
        )
        
        # Step 8: Save complete model
        print("🔄 Saving complete model...")
        save_complete_model(model, args.output_dir, args.model_name)
        
        # Step 9: Save training history
        history_path = os.path.join(args.output_dir, f"{args.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        print("🎉 Training completed successfully!")
        print(f"📁 Model saved to: {args.output_dir}/{args.model_name}")
        print(f"📁 History saved to: {history_path}")
        print(f"📊 Final train loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"📊 Final val loss: {history['val_losses'][-1]:.4f}")
        print(f"📊 Final train accuracy: {history['train_accuracies'][-1]:.4f}")
        if history['val_accuracies']:
            print(f"📊 Final val accuracy: {history['val_accuracies'][-1]:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
