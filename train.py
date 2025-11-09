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


def create_model_from_config(config: Dict[str, Any], item_data: Optional[List[Dict[str, Any]]] = None) -> RecommendationPipeline:
    """
    Create model from configuration dictionary
    
    Args:
        config: Model configuration dictionary
        item_data: Optional item data for temporal encoder (any dataset format)
        
    Returns:
        RecommendationPipeline instance
    """
    return RecommendationPipeline(
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
        
        # Step 5: Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"✅ Using device: {device}")
        
        # Step 6: Create data loaders
        print("🔄 Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            inputs=inputs,
            interactions=interactions,
            train_split=config.get('train_split', 0.8),
            batch_size=config.get('batch_size', 32),
            negative_sampling_ratio=config.get('negative_sampling_ratio', 1.0),
            seed=config.get('seed', 42)
        )
        print(f"✅ Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
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
