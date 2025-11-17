#!/usr/bin/env python3
"""
Diagnostic script to check what encoders are actually outputting
Run this BEFORE training to verify your data pipeline
"""

import torch
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_processor import Inputs
from trainer.pipeline_builder import RecommendationPipeline
from trainer.data_loader import create_data_loaders, load_interactions_from_input

def diagnose_encoders(config_path: str, data_path: str):
    """Diagnose what each encoder is producing"""
    
    print("🔍 ENCODER DIAGNOSTIC CHECK")
    print("=" * 80)
    
    # Load config and data
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    inputs = Inputs()
    inputs.configure_validators(image_check_files=False)
    result = inputs.load_from_json(data_path)
    
    if not result.is_valid:
        print("❌ Data loading failed")
        return
    
    user_data = inputs.get_user_data()
    item_data = inputs.get_item_data()
    interactions = load_interactions_from_input(inputs)
    
    # Create model
    model = RecommendationPipeline(
        embedding_dim=config['embedding_dim'],
        loss_type=config['loss_type'],
        user_num_attention_layers=config.get('user_num_attention_layers', 0),
        user_num_heads=config.get('user_num_heads', 1),
        user_dropout=config.get('user_dropout', 0.0),
        item_num_attention_layers=config.get('item_num_attention_layers', 0),
        item_num_heads=config.get('item_num_heads', 1),
        item_dropout=config.get('item_dropout', 0.0),
        interaction_num_attention_layers=config.get('interaction_num_attention_layers', 0),
        interaction_num_heads=config.get('interaction_num_heads', 1),
        interaction_dropout=config.get('interaction_dropout', 0.0),
        classifier_hidden_dims=config.get('classifier_hidden_dims', [64]),
        classifier_dropout=config.get('classifier_dropout', 0.0),
        text_encoder_config=config.get('text_encoder_config'),
        categorical_encoder_config=config.get('categorical_encoder_config'),
        continuous_encoder_config=config.get('continuous_encoder_config'),
        temporal_encoder_config=config.get('temporal_encoder_config'),
        item_data=item_data
    )
    
    model.eval()
    
    # Get one batch
    train_loader, _ = create_data_loaders(
        inputs=inputs,
        interactions=interactions,
        train_split=1.0,
        batch_size=4,
        negative_sampling_ratio=config.get('negative_sampling_ratio', 1.0),
        seed=42
    )
    
    batch = next(iter(train_loader))
    
    print("\n📊 BATCH COMPOSITION")
    print(f"   Batch size: {len(batch['labels'])}")
    print(f"   User IDs: {batch['user_ids']}")
    print(f"   Item IDs: {batch['item_ids']}")
    print(f"   Labels: {batch['labels'].tolist()}")
    
    # Check what features each user/item actually has in the batch
    print("\n🔍 RAW FEATURES IN BATCH")
    print("\nUser features:")
    for feature_type, feature_data in batch['user_features'].items():
        print(f"   {feature_type}: {list(feature_data.keys())}")
        # Show first user's data
        for field_name, field_values in feature_data.items():
            first_value = field_values[0] if isinstance(field_values, list) else field_values
            print(f"      {field_name}: {first_value}")
    
    print("\nItem features:")
    for feature_type, feature_data in batch['item_features'].items():
        print(f"   {feature_type}: {list(feature_data.keys())}")
        for field_name, field_values in feature_data.items():
            first_value = field_values[0] if isinstance(field_values, list) else field_values
            print(f"      {field_name}: {first_value}")
    
    # Now encode and check outputs
    print("\n🔬 ENCODER OUTPUTS")
    
    with torch.no_grad():
        # Encode user features
        user_encoded = model._encode_features(batch['user_features'], model.user_dimension_aligner)
        
        print("\nUser encodings:")
        for feature_name, embedding in user_encoded.items():
            print(f"   {feature_name}:")
            print(f"      Shape: {embedding.shape}")
            print(f"      Mean: {embedding.mean().item():.6f}")
            print(f"      Std: {embedding.std().item():.6f}")
            print(f"      Min: {embedding.min().item():.6f}")
            print(f"      Max: {embedding.max().item():.6f}")
            
            # Check if it's all zeros or near-zeros
            if embedding.abs().max() < 1e-6:
                print(f"      ⚠️  WARNING: Embedding is essentially zero!")
            
            # Check if it's constant across batch
            if embedding.std(dim=0).mean() < 1e-6:
                print(f"      ⚠️  WARNING: Embedding is constant across batch!")
        
        # Encode item features
        item_encoded = model._encode_features(batch['item_features'], model.item_dimension_aligner)
        
        print("\nItem encodings:")
        for feature_name, embedding in item_encoded.items():
            print(f"   {feature_name}:")
            print(f"      Shape: {embedding.shape}")
            print(f"      Mean: {embedding.mean().item():.6f}")
            print(f"      Std: {embedding.std().item():.6f}")
            print(f"      Min: {embedding.min().item():.6f}")
            print(f"      Max: {embedding.max().item():.6f}")
            
            if embedding.abs().max() < 1e-6:
                print(f"      ⚠️  WARNING: Embedding is essentially zero!")
            
            if embedding.std(dim=0).mean() < 1e-6:
                print(f"      ⚠️  WARNING: Embedding is constant across batch!")
        
        # Check fusion outputs
        print("\n🔀 FUSION OUTPUTS")
        
        user_feature_types = model._get_feature_types(user_encoded)
        item_feature_types = model._get_feature_types(item_encoded)
        
        user_fused = model.user_generator(user_encoded, user_feature_types)
        item_fused = model.item_generator(item_encoded, item_feature_types)
        
        print(f"\nUser fused embedding:")
        print(f"   Shape: {user_fused.shape}")
        print(f"   Mean: {user_fused.mean().item():.6f}")
        print(f"   Std: {user_fused.std().item():.6f}")
        if user_fused.abs().max() < 1e-6:
            print(f"   ⚠️  WARNING: Fused embedding is essentially zero!")
        
        print(f"\nItem fused embedding:")
        print(f"   Shape: {item_fused.shape}")
        print(f"   Mean: {item_fused.mean().item():.6f}")
        print(f"   Std: {item_fused.std().item():.6f}")
        if item_fused.abs().max() < 1e-6:
            print(f"   ⚠️  WARNING: Fused embedding is essentially zero!")
        
        # Check interaction output
        print("\n🤝 INTERACTION OUTPUT")
        
        interaction_emb = model.interaction_generator(user_fused, item_fused)
        
        print(f"   Shape: {interaction_emb.shape}")
        print(f"   Mean: {interaction_emb.mean().item():.6f}")
        print(f"   Std: {interaction_emb.std().item():.6f}")
        if interaction_emb.abs().max() < 1e-6:
            print(f"   ⚠️  WARNING: Interaction embedding is essentially zero!")
        
        # Check final predictions
        print("\n🎯 FINAL PREDICTIONS")
        
        loss, logits = model(batch['user_features'], batch['item_features'], batch['labels'])
        probs = torch.sigmoid(logits)
        
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Logits: {logits.squeeze().tolist()}")
        print(f"   Probabilities: {probs.squeeze().tolist()}")
        print(f"   True labels: {batch['labels'].tolist()}")
        
        # Check if predictions are all the same
        if probs.std() < 0.01:
            print(f"   ⚠️  WARNING: All predictions are essentially the same!")
            print(f"   ⚠️  Model is not learning any signal from the data!")
    
    print("\n" + "=" * 80)
    print("✅ Diagnostic complete!")
    
    # CRITICAL TEST: Check if categorical embeddings differ between different values
    print("\n🔬 CRITICAL TEST: Categorical Encoder Differentiation")
    print("Testing if categorical encoder produces different outputs for different inputs...")
    
    with torch.no_grad():
        # Test with explicit different categorical values
        test_cases = [
            {'occupation': 'software_engineer', 'location': 'USA'},
            {'occupation': 'chef', 'location': 'India'},
            {'occupation': 'doctor', 'location': 'Canada'},
        ]
        
        embeddings = []
        for test_case in test_cases:
            result = model.categorical_encoder(test_case)
            embeddings.append(result['categorical_features'])
            print(f"\n   Input: {test_case}")
            print(f"   Output mean: {result['categorical_features'].mean().item():.6f}")
            print(f"   Output std: {result['categorical_features'].std().item():.6f}")
        
        # Check if embeddings are different
        emb1, emb2, emb3 = embeddings
        diff_1_2 = (emb1 - emb2).abs().mean().item()
        diff_1_3 = (emb1 - emb3).abs().mean().item()
        diff_2_3 = (emb2 - emb3).abs().mean().item()
        
        print(f"\n   Difference between embeddings:")
        print(f"   software_engineer vs chef: {diff_1_2:.6f}")
        print(f"   software_engineer vs doctor: {diff_1_3:.6f}")
        print(f"   chef vs doctor: {diff_2_3:.6f}")
        
        if max(diff_1_2, diff_1_3, diff_2_3) < 1e-4:
            print(f"   ❌ CRITICAL ISSUE: Categorical encoder produces identical outputs!")
            print(f"   ❌ This means the model cannot distinguish between different categories!")
        else:
            print(f"   ✅ Categorical encoder is producing different outputs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--data', required=True, help='Path to data JSON')
    args = parser.parse_args()
    
    diagnose_encoders(args.config, args.data)