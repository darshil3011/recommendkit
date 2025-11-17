#!/usr/bin/env python3
"""
Generic Recommendation System Inference Driver
Provides inference capabilities for any trained recommendation model
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_processor import Inputs
from trainer.pipeline_builder import load_model_from_config


def generate_user_embedding(model, user_data, encoders_used=None, expected_num_features=None):
    """Generate embedding for a single user
    
    Args:
        model: Trained model
        user_data: User data dictionary
        encoders_used: Dict indicating which encoders were used during training (optional)
        expected_num_features: Expected number of features (if None, use all available)
    """
    model.eval()
    with torch.no_grad():
        # Prepare user features (only include features for encoders that were used during training)
        user_features = {}
        
        # If encoders_used is provided, use it to filter features
        # Otherwise, check if encoders exist in the model
        if encoders_used is not None:
            # Only include features for encoders that were actually used during training
            # CRITICAL: Always include features if encoder was used, even if data is missing
            # Encoders will return default embeddings for missing data, maintaining feature count
            if encoders_used.get('categorical', False):
                # Always include if encoder was used (encoder handles missing data)
                user_features['categorical'] = user_data.get('categorical', {})
            
            if encoders_used.get('continuous', False):
                # Always include if encoder was used (encoder handles missing data)
                user_features['continuous'] = user_data.get('continuous', {})
            
            if encoders_used.get('temporal', False):
                # Always include if encoder was used (encoder handles missing data)
                user_features['temporal'] = user_data.get('temporal', {})
            
            if encoders_used.get('text', False):
                # Always include if encoder was used (encoder handles missing data)
                user_features['text'] = user_data.get('text', {})
            
            if encoders_used.get('image', False):
                # Always include if encoder was used (encoder handles missing data with default embedding)
                user_features['image'] = user_data.get('image', {})
            
            # If we have more features than expected, exclude empty features first, then by priority
            if expected_num_features is not None and len(user_features) > expected_num_features:
                # Helper to check if feature data is non-empty
                def is_non_empty(feature_type):
                    data = user_features.get(feature_type, {})
                    if not data:
                        return False
                    if isinstance(data, dict):
                        return len(data) > 0
                    return True
                
                # Priority order: categorical > continuous > text > temporal > image
                # (exclude least important first)
                priority_order = ['image', 'temporal', 'text', 'continuous', 'categorical']
                
                # First, collect features with data vs without data
                features_with_data = [ft for ft in user_features.keys() if is_non_empty(ft)]
                features_without_data = [ft for ft in user_features.keys() if not is_non_empty(ft)]
                
                # CRITICAL: Always exclude ALL empty features first
                # Only exclude non-empty features if we still have too many after excluding empty ones
                selected = set(features_with_data)  # Start with all non-empty features
                
                # If we still have too many non-empty features, exclude by priority
                if len(selected) > expected_num_features:
                    # Sort non-empty features by priority (keep most important)
                    sorted_features = sorted(selected, key=lambda x: priority_order.index(x) if x in priority_order else 999)
                    selected = set(sorted_features[:expected_num_features])
                elif len(selected) < expected_num_features:
                    # We need more features, fill with empty features (excluding by priority)
                    remaining_slots = expected_num_features - len(selected)
                    for ft in priority_order:
                        if ft in features_without_data and remaining_slots > 0:
                            selected.add(ft)
                            remaining_slots -= 1
                
                user_features = {ft: user_features[ft] for ft in selected}
        else:
            # Fallback: check if encoders exist in the model
            if hasattr(model, 'categorical_encoder') and model.categorical_encoder is not None:
                if user_data.get('categorical'):
                    user_features['categorical'] = user_data['categorical']
            
            if hasattr(model, 'user_continuous_encoder') and model.user_continuous_encoder is not None:
                if user_data.get('continuous'):
                    user_features['continuous'] = user_data['continuous']
            
            if hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
                if user_data.get('temporal'):
                    user_features['temporal'] = user_data['temporal']
            
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                if user_data.get('text'):
                    user_features['text'] = user_data['text']
            
            if hasattr(model, 'image_encoder') and model.image_encoder is not None:
                if user_data.get('image'):
                    user_features['image'] = user_data['image']
        
        # Encode user features
        user_encoded = model._encode_features(user_features, model.user_dimension_aligner)
        user_feature_types = model._get_feature_types(user_encoded)
        
        # Debug: Print what features are being encoded
        print(f"🔍 Debug: Encoded user features: {list(user_encoded.keys())}")
        for key, val in user_encoded.items():
            if isinstance(val, torch.Tensor):
                print(f"   {key}: shape {val.shape}")
        
        # Generate user embedding
        user_embedding = model.user_generator(user_encoded, user_feature_types)
        
        return user_embedding.squeeze().cpu().numpy()


def generate_item_embedding(model, item_data, encoders_used=None, expected_num_features=None):
    """Generate embedding for a single item
    
    Args:
        model: Trained model
        item_data: Item data dictionary
        encoders_used: Dict indicating which encoders were used during training (optional)
        expected_num_features: Expected number of features (if None, use all available)
    """
    model.eval()
    with torch.no_grad():
        # Prepare item features (only include features for encoders that were used during training)
        item_features = {}
        
        # If encoders_used is provided, use it to filter features
        # Otherwise, check if encoders exist in the model
        if encoders_used is not None:
            # Only include features for encoders that were actually used during training
            # CRITICAL: Always include features if encoder was used, even if data is missing
            # Encoders will return default embeddings for missing data, maintaining feature count
            if encoders_used.get('categorical', False):
                # Always include if encoder was used (encoder handles missing data)
                item_features['categorical'] = item_data.get('categorical', {})
            
            if encoders_used.get('continuous', False):
                # Always include if encoder was used (encoder handles missing data)
                item_features['continuous'] = item_data.get('continuous', {})
            
            if encoders_used.get('text', False):
                # Always include if encoder was used (encoder handles missing data)
                item_features['text'] = item_data.get('text', {})
            
            if encoders_used.get('image', False):
                # Always include if encoder was used (encoder handles missing data with default embedding)
                item_features['image'] = item_data.get('image', {})
            
            # Note: items typically don't have temporal features, but check anyway
            if encoders_used.get('temporal', False):
                # Always include if encoder was used (encoder handles missing data)
                item_features['temporal'] = item_data.get('temporal', {})
            
            # If we have more features than expected, exclude empty features first, then by priority
            if expected_num_features is not None and len(item_features) > expected_num_features:
                # Helper to check if feature data is non-empty
                def is_non_empty(feature_type):
                    data = item_features.get(feature_type, {})
                    if not data:
                        return False
                    if isinstance(data, dict):
                        return len(data) > 0
                    return True
                
                # Priority order: categorical > continuous > text > image > temporal
                # (exclude least important first, temporal is least common for items)
                priority_order = ['temporal', 'image', 'text', 'continuous', 'categorical']
                
                # First, collect features with data
                features_with_data = [ft for ft in item_features.keys() if is_non_empty(ft)]
                features_without_data = [ft for ft in item_features.keys() if not is_non_empty(ft)]
                
                # CRITICAL: Always exclude ALL empty features first
                # Only exclude non-empty features if we still have too many after excluding empty ones
                selected = set(features_with_data)  # Start with all non-empty features
                
                # If we still have too many non-empty features, exclude by priority
                if len(selected) > expected_num_features:
                    # Sort non-empty features by priority (keep most important)
                    sorted_features = sorted(selected, key=lambda x: priority_order.index(x) if x in priority_order else 999)
                    selected = set(sorted_features[:expected_num_features])
                elif len(selected) < expected_num_features:
                    # We need more features, fill with empty features (excluding by priority)
                    remaining_slots = expected_num_features - len(selected)
                    for ft in priority_order:
                        if ft in features_without_data and remaining_slots > 0:
                            selected.add(ft)
                            remaining_slots -= 1
                
                item_features = {ft: item_features[ft] for ft in selected}
        else:
            # Fallback: check if encoders exist in the model
            if hasattr(model, 'categorical_encoder') and model.categorical_encoder is not None:
                if item_data.get('categorical'):
                    item_features['categorical'] = item_data['categorical']
            
            if hasattr(model, 'item_continuous_encoder') and model.item_continuous_encoder is not None:
                if item_data.get('continuous'):
                    item_features['continuous'] = item_data['continuous']
            
            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                if item_data.get('text'):
                    item_features['text'] = item_data['text']
            
            if hasattr(model, 'image_encoder') and model.image_encoder is not None:
                if item_data.get('image'):
                    item_features['image'] = item_data['image']
            
            if hasattr(model, 'temporal_encoder') and model.temporal_encoder is not None:
                if item_data.get('temporal'):
                    item_features['temporal'] = item_data['temporal']
        
        # Encode item features
        item_encoded = model._encode_features(item_features, model.item_dimension_aligner)
        item_feature_types = model._get_feature_types(item_encoded)
        
        # Debug: Print what features are being encoded (only for first item to avoid spam)
        if not hasattr(generate_item_embedding, '_debug_printed'):
            print(f"🔍 Debug: Item features to encode: {list(item_features.keys())}")
            print(f"🔍 Debug: Encoded item features: {list(item_encoded.keys())}")
            for key, val in item_encoded.items():
                if isinstance(val, torch.Tensor):
                    print(f"   {key}: shape {val.shape}")
            generate_item_embedding._debug_printed = True
        
        # Generate item embedding
        item_embedding = model.item_generator(item_encoded, item_feature_types)
        
        return item_embedding.squeeze().cpu().numpy()


def compute_similarity(user_embedding, item_embedding):
    """Compute cosine similarity between user and item embeddings"""
    # Flatten embeddings to 1D
    user_flat = user_embedding.flatten()
    item_flat = item_embedding.flatten()
    
    # Normalize embeddings
    user_norm = user_flat / np.linalg.norm(user_flat)
    item_norm = item_flat / np.linalg.norm(item_flat)
    
    # Compute cosine similarity
    similarity = np.dot(user_norm, item_norm)
    return similarity


def find_top_items_for_user(model, user_data, all_items, k=10, filters=None, encoders_used=None, expected_user_features=None, expected_item_features=None):
    """
    Find top-k items for a user with optional filtering
    
    Uses efficient embedding-based similarity computation for fast inference.
    
    Args:
        model: Trained model
        user_data: User data dictionary
        all_items: List of all items
        k: Number of top items to return
        filters: Dictionary of filters to apply
        encoders_used: Dict indicating which encoders were used during training
        expected_user_features: Expected number of user features
        expected_item_features: Expected number of item features
        
    Returns:
        List of top-k items with similarity scores
    """
    print(f"🔄 Finding top {k} items for user...")
    
    # Generate user embedding ONCE
    user_embedding = generate_user_embedding(model, user_data, encoders_used=encoders_used, expected_num_features=expected_user_features)
    
    # Compute similarities with all items
    similarities = []
    
    with torch.no_grad():
        for item in all_items:
            # Apply filters if provided
            if filters:
                filter_passed = True
                
                # Apply categorical filters
                if 'categorical' in filters and 'categorical' in item:
                    for field_name, field_value in filters['categorical'].items():
                        if field_name in item['categorical'] and item['categorical'][field_name] != field_value:
                            filter_passed = False
                            break
                
                # Apply continuous filters
                if filter_passed and 'continuous' in filters and 'continuous' in item:
                    for field_name, filter_range in filters['continuous'].items():
                        if field_name in item['continuous']:
                            item_value = item['continuous'][field_name]
                            if filter_range.get('min') and item_value < filter_range['min']:
                                filter_passed = False
                                break
                            if filter_range.get('max') and item_value > filter_range['max']:
                                filter_passed = False
                                break
                
                if not filter_passed:
                    continue
            
            # Generate item embedding (pass encoders_used to ensure correct features)
            item_embedding = generate_item_embedding(model, item, encoders_used=encoders_used, expected_num_features=expected_item_features)
            
            # Compute similarity
            similarity = compute_similarity(user_embedding, item_embedding)
            
            similarities.append({
                'item': item,
                'similarity': similarity
            })
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:k]


def print_recommendations(recommendations, title="Recommendations"):
    """Print formatted recommendations"""
    print(f"\n🎯 {title}")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        # Get item title from text fields or use item_id as fallback
        title_text = "Unknown"
        if 'text' in rec['item']:
            # Try common title fields
            for title_field in ['title', 'name', 'description']:
                if title_field in rec['item']['text']:
                    title_text = rec['item']['text'][title_field]
                    break
        
        if title_text == "Unknown":
            title_text = rec['item'].get('item_id', 'Unknown')
        
        print(f"{i:2d}. {title_text}")
        print(f"     Score: {rec['similarity']:.4f}")
        print(f"     Item ID: {rec['item']['item_id']}")
        
        # Show categorical features if available
        if 'categorical' in rec['item'] and rec['item']['categorical']:
            cat_info = []
            for key, value in rec['item']['categorical'].items():
                cat_info.append(f"{key}: {value}")
            if cat_info:
                print(f"     Categories: {' | '.join(cat_info)}")
        
        # Show continuous features if available
        if 'continuous' in rec['item'] and rec['item']['continuous']:
            cont_info = []
            for key, value in rec['item']['continuous'].items():
                cont_info.append(f"{key}: {value}")
            if cont_info:
                print(f"     Features: {' | '.join(cont_info)}")
        
        print()


def main():
    """Main inference pipeline"""
    parser = argparse.ArgumentParser(description='Run inference with a trained recommendation model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data JSON file')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing model files (default: models)')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Base name of model files (default: model)')
    parser.add_argument('--user_id', type=str, default=None,
                       help='Specific user ID to get recommendations for (default: use first user)')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations to generate (default: 10)')
    parser.add_argument('--filters', type=str, default=None,
                       help='JSON string of filters to apply (e.g., \'{"categorical": {"genre": "Action"}}\')')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save recommendations to (default: print to console)')
    
    args = parser.parse_args()
    
    print("🚀 Generic Recommendation System - Inference")
    print("=" * 60)
    
    # Validate input files
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    model_path = os.path.join(args.model_dir, f"{args.model_name}.pt")
    config_path = os.path.join(args.model_dir, f"{args.model_name}_config.json")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    try:
        # Step 1: Load data
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
        
        # Step 2: Load trained model
        print("🔄 Loading trained model...")
        model = load_model_from_config(config_path, model_path, item_data)
        model.eval()
        
        # Check which encoders were actually used during training (have weights in checkpoint)
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint_keys = set(checkpoint.keys())
        
        encoders_used = {
            'image': any('image_encoder' in k for k in checkpoint_keys),
            'text': any('text_encoder' in k for k in checkpoint_keys),
            'categorical': any('categorical_encoder' in k for k in checkpoint_keys),
            'continuous': any('continuous_encoder' in k or 'user_continuous_encoder' in k for k in checkpoint_keys),
            'temporal': any('temporal_encoder' in k for k in checkpoint_keys)
        }
        print(f"🔍 Encoders used during training: {[k for k, v in encoders_used.items() if v]}")
        
        # Determine expected number of features for user and item fusion from checkpoint
        user_fusion_keys = [k for k in checkpoint_keys if 'user_generator.user_fusion.projection.0.weight' in k]
        item_fusion_keys = [k for k in checkpoint_keys if 'item_generator.item_fusion.projection.0.weight' in k]
        
        user_expected_features = None
        item_expected_features = None
        
        if user_fusion_keys:
            user_input_dim = checkpoint[user_fusion_keys[0]].shape[1]
            user_expected_features = user_input_dim // model.embedding_dim
            print(f"🔍 User fusion expects {user_expected_features} features (input_dim={user_input_dim})")
        
        if item_fusion_keys:
            item_input_dim = checkpoint[item_fusion_keys[0]].shape[1]
            item_expected_features = item_input_dim // model.embedding_dim
            print(f"🔍 Item fusion expects {item_expected_features} features (input_dim={item_input_dim})")
        
        print(f"✅ Model loaded successfully")
        print(f"   Model directory: {args.model_dir}")
        print(f"   Model name: {args.model_name}")
        print(f"   Embedding dimension: {model.embedding_dim}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Step 3: Select user
        if args.user_id:
            # Find specific user
            target_user = None
            user_id_to_find = int(args.user_id)  # Convert string to int
            for user in user_data:
                if user.get('user_id') == user_id_to_find:
                    target_user = user
                    break
            if target_user is None:
                print(f"❌ User {args.user_id} not found in data")
                return 1
        else:
            # Use first user as default
            if not user_data:
                print("❌ No user data available")
                return 1
            target_user = user_data[0]
        
        print(f"👤 Target user: {target_user.get('user_id', 'Unknown')}")
        if 'categorical' in target_user:
            print(f"   Categorical features: {list(target_user['categorical'].keys())}")
        if 'continuous' in target_user:
            print(f"   Continuous features: {list(target_user['continuous'].keys())}")
        if 'text' in target_user:
            print(f"   Text features: {list(target_user['text'].keys())}")
        
        # Step 4: Parse filters
        filters = None
        if args.filters:
            try:
                filters = json.loads(args.filters)
                print(f"🔍 Applied filters: {filters}")
            except json.JSONDecodeError as e:
                print(f"❌ Invalid filters JSON: {e}")
                return 1
        
        # Step 5: Generate recommendations
        print(f"🔄 Generating {args.k} recommendations...")
        # Pass encoders_used and expected feature counts to ensure correct feature matching
        recommendations = find_top_items_for_user(
            model, target_user, item_data, k=args.k, filters=filters, 
            encoders_used=encoders_used, 
            expected_user_features=user_expected_features,
            expected_item_features=item_expected_features
        )
        
        # Step 6: Output results
        if args.output_file:
            # Save to file
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                return obj
            
            output_data = {
                'user_id': target_user.get('user_id'),
                'recommendations': [
                    {
                        'item_id': rec['item']['item_id'],
                        'similarity': float(rec['similarity']),  # Convert numpy float32 to Python float
                        'item_data': convert_to_native(rec['item'])
                    }
                    for rec in recommendations
                ]
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Recommendations saved to {args.output_file}")
        else:
            # Print to console
            print_recommendations(recommendations, f"Top {args.k} Recommendations")
        
        print("🎉 Inference completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Model loaded and working")
        print("   ✅ User embeddings generated")
        print("   ✅ Item embeddings generated")
        print("   ✅ Similarity computation working")
        print("   ✅ Top-k recommendations generated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

