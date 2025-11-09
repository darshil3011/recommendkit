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


def generate_user_embedding(model, user_data):
    """Generate embedding for a single user"""
    with torch.no_grad():
        # Encode user features
        user_encoded = model._encode_features(user_data, model.user_dimension_aligner)
        user_feature_types = model._get_feature_types(user_encoded)
        
        # Generate user embedding
        user_embedding = model.user_generator(user_encoded, user_feature_types)
        
        return user_embedding.cpu().numpy()


def generate_item_embedding(model, item_data):
    """Generate embedding for a single item"""
    with torch.no_grad():
        # Encode item features
        item_encoded = model._encode_features(item_data, model.item_dimension_aligner)
        item_feature_types = model._get_feature_types(item_encoded)
        
        # Generate item embedding
        item_embedding = model.item_generator(item_encoded, item_feature_types)
        
        return item_embedding.cpu().numpy()


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


def find_top_items_for_user(model, user_data, all_items, k=10, filters=None):
    """
    Find top-k items for a user with optional filtering
    
    Args:
        model: Trained model
        user_data: User data dictionary
        all_items: List of all items
        k: Number of top items to return
        filters: Dictionary of filters to apply
        
    Returns:
        List of top-k items with similarity scores
    """
    print(f"🔄 Finding top {k} items for user...")
    
    # Generate user embedding
    user_embedding = generate_user_embedding(model, user_data)
    
    # Compute similarities with all items
    similarities = []
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
        
        # Generate item embedding
        item_embedding = generate_item_embedding(model, item)
        
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
        recommendations = find_top_items_for_user(
            model, target_user, item_data, k=args.k, filters=filters
        )
        
        # Step 6: Output results
        if args.output_file:
            # Save to file
            output_data = {
                'user_id': target_user.get('user_id'),
                'recommendations': [
                    {
                        'item_id': rec['item']['item_id'],
                        'similarity': rec['similarity'],
                        'item_data': rec['item']
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

