#!/usr/bin/env python3
"""
Evaluation script for recommendation system
Computes Precision@K and Recall@K on test split using cosine similarity
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_processor import Inputs
from trainer.pipeline_builder import load_model_from_config
from evaluation.ranking_metrics import compute_metrics_for_user


def generate_user_embedding(model, user_data, encoders_used=None, expected_num_features=None):
    """Generate embedding for a single user (copied from inference.py for modularity)"""
    model.eval()
    with torch.no_grad():
        user_features = {}
        
        if encoders_used is not None:
            if encoders_used.get('categorical', False):
                user_features['categorical'] = user_data.get('categorical', {})
            if encoders_used.get('continuous', False):
                user_features['continuous'] = user_data.get('continuous', {})
            if encoders_used.get('temporal', False):
                user_features['temporal'] = user_data.get('temporal', {})
            if encoders_used.get('text', False):
                user_features['text'] = user_data.get('text', {})
            if encoders_used.get('image', False):
                user_features['image'] = user_data.get('image', {})
            
            if expected_num_features is not None and len(user_features) > expected_num_features:
                def is_non_empty(feature_type):
                    data = user_features.get(feature_type, {})
                    if not data:
                        return False
                    if isinstance(data, dict):
                        return len(data) > 0
                    return True
                
                priority_order = ['image', 'temporal', 'text', 'continuous', 'categorical']
                features_with_data = [ft for ft in user_features.keys() if is_non_empty(ft)]
                features_without_data = [ft for ft in user_features.keys() if not is_non_empty(ft)]
                selected = set(features_with_data)
                
                if len(selected) > expected_num_features:
                    sorted_features = sorted(selected, key=lambda x: priority_order.index(x) if x in priority_order else 999)
                    selected = set(sorted_features[:expected_num_features])
                elif len(selected) < expected_num_features:
                    remaining_slots = expected_num_features - len(selected)
                    for ft in priority_order:
                        if ft in features_without_data and remaining_slots > 0:
                            selected.add(ft)
                            remaining_slots -= 1
                
                user_features = {ft: user_features[ft] for ft in selected}
        else:
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
        
        user_encoded = model._encode_features(user_features, model.user_dimension_aligner)
        user_feature_types = model._get_feature_types(user_encoded)
        user_embedding = model.user_generator(user_encoded, user_feature_types)
        
        return user_embedding.squeeze().cpu().numpy()


def generate_item_embedding(model, item_data, encoders_used=None, expected_num_features=None):
    """Generate embedding for a single item (copied from inference.py for modularity)"""
    model.eval()
    with torch.no_grad():
        item_features = {}
        
        if encoders_used is not None:
            if encoders_used.get('categorical', False):
                item_features['categorical'] = item_data.get('categorical', {})
            if encoders_used.get('continuous', False):
                item_features['continuous'] = item_data.get('continuous', {})
            if encoders_used.get('text', False):
                item_features['text'] = item_data.get('text', {})
            if encoders_used.get('image', False):
                item_features['image'] = item_data.get('image', {})
            if encoders_used.get('temporal', False):
                item_features['temporal'] = item_data.get('temporal', {})
            
            if expected_num_features is not None and len(item_features) > expected_num_features:
                def is_non_empty(feature_type):
                    data = item_features.get(feature_type, {})
                    if not data:
                        return False
                    if isinstance(data, dict):
                        return len(data) > 0
                    return True
                
                priority_order = ['temporal', 'image', 'text', 'continuous', 'categorical']
                features_with_data = [ft for ft in item_features.keys() if is_non_empty(ft)]
                features_without_data = [ft for ft in item_features.keys() if not is_non_empty(ft)]
                selected = set(features_with_data)
                
                if len(selected) > expected_num_features:
                    sorted_features = sorted(selected, key=lambda x: priority_order.index(x) if x in priority_order else 999)
                    selected = set(sorted_features[:expected_num_features])
                elif len(selected) < expected_num_features:
                    remaining_slots = expected_num_features - len(selected)
                    for ft in priority_order:
                        if ft in features_without_data and remaining_slots > 0:
                            selected.add(ft)
                            remaining_slots -= 1
                
                item_features = {ft: item_features[ft] for ft in selected}
        else:
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
        
        item_encoded = model._encode_features(item_features, model.item_dimension_aligner)
        item_feature_types = model._get_feature_types(item_encoded)
        item_embedding = model.item_generator(item_encoded, item_feature_types)
        
        return item_embedding.squeeze().cpu().numpy()


def compute_cosine_similarity(user_embedding: np.ndarray, item_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between user embedding and all item embeddings (vectorized)
    
    Args:
        user_embedding: User embedding vector (1D array)
        item_embeddings: Item embeddings matrix (N x embedding_dim)
        
    Returns:
        Array of cosine similarities (N,)
    """
    # Normalize user embedding
    user_norm = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
    
    # Normalize item embeddings
    item_norms = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarities
    similarities = np.dot(item_norms, user_norm)
    
    return similarities


def load_interactions(file_path: str) -> List[Tuple[int, int, int]]:
    """
    Load interactions from JSON file
    
    Args:
        file_path: Path to JSON file containing list of [user_id, item_id, label] tuples
        
    Returns:
        List of (user_id, item_id, label) tuples
    """
    with open(file_path, 'r') as f:
        interactions = json.load(f)
    
    # Convert to list of tuples
    return [tuple(interaction) for interaction in interactions]


def build_user_item_mappings(interactions: List[Tuple[int, int, int]]) -> Dict[int, Set[int]]:
    """
    Build mapping from user_id to set of item_ids they interacted with
    
    Args:
        interactions: List of (user_id, item_id, label) tuples
        
    Returns:
        Dictionary mapping user_id to set of item_ids
    """
    user_items = defaultdict(set)
    for user_id, item_id, label in interactions:
        user_items[user_id].add(item_id)
    return dict(user_items)


def build_user_relevant_items(interactions: List[Tuple[int, int, int]]) -> Dict[int, Set[int]]:
    """
    Build mapping from user_id to set of relevant item_ids (where label=1)
    
    Args:
        interactions: List of (user_id, item_id, label) tuples
        
    Returns:
        Dictionary mapping user_id to set of relevant item_ids
    """
    user_relevant = defaultdict(set)
    for user_id, item_id, label in interactions:
        if label == 1:
            user_relevant[user_id].add(item_id)
    return dict(user_relevant)


def evaluate_model(
    model,
    user_data: List[Dict[str, Any]],
    item_data: List[Dict[str, Any]],
    test_interactions: List[Tuple[int, int, int]],
    train_interactions: List[Tuple[int, int, int]],
    k_values: List[int],
    encoders_used: Dict[str, bool],
    expected_user_features: Optional[int],
    expected_item_features: Optional[int],
    exclude_temporal_history: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on test set using ranking metrics
    
    Args:
        model: Trained model
        user_data: List of user data dictionaries
        item_data: List of item data dictionaries
        test_interactions: List of (user_id, item_id, label) tuples from test set
        train_interactions: List of (user_id, item_id, label) tuples from training set
        k_values: List of k values for metrics (e.g., [5, 10, 20])
        encoders_used: Dictionary indicating which encoders were used
        expected_user_features: Expected number of user features
        expected_item_features: Expected number of item features
        exclude_temporal_history: Whether to exclude items in user's temporal history
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Build mappings
    user_relevant_items = build_user_relevant_items(test_interactions)
    user_train_items = build_user_item_mappings(train_interactions)
    
    # Build user and item dictionaries for quick lookup
    user_dict = {user['user_id']: user for user in user_data}
    item_dict = {item['item_id']: item for item in item_data}
    
    # Extract temporal history items if needed
    user_temporal_items = {}
    if exclude_temporal_history:
        for user in user_data:
            user_id = user['user_id']
            temporal_items = set()
            if 'temporal' in user and user['temporal']:
                for field_name, field_value in user['temporal'].items():
                    if isinstance(field_value, list):
                        temporal_items.update(field_value)
            user_temporal_items[user_id] = temporal_items
    
    # Pre-compute all item embeddings (batch for efficiency)
    print("🔄 Generating item embeddings...")
    item_embeddings = {}
    item_ids = [item['item_id'] for item in item_data]
    
    for item in tqdm(item_data, desc="Items", leave=False):
        item_id = item['item_id']
        item_emb = generate_item_embedding(
            model, item, encoders_used=encoders_used, expected_num_features=expected_item_features
        )
        item_embeddings[item_id] = item_emb
    
    # Convert to numpy array for vectorized similarity computation
    item_embeddings_array = np.array([item_embeddings[item_id] for item_id in item_ids])
    
    # Evaluate each user in test set
    print("🔄 Evaluating users...")
    all_user_metrics = []
    test_users = list(user_relevant_items.keys())
    
    for user_id in tqdm(test_users, desc="Users"):
        if user_id not in user_dict:
            continue
        
        # Get user data
        user = user_dict[user_id]
        
        # Get relevant items for this user
        relevant_items = user_relevant_items[user_id]
        if len(relevant_items) == 0:
            continue
        
        # Generate user embedding
        user_embedding = generate_user_embedding(
            model, user, encoders_used=encoders_used, expected_num_features=expected_user_features
        )
        
        # Compute similarities with all items
        similarities = compute_cosine_similarity(user_embedding, item_embeddings_array)
        
        # Create list of (item_id, similarity) pairs
        item_similarities = list(zip(item_ids, similarities))
        
        # Filter out items to exclude
        items_to_exclude = user_train_items.get(user_id, set())
        if exclude_temporal_history and user_id in user_temporal_items:
            items_to_exclude = items_to_exclude.union(user_temporal_items[user_id])
        
        # Filter and sort by similarity
        filtered_similarities = [
            (item_id, sim) for item_id, sim in item_similarities
            if item_id not in items_to_exclude
        ]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract recommended item IDs
        recommended_items = [item_id for item_id, _ in filtered_similarities]
        
        # Compute metrics for this user
        user_metrics = compute_metrics_for_user(recommended_items, relevant_items, k_values)
        all_user_metrics.append(user_metrics)
    
    # Aggregate metrics across all users
    aggregated_metrics = {}
    for k in k_values:
        precision_key = f'precision@{k}'
        recall_key = f'recall@{k}'
        
        # Average across all users
        aggregated_metrics[precision_key] = np.mean([m[precision_key] for m in all_user_metrics])
        aggregated_metrics[recall_key] = np.mean([m[recall_key] for m in all_user_metrics])
    
    # Add number of evaluated users
    aggregated_metrics['num_users_evaluated'] = len(all_user_metrics)
    
    return aggregated_metrics


def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(description='Evaluate recommendation model on test set')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing model files (default: models)')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Base name of model files (default: model)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data JSON file')
    parser.add_argument('--test_interactions', type=str, required=True,
                       help='Path to test interactions JSON file')
    parser.add_argument('--train_interactions', type=str, required=True,
                       help='Path to train interactions JSON file')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for metrics (default: 5 10 20)')
    parser.add_argument('--exclude_temporal_history', action='store_true',
                       help='Exclude items in user temporal history from candidates')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save evaluation results (default: print to console)')
    
    args = parser.parse_args()
    
    print("📊 Recommendation System Evaluation")
    print("=" * 60)
    
    # Validate input files
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    if not os.path.exists(args.test_interactions):
        print(f"❌ Test interactions file not found: {args.test_interactions}")
        return 1
    
    if not os.path.exists(args.train_interactions):
        print(f"❌ Train interactions file not found: {args.train_interactions}")
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
        
        # Step 2: Load interactions
        print("🔄 Loading interactions...")
        test_interactions = load_interactions(args.test_interactions)
        train_interactions = load_interactions(args.train_interactions)
        print(f"✅ Loaded {len(test_interactions)} test interactions and {len(train_interactions)} train interactions")
        
        # Step 3: Load model
        print("🔄 Loading model...")
        model = load_model_from_config(config_path, model_path, item_data)
        model.eval()
        
        # Detect encoders used
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint_keys = set(checkpoint.keys())
        
        encoders_used = {
            'image': any('image_encoder' in k for k in checkpoint_keys),
            'text': any('text_encoder' in k for k in checkpoint_keys),
            'categorical': any('categorical_encoder' in k for k in checkpoint_keys),
            'continuous': any('continuous_encoder' in k or 'user_continuous_encoder' in k for k in checkpoint_keys),
            'temporal': any('temporal_encoder' in k for k in checkpoint_keys)
        }
        print(f"🔍 Encoders used: {[k for k, v in encoders_used.items() if v]}")
        
        # Determine expected number of features
        user_fusion_keys = [k for k in checkpoint_keys if 'user_generator.user_fusion.projection.0.weight' in k]
        item_fusion_keys = [k for k in checkpoint_keys if 'item_generator.item_fusion.projection.0.weight' in k]
        
        user_expected_features = None
        item_expected_features = None
        
        if user_fusion_keys:
            user_input_dim = checkpoint[user_fusion_keys[0]].shape[1]
            user_expected_features = user_input_dim // model.embedding_dim
            print(f"🔍 User fusion expects {user_expected_features} features")
        
        if item_fusion_keys:
            item_input_dim = checkpoint[item_fusion_keys[0]].shape[1]
            item_expected_features = item_input_dim // model.embedding_dim
            print(f"🔍 Item fusion expects {item_expected_features} features")
        
        print(f"✅ Model loaded successfully")
        
        # Step 4: Run evaluation
        print("🚀 Running evaluation...")
        metrics = evaluate_model(
            model=model,
            user_data=user_data,
            item_data=item_data,
            test_interactions=test_interactions,
            train_interactions=train_interactions,
            k_values=args.k,
            encoders_used=encoders_used,
            expected_user_features=user_expected_features,
            expected_item_features=item_expected_features,
            exclude_temporal_history=args.exclude_temporal_history
        )
        
        # Step 5: Output results
        print("\n" + "=" * 60)
        print("📊 Evaluation Results")
        print("=" * 60)
        print(f"Users evaluated: {metrics['num_users_evaluated']}")
        print()
        
        for k in args.k:
            precision_key = f'precision@{k}'
            recall_key = f'recall@{k}'
            print(f"K = {k}:")
            print(f"  Precision@{k}: {metrics[precision_key]:.4f}")
            print(f"  Recall@{k}:    {metrics[recall_key]:.4f}")
            print()
        
        if args.output_file:
            # Save results
            output_data = {
                'num_users_evaluated': metrics['num_users_evaluated'],
                'k_values': args.k,
                'exclude_temporal_history': args.exclude_temporal_history,
                'metrics': {k: float(v) for k, v in metrics.items() if k != 'num_users_evaluated'}
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Results saved to {args.output_file}")
        
        print("🎉 Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

