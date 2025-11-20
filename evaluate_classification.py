#!/usr/bin/env python3
"""
Classification evaluation script for recommendation system
Tests binary classification (0/1) on test set interactions
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_processor import Inputs
from trainer.pipeline_builder import load_model_from_config
from trainer.data_loader import RecommendationDataset, collate_recommendation_batch
from torch.utils.data import DataLoader
from trainer.trainer import prepare_batch


def load_interactions(file_path: str) -> List[Tuple[int, int, int]]:
    """Load interactions from JSON file"""
    with open(file_path, 'r') as f:
        interactions = json.load(f)
    return [tuple(interaction) for interaction in interactions]


def evaluate_classification(
    model,
    inputs: Inputs,
    test_interactions: List[Tuple[int, int, int]],
    batch_size: int = 32,
    threshold: float = 0.5,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on test set using binary classification
    
    Args:
        model: Trained model
        inputs: Inputs object with user and item data
        test_interactions: List of (user_id, item_id, label) tuples from test set
        batch_size: Batch size for evaluation
        threshold: Classification threshold (default 0.5)
        device: Device to run evaluation on
        
    Returns:
        Dictionary with classification metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader (same as training)
    test_dataset = RecommendationDataset(
        inputs=inputs,
        interactions=test_interactions,
        negative_sampling_ratio=0.0,  # Don't add negative samples, use only test interactions
        seed=42
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_recommendation_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print(f"🔄 Evaluating {len(test_interactions)} test interactions...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Batches"):
            # Prepare batch (move to device)
            user_data, item_data, labels = prepare_batch(batch, device)
            
            # Forward pass through model (no labels = inference mode)
            logits = model(user_data, item_data)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits).squeeze()
            
            # Convert to predictions
            predictions = (probs > threshold).long()
            
            # Store results
            all_probs.extend(probs.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Basic metrics
    accuracy = (all_predictions == all_labels).mean()
    
    # Precision, Recall, F1
    true_positives = ((all_predictions == 1) & (all_labels == 1)).sum()
    false_positives = ((all_predictions == 1) & (all_labels == 0)).sum()
    false_negatives = ((all_predictions == 0) & (all_labels == 1)).sum()
    true_negatives = ((all_predictions == 0) & (all_labels == 0)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC-ROC (if sklearn available)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except ImportError:
        auc = None
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'num_samples': len(all_labels),
        'positive_samples': int(all_labels.sum()),
        'negative_samples': int((all_labels == 0).sum())
    }
    
    if auc is not None:
        metrics['auc_roc'] = float(auc)
    
    return metrics


def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(description='Evaluate recommendation model on test set (classification)')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing model files (default: models)')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Base name of model files (default: model)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input data JSON file')
    parser.add_argument('--test_interactions', type=str, required=True,
                       help='Path to test interactions JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save evaluation results (default: print to console)')
    
    args = parser.parse_args()
    
    print("📊 Recommendation System Classification Evaluation")
    print("=" * 60)
    
    # Validate input files
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    if not os.path.exists(args.test_interactions):
        print(f"❌ Test interactions file not found: {args.test_interactions}")
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
        
        # Step 2: Load test interactions
        print("🔄 Loading test interactions...")
        test_interactions = load_interactions(args.test_interactions)
        print(f"✅ Loaded {len(test_interactions)} test interactions")
        
        # Step 3: Load model
        print("🔄 Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_from_config(config_path, model_path, item_data)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded on device: {device}")
        
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
        print("🚀 Running classification evaluation...")
        metrics = evaluate_classification(
            model=model,
            inputs=inputs,
            test_interactions=test_interactions,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=device
        )
        
        # Step 5: Output results
        print("\n" + "=" * 60)
        print("📊 Classification Evaluation Results")
        print("=" * 60)
        print(f"Total samples: {metrics['num_samples']}")
        print(f"Positive samples: {metrics['positive_samples']}")
        print(f"Negative samples: {metrics['negative_samples']}")
        print()
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print()
        print("Confusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        if args.output_file:
            # Save results
            output_data = {
                'threshold': args.threshold,
                'batch_size': args.batch_size,
                'metrics': metrics
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n✅ Results saved to {args.output_file}")
        
        print("\n🎉 Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

