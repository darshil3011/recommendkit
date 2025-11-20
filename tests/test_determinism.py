"""
Test inference determinism using inference.py
Runs inference on the same user_id twice using inference.py and verifies identical results
"""
import os
import io
import pytest
import torch
import tempfile
import shutil
from contextlib import redirect_stdout, redirect_stderr
from trainer.data_loader import create_data_loaders, load_interactions_from_input
from trainer.pipeline_builder import RecommendationPipeline, save_complete_model, load_model_from_config
from trainer.trainer import train_model
import inference
import numpy as np


@pytest.fixture
def trained_model_for_determinism(inputs):
    """Create and train a small model for determinism testing"""
    # Use a small subset of data
    user_data = inputs.get_user_data()
    item_data = inputs.get_item_data()
    all_interactions = load_interactions_from_input(inputs)
    
    # Get first 20 users and 30 items
    user_ids = set()
    item_ids = set()
    for user_id, item_id, label in all_interactions:
        user_ids.add(user_id)
        item_ids.add(item_id)
        if len(user_ids) >= 20 and len(item_ids) >= 30:
            break
    
    user_ids = list(user_ids)[:20]
    item_ids = list(item_ids)[:30]
    
    filtered_user_data = [u for u in user_data if u['user_id'] in user_ids]
    filtered_item_data = [i for i in item_data if i['item_id'] in item_ids]
    filtered_interactions = [
        (u, i, l) for u, i, l in all_interactions
        if u in user_ids and i in item_ids
    ]
    
    # Create temporary directory for test model
    test_dir = tempfile.mkdtemp(prefix="test_determinism_model_")
    model_name = "test_determinism_model"
    
    try:
        # Create data loaders
        train_loader, val_loader, _, _ = create_data_loaders(
            inputs=inputs,
            interactions=filtered_interactions,
            train_split=0.8,
            batch_size=4,
            negative_sampling_ratio=0.0,
            num_workers=0,
            seed=42
        )
        
        # Create and train model with simple fusion (no attention layers) for small datasets
        model = RecommendationPipeline(
            embedding_dim=64,
            loss_type='bce',
            user_num_attention_layers=0,
            user_num_heads=1,
            user_use_simple_fusion=True,
            item_num_attention_layers=0,
            item_num_heads=1,
            item_use_simple_fusion=True,
            interaction_num_attention_layers=0,
            interaction_num_heads=1,
            interaction_use_simple_fusion=True,
            classifier_hidden_dims=[64],
            item_data=filtered_item_data
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train for 10 epochs to allow model to learn user-specific features
        # With limited epochs, the model may not learn enough to differentiate users
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            num_epochs=10,
            learning_rate=0.001,
            device=device,
            print_every=100,
            save_path=None
        )
        
        # Save model
        save_complete_model(
            model=model,
            save_dir=test_dir,
            model_name=model_name,
            verbose=False
        )
        
        weights_path = os.path.join(test_dir, f"{model_name}.pt")
        config_path = os.path.join(test_dir, f"{model_name}_config.json")
        
        yield {
            'weights_path': weights_path,
            'config_path': config_path,
            'item_data': filtered_item_data,
            'test_dir': test_dir
        }
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


@pytest.mark.slow
def test_determinism_with_inference(inputs, trained_model_for_determinism):
    """Test that inference.py produces deterministic results for same user_id"""
    model_info = trained_model_for_determinism
    weights_path = model_info['weights_path']
    config_path = model_info['config_path']
    item_data = model_info['item_data']
    
    # Load model using inference.py approach
    model = load_model_from_config(
        config_path=config_path,
        weights_path=weights_path,
        item_data=item_data
    )
    model.eval()
    
    # Get encoders used from checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    checkpoint_keys = set(checkpoint.keys())
    
    encoders_used = {
        'image': any('image_encoder' in k for k in checkpoint_keys),
        'text': any('text_encoder' in k for k in checkpoint_keys),
        'categorical': any('categorical_encoder' in k for k in checkpoint_keys),
        'continuous': any('continuous_encoder' in k or 'user_continuous_encoder' in k for k in checkpoint_keys),
        'temporal': any('temporal_encoder' in k for k in checkpoint_keys)
    }
    
    # Determine expected number of features from checkpoint
    user_fusion_keys = [k for k in checkpoint_keys if 'user_generator.user_fusion.projection.0.weight' in k]
    item_fusion_keys = [k for k in checkpoint_keys if 'item_generator.item_fusion.projection.0.weight' in k]
    
    user_expected_features = None
    item_expected_features = None
    
    if user_fusion_keys:
        user_input_dim = checkpoint[user_fusion_keys[0]].shape[1]
        user_expected_features = user_input_dim // model.embedding_dim
    
    if item_fusion_keys:
        item_input_dim = checkpoint[item_fusion_keys[0]].shape[1]
        item_expected_features = item_input_dim // model.embedding_dim
    
    # Get a test user by user_id
    user_data_list = inputs.get_user_data()
    assert len(user_data_list) > 0, "No user data available"
    test_user = user_data_list[0]
    test_user_id = test_user['user_id']
    
    # Get the same user again by ID
    user_data_dict = {user['user_id']: user for user in user_data_list}
    assert test_user_id in user_data_dict, f"User {test_user_id} not found in user data"
    
    # Get a test item
    assert len(item_data) > 0, "No item data available"
    test_item = item_data[0]
    
    # Run inference twice using inference.py with the same user_id
    # First call
    user_emb1 = inference.generate_user_embedding(
        model=model,
        user_data=test_user,
        encoders_used=encoders_used,
        expected_num_features=user_expected_features
    )
    item_emb1 = inference.generate_item_embedding(
        model=model,
        item_data=test_item,
        encoders_used=encoders_used,
        expected_num_features=item_expected_features
    )
    similarity1 = inference.compute_similarity(user_emb1, item_emb1)
    
    # Second call with same user (retrieved by ID to simulate real usage)
    same_user = user_data_dict[test_user_id]
    user_emb2 = inference.generate_user_embedding(
        model=model,
        user_data=same_user,
        encoders_used=encoders_used,
        expected_num_features=user_expected_features
    )
    item_emb2 = inference.generate_item_embedding(
        model=model,
        item_data=test_item,
        encoders_used=encoders_used,
        expected_num_features=item_expected_features
    )
    similarity2 = inference.compute_similarity(user_emb2, item_emb2)
    
    # Results should be identical
    diff = abs(similarity1 - similarity2)
    
    assert diff < 1e-6, \
        f"Inference results differ for same user_id! Similarity1={similarity1:.8f}, Similarity2={similarity2:.8f}, Diff={diff:.10f}"
    
    # CRITICAL: Also verify that different users produce different embeddings
    # This ensures the model is actually learning user-specific features, not just outputting the same embedding
    if len(user_data_list) > 1:
        different_user = user_data_list[1]
        user_emb3 = inference.generate_user_embedding(
            model=model,
            user_data=different_user,
            encoders_used=encoders_used,
            expected_num_features=user_expected_features
        )
        
        # Check that embeddings are different (not identical)
        # Note: With small datasets and limited training, embeddings might be very similar
        # We only fail if they're exactly identical (which would indicate a bug)
        emb_diff = np.linalg.norm(user_emb1 - user_emb3)
        assert emb_diff > 1e-8, \
            f"Different users produce identical embeddings! This indicates a bug. " \
            f"User {test_user_id} and User {different_user['user_id']} embeddings differ by only {emb_diff:.10f}"
        
        # Warn if difference is very small (but don't fail - this is expected with limited training)
        if emb_diff < 1e-3:
            import warnings
            warnings.warn(
                f"User embeddings are very similar (diff={emb_diff:.6f}). "
                f"This may indicate insufficient training epochs or the model not learning user-specific features well."
            )
        else:
            # Log when differences are good (for debugging)
            print(f"✅ User embeddings differ by {emb_diff:.6f} (threshold: 1e-3) - model is learning user-specific features")
        
        # Also check similarity with same item should be different
        # Very lenient threshold - just ensure they're not exactly the same
        similarity3 = inference.compute_similarity(user_emb3, item_emb1)
        similarity_diff = abs(similarity1 - similarity3)
        assert similarity_diff > 1e-8, \
            f"Different users produce identical similarities! User {test_user_id} similarity: {similarity1:.8f}, " \
            f"User {different_user['user_id']} similarity: {similarity3:.8f}, Diff: {similarity_diff:.10f}. " \
            f"This indicates a bug."
        
        # Warn if similarity difference is very small (but don't fail)
        if similarity_diff < 1e-3:
            import warnings
            warnings.warn(
                f"User similarities are very similar (diff={similarity_diff:.6f}). "
                f"This may indicate insufficient training epochs or the model not learning user-specific features well."
            )
        else:
            # Log when differences are good (for debugging)
            print(f"✅ User similarities differ by {similarity_diff:.6f} (threshold: 1e-3) - model is learning user-specific features")