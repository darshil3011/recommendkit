"""
Test model save and load functionality
Verifies that all parameters are saved and can be loaded correctly
"""
import os
import tempfile
import shutil
import pytest
import torch
from trainer.data_loader import create_data_loaders
from trainer.pipeline_builder import RecommendationPipeline, save_complete_model, load_model_from_config
from trainer.trainer import train_model


@pytest.fixture
def small_dataset_for_save_load(inputs):
    """Create a smaller dataset for save/load testing"""
    from trainer.data_loader import load_interactions_from_input
    
    user_data = inputs.get_user_data()
    item_data = inputs.get_item_data()
    
    # Convert interactions from dict format to tuples using the helper function
    all_interactions = load_interactions_from_input(inputs)
    
    # Get unique user and item IDs from interactions
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
    
    return filtered_user_data, filtered_item_data, filtered_interactions


@pytest.mark.slow
def test_save_load(inputs, small_dataset_for_save_load):
    """Test save and load functionality"""
    user_data, item_data, interactions = small_dataset_for_save_load
    
    # Create temporary directory for test model
    test_dir = tempfile.mkdtemp(prefix="test_model_")
    model_name = "test_model"
    
    try:
        # Create data loaders
        train_loader, val_loader, _, _ = create_data_loaders(
            inputs=inputs,
            interactions=interactions,
            train_split=0.8,
            batch_size=4,
            negative_sampling_ratio=0.0,  # No negative sampling for speed
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
            item_data=item_data
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
            print_every=100,  # Don't print much
            save_path=None
        )
        
        # Set model to eval mode before saving to ensure consistent state
        model.eval()
        
        # Get original state dict
        original_state_dict = model.state_dict()
        original_keys = set(original_state_dict.keys())
        original_params = {k: v.clone() for k, v in original_state_dict.items()}
        
        assert len(original_keys) > 0, "Model has no parameters"
        
        # Save model
        save_complete_model(
            model=model,
            save_dir=test_dir,
            model_name=model_name,
            verbose=False  # Suppress output in tests
        )
        
        # Verify files exist
        weights_path = os.path.join(test_dir, f"{model_name}.pt")
        config_path = os.path.join(test_dir, f"{model_name}_config.json")
        
        assert os.path.exists(weights_path), f"Weights file not found: {weights_path}"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        
        # Load checkpoint and verify all keys are present
        checkpoint = torch.load(weights_path, map_location='cpu')
        checkpoint_keys = set(checkpoint.keys())
        
        missing_in_checkpoint = original_keys - checkpoint_keys
        assert len(missing_in_checkpoint) == 0, \
            f"{len(missing_in_checkpoint)} parameters missing from checkpoint: {sorted(list(missing_in_checkpoint))[:10]}"
        
        # Load model
        loaded_model = load_model_from_config(
            config_path=config_path,
            weights_path=weights_path,
            item_data=item_data
        )
        
        # Verify loaded state dict
        loaded_state_dict = loaded_model.state_dict()
        loaded_keys = set(loaded_state_dict.keys())
        
        missing_in_loaded = original_keys - loaded_keys
        assert len(missing_in_loaded) == 0, \
            f"{len(missing_in_loaded)} parameters missing in loaded model: {sorted(list(missing_in_loaded))[:10]}"
        
        # Verify weights match
        # Set loaded model to eval mode for consistent comparison
        loaded_model.eval()
        
        mismatched = []
        for key in original_keys:
            if key not in loaded_state_dict:
                continue  # Skip if key doesn't exist (should have been caught earlier)
            
            orig = original_params[key]
            loaded = loaded_state_dict[key]
            
            # Check if shapes match
            if orig.shape != loaded.shape:
                mismatched.append(f"{key} (shape mismatch: {orig.shape} vs {loaded.shape})")
                continue
            
            # Check if values match (use more lenient tolerance for numerical precision)
            if not torch.allclose(orig, loaded, atol=1e-5, rtol=1e-4):
                max_diff = (orig - loaded).abs().max().item()
                mismatched.append(f"{key} (max_diff: {max_diff:.2e})")
        
        assert len(mismatched) == 0, \
            f"{len(mismatched)} parameters have mismatched weights:\n" + "\n".join(sorted(mismatched)[:20])
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
