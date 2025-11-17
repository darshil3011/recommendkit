"""
Test inference determinism
Runs inference on the same user-item pair twice and verifies identical results
"""
import os
import io
import pytest
import torch
from contextlib import redirect_stdout, redirect_stderr
from trainer.pipeline_builder import load_model_from_config


@pytest.fixture
def trained_model(inputs):
    """Load a trained model for testing"""
    item_data = inputs.get_item_data()
    
    model_path = 'models/latest_model.pt'
    config_path = 'models/latest_model_config.json'
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        pytest.skip("Trained model not found. Please train a model first.")
    
    # Suppress verbose output during model loading
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = load_model_from_config(
            config_path=config_path,
            weights_path=model_path,
            item_data=item_data
        )
    
    model.eval()
    return model


def test_determinism(inputs, trained_model):
    """Test that inference produces deterministic results"""
    user_data_dict = {user['user_id']: user for user in inputs.get_user_data()}
    item_data = inputs.get_item_data()
    
    # Get first user and first item
    user_id = list(user_data_dict.keys())[0]
    user_data = user_data_dict[user_id]
    test_item = item_data[0]
    
    # Prepare features
    user_features = {
        'categorical': user_data.get('categorical', {}),
        'continuous': user_data.get('continuous', {}),
        'temporal': user_data.get('temporal', {})
    }
    
    item_features = {
        'categorical': test_item.get('categorical', {}),
        'continuous': test_item.get('continuous', {}),
        'temporal': test_item.get('temporal', {})
    }
    
    # Run inference twice
    with torch.no_grad():
        score1 = trained_model.predict_proba(user_data=user_features, item_data=item_features)
        score1_value = float(score1.item())
    
    with torch.no_grad():
        score2 = trained_model.predict_proba(user_data=user_features, item_data=item_features)
        score2_value = float(score2.item())
    
    # Compare results - should be identical
    diff = abs(score1_value - score2_value)
    
    assert diff < 1e-6, \
        f"Results differ! Score1={score1_value:.8f}, Score2={score2_value:.8f}, Diff={diff:.10f}"
