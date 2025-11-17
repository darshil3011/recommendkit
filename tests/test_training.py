"""
Test training loop with small dataset
Trains on 50 users, 50 items for 1 epoch to verify training works correctly
"""
import pytest
import torch
from trainer.data_loader import create_data_loaders
from trainer.pipeline_builder import RecommendationPipeline
from trainer.trainer import train_model


@pytest.mark.slow
def test_training(inputs, small_dataset):
    """Test training loop"""
    user_data, item_data, interactions = small_dataset
    
    # Verify we have enough interactions
    assert len(interactions) >= 10, f"Not enough interactions: {len(interactions)}"
    
    # Create data loaders
    train_loader, val_loader, train_interactions, test_interactions = create_data_loaders(
        inputs=inputs,
        interactions=interactions,
        train_split=0.8,
        batch_size=8,
        negative_sampling_ratio=0.5,  # Reduced for speed
        num_workers=0,
        seed=42
    )
    
    assert len(train_loader) > 0, "Train loader is empty"
    
    # Create model
    model = RecommendationPipeline(
        embedding_dim=64,
        loss_type='bce',
        user_num_attention_layers=2,
        user_num_heads=4,
        item_num_attention_layers=1,
        item_num_heads=4,
        interaction_num_attention_layers=1,
        interaction_num_heads=4,
        classifier_hidden_dims=[64, 32],
        item_data=item_data
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters"
    
    # Train for 1 epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        learning_rate=0.001,
        device=device,
        print_every=5,
        save_path=None  # Don't save during test
    )
    
    # Verify training completed
    assert 'train_loss' in history, "Training did not produce loss history"
    assert len(history['train_loss']) > 0, "Training loss history is empty"
    
    final_train_loss = history['train_loss'][-1]
    assert isinstance(final_train_loss, (int, float)), "Train loss is not a number"
    assert final_train_loss >= 0, "Train loss should be non-negative"
    
    # Verify validation loss if validation loader exists
    if val_loader and 'val_loss' in history:
        assert len(history['val_loss']) > 0, "Validation loss history is empty"
        final_val_loss = history['val_loss'][-1]
        assert isinstance(final_val_loss, (int, float)), "Val loss is not a number"
        assert final_val_loss >= 0, "Val loss should be non-negative"
