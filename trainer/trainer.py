"""
trainer/trainer.py
Training logic for recommendation pipeline using PyTorch built-in functionality
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import time
import sys
from tqdm import tqdm

# Import RecommendationPipeline with fallback for direct execution  
try:
    from trainer.pipeline_builder import RecommendationPipeline
except ImportError:
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline_builder import RecommendationPipeline


def train_model(model: RecommendationPipeline,
               train_loader: torch.utils.data.DataLoader,
               val_loader: Optional[torch.utils.data.DataLoader] = None,
               num_epochs: int = 100,
               learning_rate: float = 0.001,
               optimizer_type: str = "adam",
               scheduler_type: Optional[str] = "plateau",
               device: Optional[torch.device] = None,
               print_every: int = 10,
               save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train the recommendation model using PyTorch's built-in training loop
    
    Args:
        model: RecommendationPipeline to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        scheduler_type: Type of scheduler ("plateau", "cosine", "step", None)
        device: Training device
        print_every: Print progress every N epochs
        save_path: Path to save best model (optional)
        
    Returns:
        training_history: Dict with training metrics
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer using PyTorch built-ins
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create scheduler using PyTorch built-ins
    scheduler = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0
        
        # Create progress bar for training (updates in place, clears after completion)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, ncols=120, file=sys.stdout, dynamic_ncols=True, 
                         position=0, mininterval=0.1)
        
        for batch in train_pbar:
            # Extract batch data (this depends on your data loader structure)
            user_data, item_data, labels = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits = model(user_data, item_data, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.sigmoid(logits) > 0.5
                accuracy = (predictions.squeeze() == labels).float().mean()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += accuracy.item()
            num_train_batches += 1
            
            # Update progress bar with real-time metrics
            current_loss = epoch_train_loss / num_train_batches
            current_acc = epoch_train_acc / num_train_batches
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Average training metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_acc = epoch_train_acc / num_train_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        # Validation phase
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            num_val_batches = 0
            
            # Create progress bar for validation (updates in place, clears after completion)
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                           leave=False, ncols=120, file=sys.stdout, dynamic_ncols=True,
                           position=0, mininterval=0.1)
            
            with torch.no_grad():
                for batch in val_pbar:
                    user_data, item_data, labels = prepare_batch(batch, device)
                    
                    # Forward pass
                    loss, logits = model(user_data, item_data, labels)
                    
                    # Calculate accuracy
                    predictions = torch.sigmoid(logits) > 0.5
                    accuracy = (predictions.squeeze() == labels).float().mean()
                    
                    epoch_val_loss += loss.item()
                    epoch_val_acc += accuracy.item()
                    num_val_batches += 1
                    
                    # Update validation progress bar
                    current_val_loss = epoch_val_loss / num_val_batches
                    current_val_acc = epoch_val_acc / num_val_batches
                    val_pbar.set_postfix({
                        'Val_Loss': f'{current_val_loss:.4f}',
                        'Val_Acc': f'{current_val_acc:.4f}'
                    })
            
            # Average validation metrics
            avg_val_loss = epoch_val_loss / num_val_batches
            avg_val_acc = epoch_val_acc / num_val_batches
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            
            # Save best model (if validation loss improved)
            if save_path and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Import the save function
                try:
                    from trainer.pipeline_builder import save_complete_model
                    import os
                    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "models"
                    model_name = os.path.splitext(os.path.basename(save_path))[0]
                    save_complete_model(model, save_dir, model_name, verbose=True)  # Enable verbose to catch issues
                    # Use tqdm.write to avoid interfering with progress bar
                    tqdm.write(f"✅ New best model saved (val_loss: {best_val_loss:.4f})")
                except ImportError:
                    # Fallback to simple state dict saving
                    torch.save(model.state_dict(), save_path)
                    tqdm.write(f"✅ New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Save model every epoch (overwrite) - ensures we have latest weights even if training stops early
        if save_path:
            try:
                from trainer.pipeline_builder import save_complete_model
                import os
                save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "models"
                model_name = os.path.splitext(os.path.basename(save_path))[0]
                # Save every epoch (overwrites previous) - use verbose=True for first epoch to catch issues
                save_complete_model(model, save_dir, model_name, verbose=(epoch == 0))
                # Use tqdm.write to avoid interfering with progress bar
                tqdm.write(f"💾 Model saved (epoch {epoch+1}/{num_epochs})")
            except ImportError:
                # Fallback to simple state dict saving
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"💾 Model saved (epoch {epoch+1}/{num_epochs})")
        
        # Scheduler step
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(avg_val_loss if val_loader else avg_train_loss)
            else:
                scheduler.step()
        
        # Print epoch summary (use tqdm.write to avoid interfering with progress bar)
        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            tqdm.write(f"\n📊 Epoch {epoch + 1}/{num_epochs} Summary ({elapsed/60:.1f}m)")
            tqdm.write(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            if val_loader is not None:
                tqdm.write(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            tqdm.write("")  # Add spacing
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }


def prepare_batch(batch: Any, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
    """
    Prepare batch for training by moving to device and organizing data
    
    Args:
        batch: Raw batch from DataLoader
        device: Target device
        
    Returns:
        user_data: User features organized by type
        item_data: Item features organized by type  
        labels: Binary engagement labels
    """
    # This function will depend on your specific data format
    # For now, providing a template structure
    
    # Example structure - modify based on your actual data format:
    # batch = {
    #     "user_features": {
    #         "image": {...},
    #         "text": {...},
    #         "categorical": {...},
    #         "temporal": {...}
    #     },
    #     "item_features": {
    #         "image": {...},
    #         "text": {...},  
    #         "categorical": {...}
    #     },
    #     "labels": tensor([0, 1, 1, 0, ...])
    # }
    
    user_data = {}
    item_data = {}
    labels = None
    
    # Move user features to device
    if "user_features" in batch:
        for feature_type, features in batch["user_features"].items():
            if isinstance(features, dict):
                user_data[feature_type] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in features.items()}
            else:
                user_data[feature_type] = features.to(device) if isinstance(features, torch.Tensor) else features
    
    # Move item features to device  
    if "item_features" in batch:
        for feature_type, features in batch["item_features"].items():
            if isinstance(features, dict):
                item_data[feature_type] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in features.items()}
            else:
                item_data[feature_type] = features.to(device) if isinstance(features, torch.Tensor) else features
    
    # Move labels to device
    if "labels" in batch:
        labels = batch["labels"].to(device)
    
    return user_data, item_data, labels


def evaluate_model(model: RecommendationPipeline,
                  test_loader: torch.utils.data.DataLoader,
                  device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained RecommendationPipeline
        test_loader: Test data loader
        device: Evaluation device
        
    Returns:
        test_metrics: Dict with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_data, item_data, labels = prepare_batch(batch, device)

                    # DEBUG: Check what encoders output
            user_encoded = model._encode_features(user_data, model.user_dimension_aligner)
            item_encoded = model._encode_features(item_data, model.item_dimension_aligner)
            
            print("\n=== ENCODER OUTPUT DEBUG ===")
            print(f"User encoded keys: {list(user_encoded.keys())}")
            print(f"Item encoded keys: {list(item_encoded.keys())}")
            print(f"User feature types: {model._get_feature_types(user_encoded)}")
            print(f"Item feature types: {model._get_feature_types(item_encoded)}")
            print("============================\n")
    
            
            # Forward pass
            loss, logits = model(user_data, item_data, labels)
            
            # Calculate metrics
            predictions = torch.sigmoid(logits)
            binary_predictions = predictions > 0.5
            accuracy = (binary_predictions.squeeze() == labels).float().mean()
            
            total_loss += loss.item() * labels.size(0)
            total_accuracy += accuracy.item() * labels.size(0)
            total_samples += labels.size(0)
            
            # Store for detailed metrics
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate final metrics
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    
    # Calculate additional metrics using sklearn (optional)
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        binary_preds = [1 if p > 0.5 else 0 for p in all_predictions]
        
        precision = precision_score(all_labels, binary_preds)
        recall = recall_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_predictions)
        
        return {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auc": auc
        }
    except ImportError:
        return {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy
        }


# Example usage
if __name__ == "__main__":
    print("=== Recommendation Model Training ===")
    
   
    import sys
    import os
    
    # Handle imports for both direct execution and module import
    try:
        # When imported as a module
        from trainer.pipeline_builder import RecommendationPipeline
        from trainer.data_loader import create_data_loaders, load_interactions_from_input
        from input_processor import Inputs
    except ImportError:
        # When run directly from trainer directory - add parent to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pipeline_builder import RecommendationPipeline
        from data_loader import create_data_loaders, load_interactions_from_input
        from input_processor import Inputs
    
    print("Setting up demo data...")
    
    # Create Inputs object and load test data
    inputs = Inputs()
    
    # Configure minimal validation for testing (only skip file existence check)
    inputs.configure_validators(image_check_files=False)
    
    # Load test data (you'll need to ensure test_input.json exists)
    try:
        result = inputs.load_from_json("inference/test_input.json")
        if not result.is_valid:
            print("Data loading errors:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print(f"Loaded {len(inputs.get_user_data())} users and {len(inputs.get_item_data())} items")
        
        # Load real interactions from the input data
        interactions = load_interactions_from_input(inputs=inputs)
        
    except FileNotFoundError:
        print("Error: test_input.json not found.")
        print("Please create test data file first using input_processor.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create pipeline with dynamic field count detection (like in tower_extractor_corrected.py)
    # First detect actual field counts from data
    user_data = inputs.get_user_data()
    item_data = inputs.get_item_data()
    
    # Count actual image fields
    user_image_fields = len(user_data[0].get('image', {})) if user_data else 1
    item_image_fields = len(item_data[0].get('image', {})) if item_data else 1
    
    print(f"Detected image fields - User: {user_image_fields}, Item: {item_image_fields}")
    
    # For training, we need to handle both user and item images, so use the max
    max_image_fields = max(user_image_fields, item_image_fields)
    
    model = RecommendationPipeline(
        embedding_dim=256,
        loss_type="bce",
        # Use dynamic field count detection
        image_encoder_config={
            "aggregation_strategy": "concat",  
            "model_type": "cnn",
            "embedding_dim": 256,
            "num_image_fields": max_image_fields  # Use actual detected field count
        }
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        inputs=inputs,
        interactions=interactions,
        train_split=0.8,
        batch_size=32,
        negative_sampling_ratio=1.5,
        seed=42
    )
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Reduced for demo
        learning_rate=0.001,
        optimizer_type="adamw",
        scheduler_type="cosine",
        print_every=2
    )
    
    print("Training completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"Final val loss: {history['val_losses'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_accuracies'][-1]:.4f}")
    if history['val_accuracies']:
        print(f"Final val accuracy: {history['val_accuracies'][-1]:.4f}")
        