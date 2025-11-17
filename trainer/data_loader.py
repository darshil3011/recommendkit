"""
trainer/data_loader.py
Data loading and processing for recommendation training with positive/negative pairs
Integrates with input_processor.py and creates training batches
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
import random
import numpy as np
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_processor import Inputs


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation training with positive/negative pairs
    Uses data from input_processor.py format
    """
    
    def __init__(self,
                 inputs: Inputs,
                 interactions: List[Tuple[Union[str, int], Union[str, int], int]],
                 negative_sampling_ratio: float = 1.0,
                 seed: int = 42):
        """
        Args:
            inputs: Inputs object containing user and item data
            interactions: List of (user_id, item_id, label) tuples where label is 1=engaged, 0=not_engaged
            negative_sampling_ratio: Ratio of negative samples to positive samples
            seed: Random seed for sampling
        """
        self.inputs = inputs
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Get user and item data
        self.user_data_dict = {user['user_id']: user for user in inputs.get_user_data()}
        self.item_data_dict = {item['item_id']: item for item in inputs.get_item_data()}
        
        # Process interactions
        self.positive_interactions = []
        self.negative_interactions = []
        self.user_positive_items = defaultdict(set)
        
        for user_id, item_id, label in interactions:
            if label == 1:
                self.positive_interactions.append((user_id, item_id))
                self.user_positive_items[user_id].add(item_id)
            else:
                self.negative_interactions.append((user_id, item_id))
        
        # Get all users and items for negative sampling
        self.all_users = list(self.user_data_dict.keys())
        self.all_items = list(self.item_data_dict.keys())
        
        # Create training samples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[Union[str, int], Union[str, int], int]]:
        """Create training samples with negative sampling"""
        samples = []
        
        # Add all positive interactions
        for user_id, item_id in self.positive_interactions:
            samples.append((user_id, item_id, 1))
        
        # Add existing negative interactions
        for user_id, item_id in self.negative_interactions:
            samples.append((user_id, item_id, 0))
        
        # Generate additional negative samples if needed
        num_positives = len(self.positive_interactions)
        num_existing_negatives = len(self.negative_interactions)
        target_negatives = int(num_positives * self.negative_sampling_ratio)
        additional_negatives_needed = max(0, target_negatives - num_existing_negatives)
        
        if additional_negatives_needed > 0:
            additional_negatives = self._generate_negative_samples(additional_negatives_needed)
            samples.extend(additional_negatives)
        
        # Shuffle samples
        random.shuffle(samples)
        
        print(f"Created dataset with {len(self.positive_interactions)} positive and {len(samples) - len(self.positive_interactions)} negative samples")
        
        return samples
    
    def _generate_negative_samples(self, num_samples: int) -> List[Tuple[Union[str, int], Union[str, int], int]]:
        """Generate negative samples by random sampling"""
        negative_samples = []
        
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            user_id = random.choice(self.all_users)
            item_id = random.choice(self.all_items)
            
            # Check if this is not a positive interaction
            if item_id not in self.user_positive_items[user_id]:
                negative_samples.append((user_id, item_id, 0))
            
            attempts += 1
        
        return negative_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample
        
        Returns:
            sample: Dict containing user_features, item_features, and label
        """
        user_id, item_id, label = self.samples[idx]
        
        # Get user data
        user_data = self.user_data_dict.get(user_id, {})
        
        # Get item data
        item_data = self.item_data_dict.get(item_id, {})
        
        # Organize user features by type (matching your pipeline structure)
        user_features = {}
        for feature_type in ['image', 'text', 'categorical', 'continuous', 'temporal']:
            if feature_type in user_data:
                user_features[feature_type] = user_data[feature_type]
        
        # Organize item features by type
        item_features = {}
        for feature_type in ['image', 'text', 'categorical', 'continuous', 'temporal']:
            if feature_type in item_data:
                item_features[feature_type] = item_data[feature_type]
        
        return {
            "user_features": user_features,
            "item_features": item_features,
            "labels": torch.tensor(label, dtype=torch.float),
            "user_id": user_id,
            "item_id": item_id
        }


def collate_recommendation_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for recommendation batches
    Handles the nested structure of user_features and item_features
    
    Args:
        batch: List of samples from RecommendationDataset
        
    Returns:
        batched_data: Dict with batched data ready for pipeline
    """
    batch_size = len(batch)
    
    # Initialize batched structures
    batched_user_features = {}
    batched_item_features = {}
    
    # Get all possible feature types
    all_user_feature_types = set()
    all_item_feature_types = set()
    
    for sample in batch:
        all_user_feature_types.update(sample["user_features"].keys())
        all_item_feature_types.update(sample["item_features"].keys())
    
    # Batch user features by type
    for feature_type in all_user_feature_types:
        batched_user_features[feature_type] = {}
        
        # Get all possible field names for this feature type
        all_field_names = set()
        for sample in batch:
            if feature_type in sample["user_features"]:
                if isinstance(sample["user_features"][feature_type], dict):
                    all_field_names.update(sample["user_features"][feature_type].keys())
        
        # Batch each field
        for field_name in all_field_names:
            field_values = []
            for sample in batch:
                if (feature_type in sample["user_features"] and 
                    isinstance(sample["user_features"][feature_type], dict) and
                    field_name in sample["user_features"][feature_type]):
                    field_values.append(sample["user_features"][feature_type][field_name])
                else:
                    field_values.append(None)  # Missing data
            
            batched_user_features[feature_type][field_name] = field_values
    
    # Batch item features by type (same logic as user features)
    for feature_type in all_item_feature_types:
        batched_item_features[feature_type] = {}
        
        # Get all possible field names for this feature type
        all_field_names = set()
        for sample in batch:
            if feature_type in sample["item_features"]:
                if isinstance(sample["item_features"][feature_type], dict):
                    all_field_names.update(sample["item_features"][feature_type].keys())
        
        # Batch each field
        for field_name in all_field_names:
            field_values = []
            for sample in batch:
                if (feature_type in sample["item_features"] and 
                    isinstance(sample["item_features"][feature_type], dict) and
                    field_name in sample["item_features"][feature_type]):
                    field_values.append(sample["item_features"][feature_type][field_name])
                else:
                    field_values.append(None)  # Missing data
            
            batched_item_features[feature_type][field_name] = field_values
    
    # Batch labels
    labels = torch.stack([sample["labels"] for sample in batch])
    
    return {
        "user_features": batched_user_features,
        "item_features": batched_item_features,
        "labels": labels,
        "user_ids": [sample["user_id"] for sample in batch],
        "item_ids": [sample["item_id"] for sample in batch]
    }


def create_data_loaders(inputs: Inputs,
                       interactions: List[Tuple[Union[str, int], Union[str, int], int]],
                       train_split: float = 0.8,
                       batch_size: int = 32,
                       negative_sampling_ratio: float = 1.0,
                       num_workers: int = 0,
                       seed: int = 42,
                       test_split: float = 0.05) -> Tuple[DataLoader, Optional[DataLoader], List[Tuple[Union[str, int], Union[str, int], int]], List[Tuple[Union[str, int], Union[str, int], int]]]:
    """
    Create train and validation data loaders, and return test and train interactions
    
    Args:
        inputs: Inputs object with user and item data
        interactions: List of (user_id, item_id, label) interactions
        train_split: Fraction of data for training (remaining goes to validation)
        batch_size: Batch size
        negative_sampling_ratio: Ratio of negative to positive samples
        num_workers: Number of worker processes for DataLoader
        seed: Random seed
        test_split: Fraction of data for testing (default 0.05 = 5%)
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (None if train_split=1.0)
        test_interactions: Test interactions list for evaluation
        train_interactions: Training interactions list (for exclusion during evaluation)
    """
    # Split interactions into train, validation, and test
    random.seed(seed)
    random.shuffle(interactions)
    
    # First split: test set (5%)
    test_split_idx = int(len(interactions) * test_split)
    test_interactions = interactions[:test_split_idx]
    remaining_interactions = interactions[test_split_idx:]
    
    # Second split: train and validation from remaining
    # Adjust train_split to account for test split
    adjusted_train_split = train_split / (1 - test_split)
    split_idx = int(len(remaining_interactions) * adjusted_train_split)
    train_interactions = remaining_interactions[:split_idx]
    val_interactions = remaining_interactions[split_idx:] if split_idx < len(remaining_interactions) else []
    
    print(f"Split interactions: {len(train_interactions)} train, {len(val_interactions)} validation, {len(test_interactions)} test")
    
    # Create datasets
    train_dataset = RecommendationDataset(
        inputs=inputs,
        interactions=train_interactions,
        negative_sampling_ratio=negative_sampling_ratio,
        seed=seed
    )
    
    val_dataset = None
    if val_interactions:
        val_dataset = RecommendationDataset(
            inputs=inputs,
            interactions=val_interactions,
            negative_sampling_ratio=negative_sampling_ratio,
            seed=seed + 1  # Different seed for validation
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_recommendation_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_recommendation_batch,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader, test_interactions, train_interactions


def create_interactions_from_data_DEPRECATED(inputs: Inputs,
                                positive_ratio: float = 0.3,
                                interactions_per_user: int = 10,
                                seed: int = 42) -> List[Tuple[Union[str, int], Union[str, int], int]]:
    """
    DEPRECATED: Create synthetic interactions for testing purposes
    Users should now provide real interaction data in their JSON files
    
    Args:
        inputs: Inputs object with user and item data
        positive_ratio: Ratio of positive interactions
        interactions_per_user: Number of interactions per user
        seed: Random seed
        
    Returns:
        interactions: List of (user_id, item_id, label) tuples
    """
    raise DeprecationWarning(
        "Synthetic interaction creation is deprecated. "
        "Please provide real interaction data in your JSON file under 'interactions' key."
    )


def load_interactions_from_input(inputs: Inputs) -> List[Tuple[Union[str, int], Union[str, int], int]]:
    """
    Load real interactions from input data
    
    Args:
        inputs: Inputs object with user, item, and interaction data
        
    Returns:
        interactions: List of (user_id, item_id, label) tuples where label is binary (0 or 1)
    """
    interactions_data = inputs.get_interactions()
    
    if not interactions_data:
        raise ValueError(
            "No interaction data found. Please provide interaction data in your JSON file "
            "under 'interactions' key with format: "
            "[{'user_id': 1, 'item_id': 101, 'interaction_type': 'purchase', 'timestamp': '...'}]"
        )
    
    interactions = []
    positive_count = 0
    negative_count = 0
    
    for interaction in interactions_data:
        user_id = interaction['user_id']
        item_id = interaction['item_id']
        
        # Use binary label if available, otherwise default to positive
        if 'label' in interaction:
            label = interaction['label']
        else:
            # Fallback: treat all interactions as positive
            label = 1
        
        interactions.append((user_id, item_id, label))
        
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    print(f"Loaded {len(interactions)} real interactions from input data")
    print(f"Positive interactions: {positive_count}")
    print(f"Negative interactions: {negative_count}")
    print(f"Positive ratio: {positive_count / len(interactions):.3f}")
    
    return interactions


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Recommendation Data Loader ===")
    
    # Load data using input_processor
    inputs = Inputs()
    
    # Configure minimal validation for testing
    inputs.configure_validators(image_check_files=False)
    
    # Load data from JSON (make sure test_input.json exists)
    try:
        result = inputs.load_from_json("test_input.json")
        print(f"Data loading successful: {result.is_valid}")
        
        if not result.is_valid:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        print(f"Loaded {len(inputs.get_user_data())} users and {len(inputs.get_item_data())} items")
        
        # Load real interactions from input data
        interactions = load_interactions_from_input(inputs=inputs)
        
        # Create data loaders
        train_loader, val_loader, _, _ = create_data_loaders(
            inputs=inputs,
            interactions=interactions,
            train_split=0.8,
            batch_size=4,
            negative_sampling_ratio=1.5
        )
        
        # Test data loading
        print("\n=== Testing Data Loading ===")
        for i, batch in enumerate(train_loader):
            print(f"\nBatch {i + 1}:")
            print(f"  User features keys: {list(batch['user_features'].keys())}")
            print(f"  Item features keys: {list(batch['item_features'].keys())}")
            print(f"  Labels: {batch['labels']}")
            print(f"  Batch size: {len(batch['labels'])}")
            
            # Print structure of first user feature type
            if batch['user_features']:
                first_feature_type = next(iter(batch['user_features'].keys()))
                print(f"  Sample user {first_feature_type}: {batch['user_features'][first_feature_type]}")
            
            if i >= 2:  # Only show first few batches
                break
        
        print("\n✅ Data loader testing completed successfully!")
        
    except FileNotFoundError:
        print("Error: test_input.json not found. Please create test data file first.")
        print("You can use the create_example_data() function from input_processor.py")
        
    except Exception as e:
        print(f"Error during testing: {e}")