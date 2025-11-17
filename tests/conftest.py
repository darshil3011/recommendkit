"""
Pytest configuration and shared fixtures
"""
import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_processor import Inputs


@pytest.fixture(scope="session")
def inputs():
    """Load input data once for all tests"""
    inputs_obj = Inputs()
    inputs_obj.configure_validators(image_check_files=False)
    result = inputs_obj.load_from_json("datasets/post_recommendation/updated_output_split.json")
    
    if not result.is_valid:
        pytest.skip(f"Failed to load data: {result.errors}")
    
    return inputs_obj


@pytest.fixture
def small_dataset(inputs, max_users=50, max_items=50):
    """Create a small subset of the dataset for testing"""
    user_data = inputs.get_user_data()
    item_data = inputs.get_item_data()
    interactions = inputs.get_interactions()
    
    # Get unique user and item IDs from interactions
    user_ids = set()
    item_ids = set()
    for user_id, item_id, label in interactions:
        user_ids.add(user_id)
        item_ids.add(item_id)
        if len(user_ids) >= max_users and len(item_ids) >= max_items:
            break
    
    # Limit to max_users and max_items
    user_ids = list(user_ids)[:max_users]
    item_ids = list(item_ids)[:max_items]
    
    # Filter data
    filtered_user_data = [u for u in user_data if u['user_id'] in user_ids]
    filtered_item_data = [i for i in item_data if i['item_id'] in item_ids]
    filtered_interactions = [
        (u, i, l) for u, i, l in interactions
        if u in user_ids and i in item_ids
    ]
    
    return filtered_user_data, filtered_item_data, filtered_interactions


