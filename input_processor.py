import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
import warnings


class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)


class BaseFeatureInput(ABC):
    """Base class for all feature input types"""
    
    def __init__(self, feature_type: str):
        self.feature_type = feature_type
        self._schema_keys: Set[str] = set()
    
    @abstractmethod
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        """Validate feature data for a single entry"""
        pass
    
    def _validate_feature_schema_consistency(self, all_feature_data: List[Dict[str, Any]]) -> ValidationResult:
        """Ensure all entries have consistent schema for this feature type"""
        result = ValidationResult(is_valid=True)
        
        if not all_feature_data:
            return result
        
        # Get all unique keys across all entries for this feature type
        all_keys_per_entry = []
        for feature_data in all_feature_data:
            if isinstance(feature_data, dict):
                all_keys_per_entry.append(set(feature_data.keys()))
        
        if all_keys_per_entry:
            # Store the expected schema from first entry
            self._schema_keys = all_keys_per_entry[0]
            
            # Check consistency
            if len(set(tuple(sorted(keys)) for keys in all_keys_per_entry)) > 1:
                result.add_warning(f"Feature type '{self.feature_type}': Inconsistent keys across entries.")
        
        return result


class ImageFeatureInput(BaseFeatureInput):
    """Handler for image features"""
    
    def __init__(self, check_file_exists: bool = True):
        super().__init__("image")
        self.check_file_exists = check_file_exists
    
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(feature_data, dict):
            result.add_error(f"Entry {entry_id}: Image feature must be a dictionary, got {type(feature_data)}")
            return result
        
        # Validate each image path in the dictionary
        for image_key, image_path in feature_data.items():
            if image_path is None:
                continue  # Allow None values (sparse data)
            
            if not isinstance(image_path, str):
                result.add_error(f"Entry {entry_id}: Image path for '{image_key}' must be string, got {type(image_path)}")
                continue
            
            if not image_path.strip():
                result.add_error(f"Entry {entry_id}: Empty image path for '{image_key}'")
                continue
            
            if self.check_file_exists:
                path = Path(image_path)
                if not path.exists():
                    result.add_error(f"Entry {entry_id}: Image file does not exist for '{image_key}': {image_path}")
                elif not path.is_file():
                    result.add_error(f"Entry {entry_id}: Path is not a file for '{image_key}': {image_path}")
                else:
                    # Check if it's a common image format
                    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    if path.suffix.lower() not in valid_extensions:
                        result.add_warning(f"Entry {entry_id}: Unusual file extension for '{image_key}': {path.suffix}")
        
        return result


class TextFeatureInput(BaseFeatureInput):
    """Handler for text features"""
    
    def __init__(self):
        super().__init__("text")
    
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(feature_data, dict):
            result.add_error(f"Entry {entry_id}: Text feature must be a dictionary, got {type(feature_data)}")
            return result
        
        # Validate each text field in the dictionary
        for text_key, text_value in feature_data.items():
            if text_value is None:
                continue  # Allow None values (sparse data)
            
            # Skip validation for pre-tokenized fields
            if text_key.startswith('tokenized_'):
                continue
            
            if not isinstance(text_value, str):
                result.add_error(f"Entry {entry_id}: Text value for '{text_key}' must be string, got {type(text_value)}")
            elif not text_value.strip():
                result.add_warning(f"Entry {entry_id}: Empty text value for '{text_key}'")
        
        return result


class CategoricalFeatureInput(BaseFeatureInput):
    """Handler for categorical features"""
    
    def __init__(self, allowed_values: Optional[Dict[str, Set[str]]] = None):
        super().__init__("categorical")
        self.allowed_values = allowed_values or {}  # key -> allowed values
        self._discovered_values: Dict[str, Set[str]] = {}  # key -> discovered values
    
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(feature_data, dict):
            result.add_error(f"Entry {entry_id}: Categorical feature must be a dictionary, got {type(feature_data)}")
            return result
        
        # Validate each categorical field
        for cat_key, cat_value in feature_data.items():
            if cat_value is None:
                continue  # Allow None values (sparse data)
            
            if not isinstance(cat_value, (str, int, float)):
                result.add_error(f"Entry {entry_id}: Categorical value for '{cat_key}' must be string, int, or float, got {type(cat_value)}")
                continue
            
            # Convert to string for consistency
            str_value = str(cat_value)
            
            # Track discovered values
            if cat_key not in self._discovered_values:
                self._discovered_values[cat_key] = set()
            self._discovered_values[cat_key].add(str_value)
            
            # Check against allowed values if specified
            if cat_key in self.allowed_values and str_value not in self.allowed_values[cat_key]:
                result.add_error(f"Entry {entry_id}: Value '{str_value}' for '{cat_key}' not in allowed values: {self.allowed_values[cat_key]}")
        
        return result
    
    def get_unique_values(self, cat_key: str) -> Set[str]:
        """Get all unique values discovered for a specific categorical key"""
        return self._discovered_values.get(cat_key, set()).copy()
    
    def get_all_unique_values(self) -> Dict[str, Set[str]]:
        """Get all unique values for all categorical keys"""
        return {k: v.copy() for k, v in self._discovered_values.items()}


class ContinuousFeatureInput(BaseFeatureInput):
    """Handler for continuous numerical features"""
    
    def __init__(self, value_ranges: Optional[Dict[str, tuple]] = None):
        super().__init__("continuous")
        self.value_ranges = value_ranges or {}  # key -> (min_val, max_val)
    
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(feature_data, dict):
            result.add_error(f"Entry {entry_id}: Continuous feature must be a dictionary, got {type(feature_data)}")
            return result
        
        # Validate each continuous field
        for cont_key, cont_value in feature_data.items():
            if cont_value is None:
                continue  # Allow None values (sparse data)
            
            if not isinstance(cont_value, (int, float)):
                result.add_error(f"Entry {entry_id}: Continuous value for '{cont_key}' must be numeric, got {type(cont_value)}")
                continue
            
            # Check for special float values
            if isinstance(cont_value, float):
                if cont_value != cont_value:  # NaN check
                    result.add_error(f"Entry {entry_id}: NaN value not allowed for '{cont_key}'")
                    continue
                elif cont_value == float('inf') or cont_value == float('-inf'):
                    result.add_error(f"Entry {entry_id}: Infinite value not allowed for '{cont_key}'")
                    continue
            
            # Check range constraints
            if cont_key in self.value_ranges:
                min_val, max_val = self.value_ranges[cont_key]
                if min_val is not None and cont_value < min_val:
                    result.add_error(f"Entry {entry_id}: Value {cont_value} for '{cont_key}' below minimum {min_val}")
                if max_val is not None and cont_value > max_val:
                    result.add_error(f"Entry {entry_id}: Value {cont_value} for '{cont_key}' above maximum {max_val}")
        
        return result


class TemporalFeatureInput(BaseFeatureInput):
    """Handler for temporal/sequence features"""
    
    def __init__(self):
        super().__init__("temporal")
    
    def validate_feature_data(self, entry_id: str, feature_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(feature_data, dict):
            result.add_error(f"Entry {entry_id}: Temporal feature must be a dictionary, got {type(feature_data)}")
            return result
        
        # Validate each temporal field
        for temp_key, temp_value in feature_data.items():
            if temp_value is None:
                continue  # Allow None values (sparse data)
            
            if not isinstance(temp_value, list):
                result.add_error(f"Entry {entry_id}: Temporal value for '{temp_key}' must be a list, got {type(temp_value)}")
                continue
            
            if len(temp_value) == 0:
                result.add_warning(f"Entry {entry_id}: Empty temporal sequence for '{temp_key}'")
                continue
            
            # Validate sequence items - they can be any type depending on use case
            # For your example, they appear to be numbers (post IDs, times, etc.)
            for i, item in enumerate(temp_value):
                if item is None:
                    continue  # Allow None in sequences
                
                # Basic type checking - could be more specific based on requirements
                if not isinstance(item, (int, float, str, dict)):
                    result.add_warning(f"Entry {entry_id}: Unexpected type in '{temp_key}' sequence at index {i}: {type(item)}")
        
        return result


class Inputs:
    """Main class to handle user and item inputs"""
    
    def __init__(self):
        # Feature validators
        self.image_validator = ImageFeatureInput()
        self.text_validator = TextFeatureInput()
        self.categorical_validator = CategoricalFeatureInput()
        self.continuous_validator = ContinuousFeatureInput()
        self.temporal_validator = TemporalFeatureInput()
        
        # Data storage
        self.user_data: List[Dict[str, Any]] = []
        self.item_data: List[Dict[str, Any]] = []
        self.interactions: List[Dict[str, Any]] = []  # Store user-item interactions
        
        # Expected feature types
        self.expected_feature_types = {'image', 'text', 'categorical', 'continuous', 'temporal'}
    
    def configure_validators(self, 
                           image_check_files: bool = True,
                           categorical_allowed_values: Optional[Dict[str, Set[str]]] = None,
                           continuous_ranges: Optional[Dict[str, tuple]] = None):
        """Configure validation parameters"""
        self.image_validator = ImageFeatureInput(check_file_exists=image_check_files)
        self.categorical_validator = CategoricalFeatureInput(allowed_values=categorical_allowed_values)
        self.continuous_validator = ContinuousFeatureInput(value_ranges=continuous_ranges)
    
    def load_from_json(self, json_path: Union[str, Path]) -> ValidationResult:
        """Load data from JSON file"""
        result = ValidationResult(is_valid=True)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            result.add_error(f"JSON file not found: {json_path}")
            return result
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")
            return result
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
            return result
        
        return self.load_from_dict(data)
    
    def load_from_dict(self, data: Dict[str, Any]) -> ValidationResult:
        """Load data from dictionary with 'user_data' and 'item_data' keys"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data, dict):
            result.add_error(f"Expected dictionary at root level, got {type(data)}")
            return result
        
        # Load user data
        if 'user_data' in data:
            user_result = self._load_data_section(data['user_data'], 'user')
            result.errors.extend(user_result.errors)
            result.warnings.extend(user_result.warnings)
            if not user_result.is_valid:
                result.is_valid = False
        else:
            result.add_warning("No 'user_data' found in input")
        
        # Load item data
        if 'item_data' in data:
            item_result = self._load_data_section(data['item_data'], 'item')
            result.errors.extend(item_result.errors)
            result.warnings.extend(item_result.warnings)
            if not item_result.is_valid:
                result.is_valid = False
        else:
            result.add_warning("No 'item_data' found in input")
        
        # Load interactions data
        if 'interactions' in data:
            interactions_result = self._load_interactions(data['interactions'])
            result.errors.extend(interactions_result.errors)
            result.warnings.extend(interactions_result.warnings)
            if not interactions_result.is_valid:
                result.is_valid = False
        else:
            result.add_warning("No 'interactions' found in input - this is required for training")
        
        return result
    
    def _load_data_section(self, data_section: List[Dict[str, Any]], section_type: str) -> ValidationResult:
        """Load and validate a data section (user or item)"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data_section, list):
            result.add_error(f"{section_type}_data must be a list, got {type(data_section)}")
            return result
        
        validated_entries = []
        
        # Validate each entry
        for i, entry in enumerate(data_section):
            entry_result = self._validate_single_entry(entry, f"{section_type}_{i}")
            result.errors.extend(entry_result.errors)
            result.warnings.extend(entry_result.warnings)
            
            if entry_result.is_valid:
                validated_entries.append(entry)
            else:
                result.is_valid = False
        
        # Store validated data
        if section_type == 'user':
            self.user_data = validated_entries
        else:
            self.item_data = validated_entries
        
        # Validate schema consistency across entries
        consistency_result = self._validate_section_consistency(validated_entries, section_type)
        result.errors.extend(consistency_result.errors)
        result.warnings.extend(consistency_result.warnings)
        if not consistency_result.is_valid:
            result.is_valid = False
        
        return result
    
    def _validate_single_entry(self, entry: Dict[str, Any], entry_id: str) -> ValidationResult:
        """Validate a single data entry"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(entry, dict):
            result.add_error(f"Entry {entry_id}: Must be a dictionary, got {type(entry)}")
            return result
        
        # Check for required ID field
        if 'user_id' not in entry and 'item_id' not in entry:
            result.add_error(f"Entry {entry_id}: Must have either 'user_id' or 'item_id'")
        
        # Validate each feature type
        for feature_type, feature_data in entry.items():
            if feature_type in ['user_id', 'item_id']:
                continue  # Skip ID fields
            
            if feature_type not in self.expected_feature_types:
                result.add_warning(f"Entry {entry_id}: Unknown feature type '{feature_type}'")
                continue
            
            if feature_data is None:
                continue  # Allow None for entire feature types (sparse data)
            
            # Validate based on feature type
            if feature_type == 'image':
                feature_result = self.image_validator.validate_feature_data(entry_id, feature_data)
            elif feature_type == 'text':
                feature_result = self.text_validator.validate_feature_data(entry_id, feature_data)
            elif feature_type == 'categorical':
                feature_result = self.categorical_validator.validate_feature_data(entry_id, feature_data)
            elif feature_type == 'continuous':
                feature_result = self.continuous_validator.validate_feature_data(entry_id, feature_data)
            elif feature_type == 'temporal':
                feature_result = self.temporal_validator.validate_feature_data(entry_id, feature_data)
            else:
                continue
            
            result.errors.extend(feature_result.errors)
            result.warnings.extend(feature_result.warnings)
            if not feature_result.is_valid:
                result.is_valid = False
        
        return result
    
    def _validate_section_consistency(self, entries: List[Dict[str, Any]], section_type: str) -> ValidationResult:
        """Validate consistency across all entries in a section"""
        result = ValidationResult(is_valid=True)
        
        if not entries:
            return result
        
        # Check feature type consistency
        feature_types_per_entry = []
        for entry in entries:
            feature_types = set(key for key in entry.keys() if key not in ['user_id', 'item_id'])
            feature_types_per_entry.append(feature_types)
        
        if feature_types_per_entry:
            # All entries should have the same feature types available (though values can be sparse)
            all_feature_types = set().union(*feature_types_per_entry)
            for i, feature_types in enumerate(feature_types_per_entry):
                missing_features = all_feature_types - feature_types
                if missing_features:
                    result.add_warning(f"{section_type} entry {i}: Missing feature types: {missing_features}")
        
        return result
    
    def _load_interactions(self, interactions_data: List[Dict[str, Any]]) -> ValidationResult:
        """Load and validate interactions data"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(interactions_data, list):
            result.add_error(f"Expected list for interactions, got {type(interactions_data)}")
            return result
        
        self.interactions = []
        
        for i, interaction in enumerate(interactions_data):
            if not isinstance(interaction, dict):
                result.add_error(f"Interaction {i}: Expected dictionary, got {type(interaction)}")
                continue
            
            # Validate required fields
            required_fields = {'user_id', 'item_id'}
            missing_fields = required_fields - set(interaction.keys())
            if missing_fields:
                result.add_error(f"Interaction {i}: Missing required fields: {missing_fields}")
                continue
            
            # Validate data types
            try:
                user_id = interaction['user_id']
                item_id = interaction['item_id'] 
                
                # Optional fields
                interaction_type = interaction.get('interaction_type', 'implicit')
                timestamp = interaction.get('timestamp', None)
                
                # Create clean interaction record
                clean_interaction = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'interaction_type': interaction_type,
                    'timestamp': timestamp
                }
                
                self.interactions.append(clean_interaction)
                
            except (ValueError, TypeError) as e:
                result.add_error(f"Interaction {i}: Invalid data types - {e}")
        
        print(f"Loaded {len(self.interactions)} interactions")
        return result
    
    def get_user_data(self) -> List[Dict[str, Any]]:
        """Get all user data entries"""
        return self.user_data.copy()
    
    def get_item_data(self) -> List[Dict[str, Any]]:
        """Get all item data entries"""
        return self.item_data.copy()
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """Get all interaction data entries"""
        return self.interactions.copy()
    
    def get_user_by_id(self, user_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get a specific user by ID"""
        for user in self.user_data:
            if user.get('user_id') == user_id:
                return user
        return None
    
    def get_item_by_id(self, item_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID"""
        for item in self.item_data:
            if item.get('item_id') == item_id:
                return item
        return None
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded features"""
        stats = {
            'user_count': len(self.user_data),
            'item_count': len(self.item_data),
            'categorical_values': self.categorical_validator.get_all_unique_values(),
        }
        return stats


# Example usage and helper functions
def create_example_data():
    """Create example data matching your format"""
    return {
        "user_data": [
            {
                "user_id": 1,
                "image": {"profile_pic": "/image/profile1.jpg"},
                "text": {"bio": "I am a college guy in Texas", "summary": "some text"},
                "categorical": {"country": "USA", "gender": "male", "state": "texas"},
                "continuous": {"age": 20.5, "income": 25000.0},
                "temporal": {
                    "prev_50_posts": [34, 56, 7646, 342, 123, 456],
                    "last_10_session_times": [5, 11, 15, 2, 3, 5, 8, 12, 4, 7]
                }
            },
            {
                "user_id": 2,
                "image": {"profile_pic": "/image/profile2.jpg"},
                "text": {"bio": "Artist from California", "summary": None},
                "categorical": {"country": "USA", "gender": "female", "state": "california"},
                "continuous": {"age": 28.0, "income": 45000.0},
                "temporal": {
                    "prev_50_posts": [12, 89, 234, 567],
                    "last_10_session_times": [10, 5, 20, 15, 8]
                }
            }
        ],
        "item_data": [
            {
                "item_id": 101,
                "image": {"main_image": "/items/item101.jpg", "thumbnail": "/items/thumb101.jpg"},
                "text": {"title": "Awesome Product", "description": "This is a great product"},
                "categorical": {"category": "electronics", "brand": "TechCorp", "condition": "new"},
                "continuous": {"price": 99.99, "rating": 4.5, "weight": 1.2},
                "temporal": {
                    "price_history": [89.99, 94.99, 99.99, 95.99],
                    "view_counts_daily": [10, 15, 23, 18, 31, 27, 19]
                }
            }
        ]
    }


if __name__ == "__main__":
    # Example usage
    inputs = Inputs()
    
    # Configure minimal validation for testing
    inputs.configure_validators(image_check_files=False)
    
    # Load example data
    #example_data = create_example_data()
    #result = inputs.load_from_dict(example_data)
    result = inputs.load_from_json("test_input.json")

    print(f"Validation successful: {result.is_valid}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Access loaded data
    print(f"\nLoaded {len(inputs.get_user_data())} users and {len(inputs.get_item_data())} items")
    
    # Get specific user
    user1 = inputs.get_user_by_id(1)
    if user1:
        print(f"User 1 bio: {user1['text']['bio']}")