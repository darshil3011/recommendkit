# Tests

This directory contains pytest test scripts for the recommendation system.

## Test Scripts

### 1. `test_training.py`
Tests the training loop with a small dataset.
- Creates a small dataset (50 users, 50 items)
- Trains for 1 epoch
- Verifies training completes successfully

### 2. `test_save_load.py`
Tests model save and load functionality.
- Creates and trains a small model
- Saves the model to a temporary directory
- Loads the model back
- Verifies all parameters are saved and loaded correctly
- Verifies weights match exactly

### 3. `test_determinism.py`
Tests inference determinism.
- Loads a trained model
- Runs inference on the same user-item pair twice
- Verifies results are identical (deterministic)

## Running Tests

### Using pytest (Recommended)

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_determinism.py

# Run with verbose output
pytest tests/ -v

# Run only fast tests (skip slow ones)
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Running Individual Tests

```bash
# Run specific test function
pytest tests/test_determinism.py::test_determinism

# Run with more detailed output
pytest tests/ -vv
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:
- `inputs`: Loads input data once for all tests (session-scoped)
- `small_dataset`: Creates a small subset of the dataset for testing

## Requirements

- **pytest**: Install with `pip install pytest`
- A trained model must exist at `models/latest_model.pt` and `models/latest_model_config.json` for `test_determinism.py`
- The dataset must be available at `datasets/post_recommendation/updated_output_split.json`

## Test Markers

Tests are marked with:
- `@pytest.mark.slow`: Tests that take longer to run (training, save/load)
- Can be skipped with `-m "not slow"`

## Configuration

See `pytest.ini` for pytest configuration options.

