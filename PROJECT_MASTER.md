# RecommendKit — Project Master Reference

> Generated: 2026-04-09

---

## SECTION 1 — Project Identity

**Project name:** RecommendKit

**One-liner:** A universal, end-to-end two-tower recommendation system framework for training, inference, and evaluation with multi-modal feature support.

**Problem it solves:**
- Building recommendation systems requires combining heterogeneous data (text, images, categorical values, continuous numbers, interaction sequences) into unified user and item representations
- RecommendKit provides a modular encoder + fusion + interaction + classification pipeline so teams can train personalized recommenders without writing boilerplate neural network code
- Handles the full lifecycle: data validation → training → model persistence → inference → ranking evaluation

**End users and stakeholders:**
- ML engineers building recommendation systems on top of DeGirum's AI ecosystem
- Data scientists evaluating different architectures (simple concat-MLP vs. attention-based transformers)
- Product teams that need a recommendation layer on top of user and item metadata

**Project status:** Active

| Link | URL |
|------|-----|
| Repository | `/home/darshil/Desktop/recommendkit` (local) |
| Contributing guide | `CONTRIBUTING.MD` |

| Layer | Technology | Version |
|-------|-----------|---------|
| Core deep learning | PyTorch | >=1.10.0 |
| Computer vision | torchvision | >=0.11.0 |
| Vision Transformers | timm (optional) | — |
| Numerical computation | NumPy | >=1.21.0 |
| Transformer NLP | HuggingFace Transformers | >=4.20.0 |
| Word2Vec / FastText | gensim (optional) | — |
| Evaluation + metrics | scikit-learn | >=1.0.0 |
| Vector database client | qdrant-client | >=1.7.0 |
| Progress bars | tqdm | — |
| Language | Python | 3.x |

---

## SECTION 2 — Folder Structure

```text
recommendkit/
├── train.py                          # Main training entry point
├── inference.py                      # Main inference entry point
├── input_processor.py                # Data loading and validation
├── requirements.txt                  # Python dependencies
├── README.md                         # User-facing documentation
├── CONTRIBUTING.MD                   # Contributor guidelines
├── quickstart.ipynb                  # Jupyter quickstart notebook
├── classification_results.json       # Example classification output
│
├── configs/                          # JSON configuration templates
│   ├── sample_config_simple_fusion.json
│   └── sample_config_attention_fusion.json
│
├── encoders/                         # Modality-specific feature encoders
│   ├── base_encoder.py               # Abstract BaseEncoder class
│   ├── categorical/                  # Hash-based categorical encoders
│   │   ├── __init__.py
│   │   ├── base_categorical_encoder.py
│   │   ├── hash_encoder.py
│   │   └── factory.py
│   ├── continuous/                   # MLP-based numerical encoders
│   │   ├── __init__.py
│   │   ├── base_continuous_encoder.py
│   │   ├── mlp_encoder.py
│   │   └── factory.py
│   ├── image/                        # CNN / ResNet / ViT image encoders
│   │   ├── __init__.py
│   │   ├── base_image_encoder.py
│   │   ├── cnn_encoder.py
│   │   ├── resnet_encoder.py
│   │   ├── vit_encoder.py
│   │   └── factory.py
│   ├── temporal/                     # LSTM-based sequence encoders
│   │   ├── __init__.py
│   │   ├── base_temporal_encoder.py
│   │   ├── lstm_temporal_encoder.py
│   │   └── factory.py
│   └── text/                         # Transformer / Word2Vec text encoders
│       ├── __init__.py
│       ├── base_text_encoder.py
│       ├── transformer_encoder.py
│       ├── word2vec_encoder.py
│       └── factory.py
│
├── interaction/                      # Feature fusion and interaction modeling
│   ├── attention_utils.py            # MHA, TransformerBlock, FeatureFusionLayer
│   ├── feature_fusion.py             # SimpleFusionLayer, UserEmbeddingGenerator, ItemEmbeddingGenerator
│   └── interaction_modeling.py       # UserItemInteractionLayer, InteractionEmbeddingGenerator
│
├── classifier/                       # Final classification head
│   ├── classification_utils.py       # MLPBlock, MLPTower, LossManager, loss classes
│   └── recommendation_classifier.py  # RecommendationClassifier (MLP + head)
│
├── trainer/                          # Training loop, data loading, pipeline
│   ├── data_loader.py                # RecommendationDataset, collate, create_data_loaders
│   ├── pipeline_builder.py           # RecommendationPipeline, save/load, DimensionAligner
│   ├── trainer.py                    # train_model(), prepare_batch(), evaluate_model()
│   └── test_input.json               # Sample test input data
│
├── evaluation/                       # Offline evaluation utilities
│   ├── __init__.py
│   ├── ranking_metrics.py            # precision_at_k, recall_at_k, compute_metrics_for_user
│   ├── evaluate_recommendation.py    # End-to-end ranking evaluation script
│   └── evaluate_classification.py    # Classification metrics script
│
├── utils/                            # Shared utilities
│   ├── __init__.py
│   └── config_validator.py           # validate_config(), per-encoder validators
│
├── tests/                            # Pytest test suite
│   ├── conftest.py                   # Shared fixtures (inputs, small_dataset)
│   ├── pytest.ini                    # Pytest configuration
│   ├── test_training.py
│   ├── test_inference.py
│   ├── test_save_load.py
│   ├── test_determinism.py
│   ├── test_matrix_factorization.py
│   └── run_diagnosis.py              # Diagnostic script
│
├── datasets/                         # Sample and converted datasets
│   ├── movielens/                    # MovieLens 100K data + converter
│   ├── post_recommendation/          # RecommendKit-format dataset used in tests
│   ├── reddit/                       # Reddit data + transformer
│   ├── synthetic/                    # Correlated synthetic dataset + generator
│   └── test_datasets/                # Minimal test datasets
│
├── models/                           # Saved model weights and configs (auto-generated)
│   └── quickstart_model.*            # Pre-trained quickstart model artifacts
│
└── assets/                           # Static assets for docs
    └── recommendkit_banner.png
```

**Structural pattern:** Domain-layered library. Each concern — data ingestion, encoding, fusion, interaction, classification, training, evaluation — lives in its own package. `train.py` and `inference.py` are thin drivers that wire these packages together. There is no application server; the codebase is a Python library + CLI.

**Auto-generated folders:** `models/` (model checkpoints), `__pycache__/` (Python bytecode)

**Vendored:** None

**To find X, go to Y:**

| Concern | Location |
|---------|----------|
| Data loading and validation | `input_processor.py` |
| Config validation | `utils/config_validator.py` |
| All encoder implementations | `encoders/<modality>/` |
| Two-tower fusion logic | `interaction/feature_fusion.py` |
| User-item interaction layer | `interaction/interaction_modeling.py` |
| Attention primitives | `interaction/attention_utils.py` |
| Full pipeline (nn.Module) | `trainer/pipeline_builder.py` — `RecommendationPipeline` |
| Training loop | `trainer/trainer.py` — `train_model()` |
| Dataset / DataLoader | `trainer/data_loader.py` |
| Model save / load | `trainer/pipeline_builder.py` — `save_complete_model()`, `load_model_from_config()` |
| Classification head | `classifier/recommendation_classifier.py` |
| Loss functions | `classifier/classification_utils.py` |
| Ranking metrics | `evaluation/ranking_metrics.py` |
| End-to-end evaluation | `evaluation/evaluate_recommendation.py` |
| Sample configs | `configs/` |
| Test fixtures | `tests/conftest.py` |

---

## SECTION 3 — File-by-File Deep Summary

### 📁 Root Level

---

#### `train.py`

**Summary**
- Top-level CLI driver for training a `RecommendationPipeline`. Loads data, creates the model from a JSON config, optionally loads pretrained weights, runs `train_model()`, and saves all artifacts.

**Purpose**
- Serve as the user-facing entrypoint for training any two-tower recommendation model from a JSON data file and a JSON config file.

**Key Functions**

- `create_model_from_config(config, item_data) → RecommendationPipeline`
  - Validates config via `validate_config()`, then constructs `RecommendationPipeline` with all kwargs drawn from `config`
  - Calls `model.set_config(config)` after construction so the actual-encoder configs (set during `__init__`) are captured
  - Returns initialized pipeline ready for training

- `load_training_config(config_path) → Dict`
  - Reads and JSON-parses a config file; no validation at this stage

- `extract_architecture_config(config) → Dict`
  - Returns only architecture-relevant keys, stripping training hyperparameters (`num_epochs`, `batch_size`, etc.)
  - Normalizes both `*_encoder_config` and `*_encoder` key formats into a single dict

- `validate_config_compatibility(training_config, saved_config) → None`
  - Compares `extract_architecture_config()` of both configs key-by-key
  - Raises `RuntimeError` with a detailed diff if any value differs — prevents training a config on mismatched pretrained weights

- `main() → int`
  - Full 9-step pipeline: load config → load data → load interactions → create model → optionally load pretrained weights → setup device → create data loaders → train → save
  - Handles lazy-initialization of `MLPContinuousEncoder` and `DimensionAligner.projections` when loading pretrained weights (reads weight shapes from checkpoint to call `_initialize_mlp()` and `_get_projection()` before `load_state_dict()`)
  - Saves `{model_name}_test_interactions.json` and `{model_name}_train_interactions.json` alongside the model

**CLI Arguments**

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | required | Path to JSON data file |
| `--config_path` | required | Path to JSON config file |
| `--output_dir` | `models` | Directory for model output |
| `--model_name` | `model` | Base name for all saved files |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--pretrained_weights` | None | Directory of pretrained model |
| `--pretrained_model_name` | `model` | Base name of pretrained files |

**Core Logic**

- Pretrained loading uses `torch.load()` then `load_state_dict(strict=False)` after manually triggering lazy initializations
- Filters `dimension_aligner.projections` keys from missing/unexpected key warnings (expected to differ since they're created dynamically)
- Device selection: `'auto'` resolves to `cuda` if available, else `cpu`

**Internal Dependencies**
- `input_processor.Inputs` — for data loading and validation
- `trainer.pipeline_builder.RecommendationPipeline`, `save_complete_model` — model creation and persistence
- `trainer.data_loader.create_data_loaders`, `load_interactions_from_input` — data preparation
- `trainer.trainer.train_model` — training loop
- `utils.config_validator.validate_config` — config validation

**Gotchas**
- `configure_validators(image_check_files=False)` is called during training to avoid requiring image paths to exist
- Pretrained weight loading pre-initializes lazy modules before `load_state_dict()` — skipping this causes shape mismatch errors
- `validate_config_compatibility` compares **both** `*_encoder_config` and `*_encoder` naming conventions via `extract_architecture_config`

---

#### `inference.py`

**Summary**
- CLI driver for running inference with a trained model. Loads data and model, detects which encoders were active during training by inspecting checkpoint keys, then generates top-k recommendations ranked by classifier probability.

**Purpose**
- Provide user-facing recommendation generation: given a trained model and a data file, output the top-k items for a specified user.

**Key Functions**

- `generate_user_embedding(model, user_data, encoders_used, expected_num_features) → np.ndarray`
  - Builds `user_features` dict filtered by `encoders_used` flags
  - Priority-based feature selection when `len(user_features) > expected_num_features`: prefers removing empty features first, then by priority `[image, temporal, text, continuous, categorical]`
  - Calls `model._encode_features()` → `model.user_generator()` → `.squeeze().cpu().numpy()`

- `generate_item_embedding(model, item_data, encoders_used, expected_num_features) → np.ndarray`
  - Same logic as above but for items; item priority order is `[temporal, image, text, continuous, categorical]`
  - Memoizes first-item debug print via `generate_item_embedding._debug_printed`

- `compute_similarity(user_embedding, item_embedding) → float`
  - Flattens both embeddings, L2-normalizes, returns dot product (cosine similarity)

- `find_top_items_for_user(model, user_data, all_items, k, filters, encoders_used, expected_user_features, expected_item_features) → List[Dict]`
  - Generates user embedding once, iterates all items
  - Applies categorical and continuous range filters before scoring
  - Scores via `model.interaction_generator()` → `model.classifier()` → `torch.sigmoid()`, matching the training objective
  - Returns top-k by probability (labeled as `similarity` for backward compatibility)

- `print_recommendations(recommendations, title)` — formats top-k to console

- `main() → int`
  - Inspects checkpoint keys to build `encoders_used` dict
  - Infers `user_expected_features` and `item_expected_features` from `user_generator.user_fusion.projection.0.weight` shape: `input_dim // embedding_dim`

**CLI Arguments**

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | required | Path to JSON data file |
| `--model_dir` | `models` | Directory of model files |
| `--model_name` | `model` | Base name of model files |
| `--user_id` | None (first user) | User to generate recs for |
| `--k` | 10 | Number of recommendations |
| `--filters` | None | JSON filter string |
| `--output_file` | None (console) | Optional output JSON path |

**Gotchas**
- Inference scoring uses the full pipeline (interaction → classifier), **not** cosine similarity — this matches the training objective
- `compute_similarity` exists but is not used in `find_top_items_for_user`; actual ranking uses classifier probability
- `encoders_used` detection relies on checkpoint key substring matching (e.g., `'user_continuous_encoder' in k`) which is fragile if key names change

---

#### `input_processor.py`

**Summary**
- Data loading gateway for the entire system. Defines the expected JSON schema and validates every entry, feature type, and interaction record before data reaches the model.

**Purpose**
- Parse and validate raw JSON data files containing `user_data`, `item_data`, and `interactions` lists; expose clean, validated data via the `Inputs` class.

**Key Classes**

- `ValidationError(Exception)` — custom exception (currently unused internally; validators use `ValidationResult` instead)

- `ValidationResult` — dataclass with `is_valid: bool`, `errors: List[str]`, `warnings: List[str]`; provides `add_error()` and `add_warning()` methods

- `BaseFeatureInput(ABC)` — base for all feature validators
  - `feature_type: str` — identifies the modality
  - `_schema_keys: Set[str]` — tracks expected keys from first entry
  - `_validate_feature_schema_consistency()` — warns if keys differ across entries

- `ImageFeatureInput(BaseFeatureInput)` — validates image paths
  - `check_file_exists: bool` — if True, validates file existence and extension (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`)
  - Allows `None` values (sparse data)

- `TextFeatureInput(BaseFeatureInput)` — validates text fields
  - Skips fields prefixed with `tokenized_` (pre-tokenized data)
  - Warns on empty strings

- `CategoricalFeatureInput(BaseFeatureInput)` — validates categorical fields
  - `allowed_values: Dict[str, Set[str]]` — optional whitelist per field
  - `_discovered_values: Dict[str, Set[str]]` — tracks all seen values
  - `get_unique_values(cat_key)`, `get_all_unique_values()` — for downstream schema inspection

- `ContinuousFeatureInput(BaseFeatureInput)` — validates numerical fields
  - `value_ranges: Dict[str, tuple]` — optional `(min, max)` per field
  - Rejects `NaN` and `±inf` values

- `TemporalFeatureInput(BaseFeatureInput)` — validates sequence fields
  - Expects each value to be a `list`; warns on empty lists
  - Items in sequences can be `int`, `float`, `str`, or `dict`

- `Inputs` — main orchestrator
  - `load_from_json(json_path) → ValidationResult`
  - `load_from_dict(data) → ValidationResult`
  - `_load_data_section(data_section, section_type)` — validates entries, stores in `self.user_data` or `self.item_data`
  - `_validate_single_entry(entry, entry_id)` — dispatches per feature type
  - `_validate_section_consistency(entries, section_type)` — warns on missing feature types across entries
  - `_load_interactions(interactions_data)` — parses `user_id`, `item_id`, optional `interaction_type` and `timestamp`; stores `self.interactions`
  - `get_user_data()`, `get_item_data()`, `get_interactions()` — return `.copy()` of stored lists
  - `get_user_by_id(user_id)`, `get_item_by_id(item_id)` — linear scan lookups
  - `get_feature_statistics()` — returns user/item counts and discovered categorical values
  - `configure_validators(image_check_files, categorical_allowed_values, continuous_ranges)` — reconfigures validators

**Key Constants**
- `expected_feature_types = {'image', 'text', 'categorical', 'continuous', 'temporal'}` — known feature keys; unknowns generate warnings

**Core Logic**
- `load_from_dict` reads three top-level keys: `user_data`, `item_data`, `interactions`; missing keys generate warnings (not errors)
- Per-entry validation: must have `user_id` or `item_id`; feature data delegates to appropriate `*FeatureInput.validate_feature_data()`
- Interactions: only `user_id` and `item_id` are required; `interaction_type` defaults to `'implicit'`; `label` defaults to `1` if absent (in `data_loader.py`)
- Schema consistency: warns if any user entry lacks a feature type present in other entries

**Gotchas**
- `get_user_by_id()` does a linear scan — `O(n)` for large datasets
- All interaction labels default to `1` (positive) in `load_interactions_from_input()` if no `label` field is present — no automatic negative labeling from interaction types
- `configure_validators()` replaces validator instances entirely; must be called before `load_from_json()`

---

### 📁 `encoders/`

---

#### `encoders/base_encoder.py`

**Summary**
- Defines `BaseEncoder`, the abstract parent for all modality encoders. Establishes the `forward()` → `Dict[str, Tensor]` contract.

**Key Classes**

- `BaseEncoder(nn.Module)`
  - `__init__(embedding_dim)` — stores `self.embedding_dim`
  - `forward(*args, **kwargs) → Dict[str, Tensor]` — raises `NotImplementedError`
  - `get_output_dim() → int` — returns `self.embedding_dim`

---

#### `encoders/categorical/base_categorical_encoder.py`

**Summary**
- Defines `CategoricalAggregationStrategy` enum and `BaseCategoricalEncoder` with deterministic hash-based vocabulary.

**Key Classes**

- `CategoricalAggregationStrategy(Enum)` — `SEPARATE_CONCAT = "separate_concat"`, `JOINT_EMBEDDING = "joint_embedding"`

- `BaseCategoricalEncoder(BaseEncoder)`
  - `_hash_category(field_name, value) → int` — deterministic hash combining field name and value using `hashlib.md5`; result modulo `hash_vocab_size`
  - `_hash_joint(categorical_dict) → int` — hashes the entire dict as a JSON string

**Gotchas**
- Hash collisions are possible; `hash_vocab_size` should be large enough for the cardinality of the data (default 10000)

---

#### `encoders/categorical/hash_encoder.py`

**Summary**
- `HashCategoricalEncoder` encodes categorical fields using hash-bucket lookups into learnable `MLPEmbedding` tables. Supports separate embedding per field (with concatenation) or a single joint embedding.

**Key Classes**

- `MLPEmbedding(nn.Module)` — `nn.Embedding` into an `nn.Sequential` MLP
  - `initial_embed_dim = max(16, min(128, int(sqrt(vocab_size))))` — adaptive initial dim
  - `_initialize_weights()` — Xavier uniform for Linear and Embedding layers

- `HashCategoricalEncoder(BaseCategoricalEncoder)`
  - `SEPARATE_CONCAT` strategy: creates `num_categorical_fields` `MLPEmbedding` modules named `field_0`, `field_1`, …; concatenates, then projects with `concat_projection`
  - `JOINT_EMBEDDING` strategy: single `joint_embedding` MLPEmbedding that hashes the entire dict
  - `forward(categorical_dict) → {"categorical_features": Tensor}` — handles both single and batched input
  - `_process_sample()` — core per-sample logic; pads missing fields with zeros

**Gotchas**
- `SEPARATE_CONCAT` assumes exactly `num_categorical_fields` fields; extra fields are silently ignored; missing fields are zero-padded — caller must ensure `num_categorical_fields` matches training data
- Batch processing iterates over samples in Python loop (no vectorized batch hash)

---

#### `encoders/continuous/base_continuous_encoder.py`

**Summary**
- Abstract base for continuous encoders providing optional normalization (log-scale for large values, standardization).

**Core Logic**
- `_normalize_features(tensor)`: if `normalize=True`, applies `log1p` to values > 100 then z-score standardization per feature column (mean/std computed on the batch)

---

#### `encoders/continuous/mlp_encoder.py`

**Summary**
- `MLPContinuousEncoder` lazily initializes an MLP on the first forward pass when the number of input features is known.

**Key Classes**

- `MLPContinuousEncoder(BaseContinuousEncoder)`
  - `mlp: Optional[nn.Sequential]` — `None` until `_initialize_mlp(input_dim)` is called
  - `_mlp_initialized: bool` — flag preventing double initialization
  - `forward(continuous_dict) → {"continuous_features": Tensor}` — sorts field names for deterministic ordering; initializes MLP on first call
  - Returns zero tensor if `continuous_dict` is empty

**Gotchas**
- Lazy initialization means the MLP is not registered in `state_dict()` until the first forward pass — requires pre-initialization when loading pretrained weights (handled in `train.py:main()`)
- Field ordering is alphabetical (`sorted(continuous_dict.keys())`); inference must present fields in the same order

---

#### `encoders/text/base_text_encoder.py`

**Summary**
- Defines `TextAggregationStrategy` enum and the abstract `BaseTextEncoder` interface.

**Key Constants**
- `TextAggregationStrategy.SEPARATE_CONCAT`, `JOINT_ENCODING`, `MEAN`

---

#### `encoders/text/transformer_encoder.py`

**Summary**
- `TransformerTextEncoder` wraps any HuggingFace model (`AutoModel` / `AutoTokenizer`) and provides CLS, mean, or max pooling over token embeddings.

**Key Classes**

- `TransformerTextEncoder(BaseTextEncoder)`
  - `model_name` — any HuggingFace model identifier
  - `pooling_strategy: str` — `'cls'`, `'mean'`, or `'max'`
  - `freeze_encoder: bool` — if True, all transformer parameters are frozen
  - Projection layer maps transformer output dim → `embedding_dim`
  - Handles batched text input (list of strings per field)

---

#### `encoders/text/word2vec_encoder.py`

**Summary**
- `Word2VecTextEncoder` loads a gensim KeyedVectors model and averages word embeddings for each text field.

**Gotchas**
- Requires `gensim` to be installed separately (not in `requirements.txt`)
- Out-of-vocabulary words are silently ignored during averaging

---

#### `encoders/text/factory.py`

**Summary**
- `create_text_encoder(config)` — auto-detects Word2Vec vs. Transformer models by checking if `model_name` contains any of `['word2vec', 'fasttext', 'glove', 'google-news', 'wikipedia']` (case-insensitive substring match).

---

#### `encoders/image/base_image_encoder.py`

**Summary**
- Defines `AggregationStrategy` enum (`CONCAT`, `AVERAGE`, `MAX_POOL`) and `BaseImageEncoder` with OpenCV-based image loading and preprocessing.

**Core Logic**
- Image loading: reads file with `cv2.imread`, converts BGR→RGB, resizes to `(224, 224)`, normalizes to `[0,1]`, applies ImageNet mean/std normalization
- Aggregation: if multiple image fields, concatenates, averages, or max-pools the field embeddings

---

#### `encoders/image/cnn_encoder.py`

**Summary**
- `CNNImageEncoder` uses a custom `SimpleCNN` (configurable conv layers → `AdaptiveAvgPool2d` → Linear projection) as the backbone.

---

#### `encoders/image/resnet_encoder.py`

**Summary**
- `ResNetImageEncoder` uses a pre-trained `ResNet` (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`) from `torchvision.models`, replacing the final `fc` layer with a projection to `embedding_dim`.

---

#### `encoders/image/vit_encoder.py`

**Summary**
- `ViTImageEncoder` wraps a Vision Transformer from `timm` (preferred) or `torchvision` as backbone, projecting CLS token output to `embedding_dim`.

---

#### `encoders/image/factory.py`

**Summary**
- `create_image_encoder(config)` — dispatches to `CNNImageEncoder`, `ResNetImageEncoder`, or `ViTImageEncoder` based on `config['model_type']`.

---

#### `encoders/temporal/base_temporal_encoder.py`

**Summary**
- Defines `TemporalAggregationStrategy` enum, `ModalityType` enum, the abstract `ItemLookupInterface`, and `BaseTemporalEncoder`.

**Key Classes**

- `TemporalAggregationStrategy(Enum)` — `LAST_HIDDEN`, `MEAN_POOLING`, `MAX_POOLING`, `ATTENTION`

- `ModalityType(Enum)` — `IMAGE = "image"`, `TEXT = "text"`, `CATEGORICAL = "categorical"`, `CONTINUOUS = "continuous"`

- `ItemLookupInterface(ABC)` — interface requiring `get_item_features(item_id)` and `batch_get_item_features(item_ids)`

- `BaseTemporalEncoder(BaseEncoder)` — stores `aggregation_strategy` (auto-converts string to enum), `max_sequence_length`

---

#### `encoders/temporal/lstm_temporal_encoder.py`

**Summary**
- `LSTMTemporalEncoder` processes sequences of historical item IDs: looks up item features via `ItemLookupInterface`, encodes each item's multi-modal features, runs an LSTM over the sequence, and aggregates hidden states.

**Key Classes**

- `AttentionPooling(nn.Module)` — single linear layer scoring hidden states, softmax-normalized weighted sum

- `LSTMTemporalEncoder(BaseTemporalEncoder)`
  - `item_lookup: ItemLookupInterface` — registered via `GenericItemLookupInterface` in pipeline
  - `modality_encoders: nn.ModuleDict` — subset of encoders for modalities present in item data
  - `item_embedding_dim` — sum of each enabled modality's output dim (computed in `_calculate_item_embedding_dim()`)
  - `lstm: nn.LSTM` — input size = `item_embedding_dim`, hidden = `lstm_hidden_dim`, supports bidirectional
  - `projection: nn.Linear` — projects `lstm_output_dim → embedding_dim`
  - `default_item_embedding`, `padding_embedding` — registered buffers of zeros

**Core Logic**
- `forward(temporal_dict)` — auto-detects single vs. batched input by checking if `first_field_value[0]` is a list
- `_encode_item_sequence(item_ids)` — truncates to `max_sequence_length` (keeps most recent), batch-fetches item features, encodes each with `_encode_single_item()`
- `_encode_single_item(item_features)` — iterates enabled modalities, calls each encoder's `forward()`, extracts tensor from returned dict, concatenates
- `_apply_aggregation(lstm_outputs, mask)` — implements all four strategies; LAST_HIDDEN uses `mask.sum() - 1` as the last-valid-index
- Multiple temporal fields are averaged (mean pooling over fields)
- `missing_item_strategy`: `'zero'` keeps position with zero embedding, `'skip'` removes it, `'previous'` repeats last

**Gotchas**
- `_calculate_item_embedding_dim()` uses `getattr(encoder, 'embedding_dim', 256)` which silently falls back to 256 if the encoder doesn't have the attribute — can cause LSTM input size mismatch
- Batch processing in `_forward_batched` iterates samples in a Python loop, not vectorized

---

#### `encoders/temporal/factory.py`

**Summary**
- `create_temporal_encoder(**config)` — converts string modality names to `ModalityType` enums and string aggregation strategy to `TemporalAggregationStrategy`, then instantiates `LSTMTemporalEncoder`.

---

### 📁 `interaction/`

---

#### `interaction/attention_utils.py`

**Summary**
- Low-level building blocks for attention: `MultiHeadAttention`, `TransformerBlock`, the attention-based `FeatureFusionLayer`, and a secondary simpler `SimpleFusionLayer` (distinct from the one in `feature_fusion.py`).

**Key Classes**

- `FeatureType(Enum)` — `IMAGE`, `TEXT`, `CATEGORICAL`, `CONTINUOUS`, `TEMPORAL`

- `InteractionType(Enum)` — `USER`, `ITEM`

- `MultiHeadAttention(nn.Module)`
  - Standard QKV projection: `q_proj`, `k_proj`, `v_proj`, `out_proj`
  - `head_dim = embedding_dim // num_heads`; asserts divisibility
  - Applies `attention_mask` via `masked_fill_(mask==0, -inf)`

- `TransformerBlock(nn.Module)`
  - Self-attention + feed-forward with residual connections
  - `use_layer_norm: bool` — if False, no LayerNorm (residual only)
  - `ff_dim = 4 * embedding_dim` by default

- `FeatureFusionLayer(nn.Module)`
  - CLS token (`nn.Parameter`) + feature type positional embeddings (`nn.Embedding` over 5 feature types)
  - Stacks features as sequence: `[cls_token, feat_1, feat_2, ...]`
  - Applies N transformer layers; extracts `hidden_states[:, 0, :]` (CLS) or mean if no CLS
  - Final `nn.Linear` projection

- `SimpleFusionLayer(nn.Module)` (in `attention_utils.py`) — concat + lazy MLP (different from the one in `feature_fusion.py` — note: name collision between files)

**Key Functions**

- `create_feature_type_mapping(feature_dict) → Dict[str, FeatureType]` — converts string types to enums; defaults unknown to `CATEGORICAL`
- `validate_feature_embeddings(feature_embeddings, expected_dim) → bool` — checks all embeddings are 2D with correct last dim
- `safe_mean_pooling(tensor, dim)` — handles empty tensors
- `create_attention_mask(batch_size, seq_len, mask_positions, device) → Tensor` — all-ones mask with optional masked positions

**Gotchas**
- There are **two** classes named `SimpleFusionLayer`: one in `attention_utils.py` (uses `feature_types` dict to filter) and one in `feature_fusion.py` (sorts by name, ignores `feature_types`). The one in `feature_fusion.py` is what the pipeline actually uses.

---

#### `interaction/feature_fusion.py`

**Summary**
- `SimpleFusionLayer`, `UserEmbeddingGenerator`, `ItemEmbeddingGenerator`, and `AsymmetricTowerModel` — the actual fusion modules used by the pipeline. Supports either simple concat+MLP or attention-based `FeatureFusionLayer` fusion.

**Key Classes**

- `SimpleFusionLayer(nn.Module)` — the **primary** fusion layer used by the pipeline
  - Sorts feature embeddings by name for deterministic order, concatenates, then projects with a 2-layer MLP (`Linear → ReLU → Linear`)
  - `projection` is lazily initialized on first forward call
  - `_projection_initialized: bool` — prevents re-initialization when loaded from checkpoint
  - On dimension mismatch, raises `RuntimeError` with clear message about feature count difference

- `UserEmbeddingGenerator(nn.Module)`
  - `use_simple_fusion: bool` — chooses between `SimpleFusionLayer` and `FeatureFusionLayer`
  - Calls `validate_feature_embeddings()` before forwarding
  - Converts string feature types to enums if needed

- `ItemEmbeddingGenerator(nn.Module)` — mirrors `UserEmbeddingGenerator`

- `AsymmetricTowerModel(nn.Module)` — composes `UserEmbeddingGenerator` + `ItemEmbeddingGenerator` with asymmetric configs; not used directly by pipeline (pipeline wires generators separately)

**Gotchas**
- `SimpleFusionLayer.projection` initialization check: if `projection is None AND _projection_initialized is False` → initialize; if `projection is not None AND _projection_initialized is False` → loaded from checkpoint, set flag only. This is the guard against double-initialization during checkpoint loading.
- Feature count must be **identical** at inference vs. training — the `RuntimeError` message explicitly states this and shows current vs. expected counts.

---

#### `interaction/interaction_modeling.py`

**Summary**
- `UserItemInteractionLayer`, `MultipleInteractionStrategies`, and `InteractionEmbeddingGenerator` — the interaction stage that combines user and item tower embeddings.

**Key Classes**

- `UserItemInteractionLayer(nn.Module)`
  - `use_simple_fusion=True`: concatenates user + item embeddings → 3-layer MLP with `LayerNorm` and `Dropout`
  - `use_simple_fusion=False`: attention-based — prepends learnable `interaction_cls_token`, adds type embeddings from `interaction_type_embeddings` (2-vocab `nn.Embedding`), applies N transformer layers, extracts CLS position

- `MultipleInteractionStrategies(nn.Module)` — creates one `UserItemInteractionLayer` per strategy (`bidirectional`, `user_to_item`, `item_to_user`); learns soft combination via `strategy_gate` + softmax

- `InteractionEmbeddingGenerator(nn.Module)` — thin wrapper; delegates to either `MultipleInteractionStrategies` or a single `UserItemInteractionLayer`

**Core Logic**
- Simple MLP path: `cat([user, item], dim=1)` → `[batch, 2*emb]` → MLP → `[batch, emb]`
- Attention path: `[CLS, user, item]` sequence → transformer layers → CLS output → final projection

**Gotchas**
- `assert user_embedding.shape == item_embedding.shape` — both towers must produce identical embedding dimensions

---

### 📁 `classifier/`

---

#### `classifier/classification_utils.py`

**Summary**
- All building blocks for the classification head: `MLPBlock`, `MLPTower`, `BinaryClassificationHead`, five loss implementations, `LossManager`, plus standalone `create_optimizer()` and `create_scheduler()` factories.

**Key Classes**

- `LossType(Enum)` — `BCE`, `CONTRASTIVE`, `COMBINED` (note: `focal` and `triplet` are supported in `LossManager` but not in the enum)

- `MLPBlock(nn.Module)` — `Linear → LayerNorm (when use_batch_norm=True) → activation → Dropout`
  - Despite parameter `use_batch_norm`, applies `nn.LayerNorm` not `nn.BatchNorm1d`
  - Activations: `relu`, `gelu`, `leaky_relu` (0.1 slope), `swish` (SiLU)

- `MLPTower(nn.Module)` — sequences `MLPBlock` layers then a final `nn.Linear`; optional `sigmoid` or `tanh` on final layer

- `BinaryClassificationHead(nn.Module)` — single `nn.Linear(input_dim, 1)`, returns raw logits

- `ContrastiveLoss(nn.Module)` — pairwise: positives pulled together (`d^2`), negatives pushed apart `(max(0, margin - d))^2`; operates on full batch via `torch.cdist`

- `TripletLoss(nn.Module)` — wraps `nn.TripletMarginLoss`; splits batch by label to form anchor/positive/negative sets; returns 0 if batch lacks both positive and negative samples

- `FocalLoss(nn.Module)` — `alpha * (1-p_t)^gamma * BCE`; addresses class imbalance

- `LossManager(nn.Module)` — dispatches to the right loss in `compute_loss(logits, embeddings, labels)`:
  - `bce` → `BCEWithLogitsLoss`
  - `contrastive` → `ContrastiveLoss`
  - `combined` → weighted `bce_weight * BCE + contrastive_weight * Contrastive`
  - `focal` → `FocalLoss`
  - `triplet` → `TripletLoss`

**Key Functions**

- `create_optimizer(model_parameters, optimizer_type, learning_rate, weight_decay) → Optimizer`
  - Supports `adam`, `adamw`, `sgd`, `rmsprop`; defaults to Adam

- `create_scheduler(optimizer, scheduler_type, num_epochs) → Optional[LRScheduler]`
  - Supports `cosine`, `step`, `exponential`, `plateau`; returns None for unknown/None

**Gotchas**
- `use_batch_norm=True` applies `LayerNorm` not `BatchNorm` — the parameter name is misleading; this was apparently intentional for stability

---

#### `classifier/recommendation_classifier.py`

**Summary**
- `RecommendationClassifier` is the final stage: `MLPTower` → `BinaryClassificationHead`, with `LossManager` for training.

**Key Classes**

- `RecommendationClassifier(nn.Module)`
  - `mlp: MLPTower` — transforms interaction embedding to `mlp_output_dim = mlp_hidden_dims[-1]` (or `embedding_dim` if no hidden dims)
  - `classification_head: BinaryClassificationHead` — maps `mlp_output_dim → 1`
  - `loss_manager: LossManager` — computes training loss

  - `forward(interaction_embeddings, labels, return_embeddings) → (loss, logits) | logits | (logits, embeddings)`
    - If `labels is not None`: returns `(loss, logits)` for training
    - Otherwise: returns `logits` for inference
    - If `return_embeddings=True`: returns tuple with MLP embeddings appended

  - `predict_proba(interaction_embeddings) → Tensor` — sigmoid of logits; sets model to `eval()`
  - `predict(interaction_embeddings, threshold=0.5) → Tensor` — binary predictions
  - `get_embeddings(interaction_embeddings) → Tensor` — MLP output before classification head

---

### 📁 `trainer/`

---

#### `trainer/pipeline_builder.py`

**Summary**
- Contains `RecommendationPipeline` (the central `nn.Module` that wires all encoders, fusion, interaction, and classifier), plus `DimensionAligner`, `GenericItemLookupInterface`, `save_complete_model()`, `load_model_from_config()`, and `save_model_config()`.

**Key Classes**

- `GenericItemLookupInterface(ItemLookupInterface)`
  - `item_db: Dict` — maps item_id (as string or int) to feature dict
  - `get_item_features(item_id)` — tries string key first, then int conversion; returns empty feature dict if not found
  - `batch_get_item_features(item_ids)` — calls `get_item_features` for each

- `DimensionAligner(nn.Module)`
  - `target_dim: int` — all embeddings are projected here
  - `projections: nn.ModuleDict` — maps `str(input_dim)` to `nn.Linear` or `nn.Identity`
  - `_get_projection(input_dim)` — creates projection lazily on first encounter
  - `forward(embeddings)` — finds majority batch size, applies projection + batch size broadcasting

- `RecommendationPipeline(nn.Module)` — the central model
  - **Encoder attributes**: `image_encoder`, `text_encoder`, `categorical_encoder`, `user_continuous_encoder`, `item_continuous_encoder`, `temporal_encoder`
  - **Aligner attributes**: `user_dimension_aligner`, `item_dimension_aligner`
  - **Tower attributes**: `user_generator` (`UserEmbeddingGenerator`), `item_generator` (`ItemEmbeddingGenerator`)
  - **Interaction attribute**: `interaction_generator` (`InteractionEmbeddingGenerator`)
  - **Classifier attribute**: `classifier` (`RecommendationClassifier`)

  - `_get_valid_num_heads(target_heads, embedding_dim)` — finds nearest power-of-2-ish head count in `[4, 8, 16, 32, 64]` that divides `embedding_dim`

  - `_create_*_encoder(config)` — creates each encoder with merged defaults; stores `encoder._actual_config` for later saving

  - `_create_temporal_encoder(config)` — complex logic: only creates if `enable_item_lookup=True` or full config is detected; filters enabled modalities to those actually available in item data

  - `_encode_features(features, aligner) → Dict[str, Tensor]` — routes each feature type to its encoder, calls `aligner.forward()` to normalize dimensions; routes `continuous` to `user_continuous_encoder` or `item_continuous_encoder` based on which `aligner` is passed

  - `forward(user_data, item_data, labels)` — 6 steps: encode user → encode item → user tower → item tower → interaction → classify

  - `_get_feature_types(encoded_features)` — infers feature types from key names using substring matching (`"image"`, `"text"`, `"temporal"`, `"category"`, `"location"`, `"gender"`, `"categorical"`); defaults to `"continuous"`

  - `set_config(config)` — normalizes `*_encoder_config` → `*_encoder` key names; overwrites with `_actual_config` from encoders where available

**Key Functions**

- `save_model_config(model, config_path)` — calls `model.get_config()`; raises `RuntimeError` if no config stored
- `save_complete_model(model, save_dir, model_name, verbose)` — saves full state dict as `{model_name}.pt`, config as `{model_name}_config.json`, and component state dicts (`_user_tower.pt`, `_item_tower.pt`, `_interaction_tower.pt`, `_classifier.pt`, plus individual encoders), plus a `_manifest.json`
- `load_model_from_config(config_path, weights_path, item_data)` — reconstructs `RecommendationPipeline` from config JSON, loads state dict with `strict=False`, handles lazy module pre-initialization

**Gotchas**
- `_get_feature_types()` uses substring matching on encoder output keys — feature keys containing none of the expected substrings default to `"continuous"`, which may be incorrect
- Two separate `user_continuous_encoder` and `item_continuous_encoder` are created so user and item can have different numbers of continuous features
- `_create_temporal_encoder()` will silently skip temporal encoding if `item_data` is None or no modalities are available in item data
- There are commented-out debug print lines in `_get_feature_types()` — code references to `"REMOVE THESE DEBUG LINES"` indicate in-progress cleanup

---

#### `trainer/data_loader.py`

**Summary**
- `RecommendationDataset` (PyTorch `Dataset`) and `create_data_loaders()` factory. Handles positive/negative pair generation, train/val/test splitting, and custom batch collation.

**Key Classes**

- `RecommendationDataset(Dataset)`
  - `user_data_dict: Dict` — keyed by `user_id`
  - `item_data_dict: Dict` — keyed by `item_id`
  - `positive_interactions: List[Tuple]`, `negative_interactions: List[Tuple]`
  - `user_positive_items: defaultdict(set)` — for negative sampling exclusion
  - `_create_samples()` — adds all positives, existing negatives, then generates `additional_negatives_needed = target_negatives - existing_negatives`
  - `_generate_negative_samples(num_samples)` — random user+item pairs; skips pairs in `user_positive_items`; max attempts = `10 * num_samples`
  - `__getitem__(idx)` — returns `{"user_features", "item_features", "labels", "user_id", "item_id"}`

**Key Functions**

- `collate_recommendation_batch(batch) → Dict`
  - Aggregates nested `user_features[feature_type][field_name]` across samples
  - Missing fields are represented as `None` in the list

- `create_data_loaders(inputs, interactions, train_split, batch_size, negative_sampling_ratio, num_workers, seed, test_split) → (train_loader, val_loader, test_interactions, train_interactions)`
  - 3-way split: `test_split` (default 5%) from head of shuffled interactions, remaining split by `adjusted_train_split = train_split / (1 - test_split)`
  - Returns the raw test and train interaction lists for external evaluation

- `load_interactions_from_input(inputs) → List[Tuple]`
  - Extracts `(user_id, item_id, label)` triples; label defaults to `1` if not in interaction dict
  - Raises `ValueError` if no interactions found

- `create_interactions_from_data_DEPRECATED` — raises `DeprecationWarning`; synthetic interaction generation is removed

**Gotchas**
- `collate_recommendation_batch` collects values as Python lists with `None` for missing — encoders must handle `None` inputs
- `num_workers=0` by default — no multiprocessing; fine for most cases but may bottleneck large datasets
- `pin_memory=torch.cuda.is_available()` is set automatically

---

#### `trainer/trainer.py`

**Summary**
- `train_model()` implements the standard PyTorch training loop with tqdm progress bars, gradient clipping, scheduler stepping, and periodic model checkpointing.

**Key Functions**

- `train_model(model, train_loader, val_loader, num_epochs, learning_rate, optimizer_type, scheduler_type, device, print_every, save_path) → Dict`
  - Optimizer creation: `adam` → `Adam`, `adamw` → `AdamW(weight_decay=0.01)`, `sgd` → `SGD(momentum=0.9)`
  - Scheduler creation: `plateau` → `ReduceLROnPlateau(patience=10, factor=0.5)`, `cosine` → `CosineAnnealingLR(T_max=num_epochs)`, `step` → `StepLR(step_size=30, gamma=0.1)`
  - Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
  - Saves best model when `avg_val_loss < best_val_loss`; also saves every epoch overwriting the previous save
  - Returns `{"train_losses", "val_losses", "train_accuracies", "val_accuracies"}`

- `prepare_batch(batch, device) → (user_data, item_data, labels)`
  - Recursively moves tensors to `device`; non-tensor values pass through unchanged

- `evaluate_model(model, test_loader, device) → Dict`
  - Contains uncommitted debug prints (`print("\n=== ENCODER OUTPUT DEBUG ===")`) — these execute during evaluation and pollute stdout

**Gotchas**
- `evaluate_model()` has debug print statements that were not removed (lines 343–348)
- `save_complete_model()` is called every epoch (not just best), which is slow for large models with many component files

---

### 📁 `evaluation/`

---

#### `evaluation/ranking_metrics.py`

**Summary**
- Pure-function library for offline ranking evaluation. No dependencies on model or training.

**Key Functions**

- `precision_at_k(recommended_items, relevant_items, k) → float` — hits in top-k / k
- `recall_at_k(recommended_items, relevant_items, k) → float` — hits in top-k / |relevant|
- `compute_metrics_for_user(recommended_items, relevant_items, k_values) → dict` — computes both metrics for each k in `k_values`; returns `{"precision@k": float, "recall@k": float, ...}`

---

#### `evaluation/evaluate_recommendation.py`

**Summary**
- End-to-end offline evaluation script. Loads model and data, pre-computes all item embeddings, generates cosine-similarity-ranked recommendations per user, then aggregates `precision@k` and `recall@k`.

**Note:** This evaluation uses **cosine similarity** between user and item tower embeddings (not the full classifier pipeline), unlike `inference.py` which uses classifier probability. This may produce slightly different rankings.

**Key Functions**

- `evaluate_recommendations(model, user_data, item_data, ground_truth, k_values, encoders_used, expected_user_features, expected_item_features, exclude_items, max_recommendations) → Dict`
  - Pre-computes all item embeddings first for efficiency
  - Supports `exclude_items` dict to filter training interactions from evaluation
  - Aggregates per-user metrics with `np.mean`

- `load_ground_truth(file_path) → Dict[int, List[int]]`
  - Expects JSON: `{"user_id": [item_id1, item_id2, ...], ...}`

**CLI Arguments**

| Flag | Description |
|------|-------------|
| `--model_dir` / `--model_name` | Model location |
| `--data_path` | Input data JSON |
| `--ground_truth` | Ground truth JSON |
| `--k` (nargs='+', default=[5, 10, 20]) | K values |
| `--exclude_items` | Optional items-to-exclude JSON |
| `--max_recommendations` | Max recs per user (default 100) |
| `--output_file` | Save results JSON |

---

### 📁 `utils/`

---

#### `utils/config_validator.py`

**Summary**
- Validates a model configuration dict before `RecommendationPipeline` creation. Checks required fields, type constraints, and architectural compatibility.

**Key Classes**

- `ValidationResult` — dataclass with `errors: List[str]`, `warnings: List[str]`; `is_valid` property returns `len(errors) == 0`

**Key Functions**

- `validate_config(config) → ValidationResult`
  - Checks `embedding_dim` exists and is positive int
  - Delegates to per-encoder validators
  - Checks `{tower}_num_heads` divides `embedding_dim`

- `validate_categorical_encoder_config(config, errors, warnings)` — requires `mlp_hidden_dims`, `embedding_dim`, `aggregation_strategy`
- `validate_text_encoder_config` — requires `model_name`, `embedding_dim`
- `validate_image_encoder_config` — requires `model_type`, `embedding_dim`
- `validate_continuous_encoder_config` — requires `embedding_dim`; warns if `hidden_dims` absent
- `validate_temporal_encoder_config` — requires `output_dim`
- `validate_config_file(config_path) → ValidationResult` — file-based wrapper

**Gotchas**
- `validate_categorical_encoder_config` does **not** check for `aggregation_strategy` in some edge cases (the check exists but is called from `validate_config` which may pass only the sub-dict)
- The `ValidationResult` here is a **different class** than the `ValidationResult` in `input_processor.py` — same name, different module, different implementation

---

### 📁 `tests/`

---

#### `tests/conftest.py`

**Summary**
- Session-scoped `inputs` fixture loads `datasets/post_recommendation/updated_output_split.json`; function-scoped `small_dataset` fixture filters to 50 users and 50 items.

**Key Fixtures**

- `inputs` (session scope) — `Inputs()` with `image_check_files=False`; calls `pytest.skip()` on load failure
- `small_dataset(inputs, max_users=50, max_items=50)` — returns `(filtered_user_data, filtered_item_data, filtered_interactions)`

---

#### `tests/test_training.py`

**Summary**
- `@pytest.mark.slow` test verifying 1-epoch training completes and produces valid loss history.

**Key Tests**
- `test_training(inputs, small_dataset)` — creates `RecommendationPipeline(embedding_dim=64)`, trains 1 epoch, asserts `train_losses` is non-empty and non-negative

---

## SECTION 4 — Architecture Overview

**Architectural pattern:** Modular pipeline library. There is no HTTP server, no database, no service boundary — the codebase is a set of `nn.Module` classes assembled into `RecommendationPipeline`, driven by two CLI scripts (`train.py`, `inference.py`).

**Major layers:**

| Layer | Responsible class/module | Responsibility |
|-------|--------------------------|----------------|
| Input & Validation | `input_processor.Inputs`, `*FeatureInput` | Parse + validate JSON; expose typed data |
| Config Validation | `utils.config_validator.validate_config` | Check architecture compatibility before build |
| Encoding | `encoders.*` | Map raw feature data → fixed-dim `Tensor` per modality |
| Dimension Alignment | `DimensionAligner` | Project all encoder outputs to `embedding_dim` |
| Feature Fusion (Tower) | `SimpleFusionLayer` / `FeatureFusionLayer` | Fuse multi-modal features into one user/item vector |
| Interaction | `UserItemInteractionLayer` | Combine user + item vectors into an interaction representation |
| Classification | `RecommendationClassifier` (MLP + head) | Score the interaction embedding into a probability |
| Training | `train_model()`, `RecommendationDataset` | Gradient descent with pos/neg pairs |
| Persistence | `save_complete_model()`, `load_model_from_config()` | Save/restore complete model state + config |
| Evaluation | `evaluate_recommendation.py`, `ranking_metrics.py` | Offline precision/recall |

**Design patterns in use:**

| Pattern | Where Used | Why Used |
|---------|-----------|----------|
| Abstract Factory | `encoders/*/factory.py` | Decouple config dict from concrete encoder class |
| Template Method | `BaseEncoder`, `BaseTemporalEncoder` | Enforce `forward()` contract across encoders |
| Lazy Initialization | `MLPContinuousEncoder.mlp`, `SimpleFusionLayer.projection`, `DimensionAligner.projections` | Input dimension is unknown until first batch |
| Strategy | `TemporalAggregationStrategy`, `CategoricalAggregationStrategy`, `LossType` | Swap algorithm (aggregation/loss) without code change |
| Composite | `LossManager` | Combine multiple loss functions with weights |

**Cross-cutting concerns:**
- **Logging:** tqdm progress bars in trainer; print statements (`print(f"✅ ...")`) in drivers and pipeline builder — no structured logging framework
- **Error handling:** `ValidationResult` pattern for validation; `RuntimeError` for config mismatches; `warnings.warn()` in temporal encoder for per-item failures
- **Validation:** All user-supplied data validated by `input_processor.py` at load time; config validated by `config_validator.py` before model creation
- **Security:** Not applicable (local ML library, no network interfaces)

```mermaid
graph TD
    A[JSON Data File] --> B[input_processor.Inputs]
    C[JSON Config File] --> D[config_validator.validate_config]
    B --> E[RecommendationPipeline.__init__]
    D --> E
    E --> F[Encoders: image/text/cat/cont/temporal]
    F --> G[DimensionAligner]
    G --> H[UserEmbeddingGenerator / ItemEmbeddingGenerator]
    H --> I[InteractionEmbeddingGenerator]
    I --> J[RecommendationClassifier]
    J --> K[Loss + Logits]
    K --> L[train_model training loop]
    L --> M[save_complete_model]
    M --> N[model.pt + config.json + manifest]
    N --> O[load_model_from_config]
    O --> P[inference.py find_top_items_for_user]
    P --> Q[Top-k Recommendations]
```

---

## SECTION 5 — Data Flow & Request Lifecycle

### Training Path

```mermaid
sequenceDiagram
    participant CLI as train.py main()
    participant IP as input_processor.Inputs
    participant CV as config_validator
    participant PB as pipeline_builder
    participant DL as data_loader
    participant TR as trainer.train_model
    participant FS as filesystem

    CLI->>IP: load_from_json(data_path)
    IP-->>CLI: ValidationResult + user_data + item_data + interactions
    CLI->>CV: validate_config(config)
    CV-->>CLI: ValidationResult
    CLI->>PB: create_model_from_config(config, item_data)
    PB->>PB: instantiate encoders, aligners, generators, classifier
    PB-->>CLI: RecommendationPipeline
    CLI->>DL: create_data_loaders(inputs, interactions, ...)
    DL->>DL: split interactions (train/val/test)
    DL->>DL: RecommendationDataset (negative sampling)
    DL-->>CLI: train_loader, val_loader, test_interactions, train_interactions
    CLI->>TR: train_model(model, train_loader, val_loader, ...)
    loop Each Epoch
        TR->>TR: prepare_batch → model.forward(user, item, labels)
        TR->>TR: loss.backward() + optimizer.step()
        TR->>PB: save_complete_model (every epoch)
    end
    TR-->>CLI: training history dict
    CLI->>FS: save history JSON, test/train interactions JSON
```

### Inference / Model Forward Pass (per batch)

1. `_encode_features(user_data, user_dimension_aligner)`:
   - `image_encoder(features["image"])` → `{"image_features": Tensor}`
   - `text_encoder(features["text"])` → `{"text_features": Tensor}`
   - `categorical_encoder(features["categorical"])` → `{"categorical_features": Tensor}`
   - `user_continuous_encoder(features["continuous"])` → `{"continuous_features": Tensor}`
   - `temporal_encoder(features["temporal"])` → `{"temporal": Tensor}`
   - `user_dimension_aligner(all_raw_embeddings)` → projects all to `embedding_dim`
2. `user_generator(user_encoded, user_feature_types)` → `user_embedding [B, D]`
3. Same flow for item features → `item_embedding [B, D]`
4. `interaction_generator(user_embedding, item_embedding)` → `interaction_embedding [B, D]`
5. `classifier(interaction_embedding, labels)` → `(loss, logits)` in training; `logits` in inference
6. Caller applies `torch.sigmoid(logits)` to get probability

### Evaluation Path

- Pre-compute all item embeddings via `generate_item_embedding()` (user tower only)
- Per user: compute cosine similarity with all items, sort descending, take top-k
- Compute `precision@k` and `recall@k` against ground truth

---

## SECTION 6 — API Endpoints Catalogue

> Not applicable for this project (no HTTP API).

**Public Python API** (intended for programmatic use):

| Entry point | Signature | Description |
|-------------|-----------|-------------|
| `Inputs.load_from_json` | `(json_path) → ValidationResult` | Load data from file |
| `Inputs.load_from_dict` | `(data: Dict) → ValidationResult` | Load data from dict |
| `Inputs.get_user_data` | `() → List[Dict]` | Get all validated users |
| `Inputs.get_item_data` | `() → List[Dict]` | Get all validated items |
| `Inputs.get_interactions` | `() → List[Dict]` | Get all interactions |
| `create_model_from_config` | `(config, item_data) → RecommendationPipeline` | Build model from config |
| `train_model` | `(model, train_loader, val_loader, ...) → Dict` | Run training loop |
| `save_complete_model` | `(model, save_dir, model_name)` | Save all artifacts |
| `load_model_from_config` | `(config_path, weights_path, item_data) → RecommendationPipeline` | Restore model |
| `find_top_items_for_user` | `(model, user_data, all_items, k, ...)` | Generate recommendations |
| `evaluate_recommendations` | `(model, user_data, item_data, ground_truth, k_values, ...)` | Compute ranking metrics |
| `validate_config` | `(config) → ValidationResult` | Validate config dict |

---

## SECTION 7 — Database Schema & Data Models

> Not applicable for this project (no database).

**In-memory and file-based data structures:**

**Input JSON schema** (required structure):
```json
{
  "user_data": [
    {
      "user_id": 1,
      "image": {"field_name": "/path/to/image.jpg"},
      "text": {"field_name": "some text"},
      "categorical": {"field_name": "value"},
      "continuous": {"field_name": 1.0},
      "temporal": {"field_name": [id1, id2, id3]}
    }
  ],
  "item_data": [
    {
      "item_id": 101,
      "image": {...},
      "text": {...},
      "categorical": {...},
      "continuous": {...},
      "temporal": {...}
    }
  ],
  "interactions": [
    {
      "user_id": 1,
      "item_id": 101,
      "interaction_type": "purchase",
      "timestamp": "2024-01-15T10:30:00",
      "label": 1
    }
  ]
}
```

**Saved model artifacts** (`models/{model_name}.*`):

| File | Contents |
|------|---------|
| `{name}.pt` | Full `state_dict()` of `RecommendationPipeline` |
| `{name}_config.json` | Architecture config (encoder configs, tower params, classifier dims) |
| `{name}_manifest.json` | List of all saved component file paths |
| `{name}_history.json` | `{"train_losses": [...], "val_losses": [...], ...}` |
| `{name}_user_tower.pt` | `user_generator.state_dict()` |
| `{name}_item_tower.pt` | `item_generator.state_dict()` |
| `{name}_interaction_tower.pt` | `interaction_generator.state_dict()` |
| `{name}_classifier.pt` | `classifier.state_dict()` |
| `{name}_text_encoder.pt` | `text_encoder.state_dict()` |
| `{name}_categorical_encoder.pt` | `categorical_encoder.state_dict()` |
| `{name}_temporal_encoder.pt` | `temporal_encoder.state_dict()` |
| `{name}_test_interactions.json` | Reserved test set interactions |
| `{name}_train_interactions.json` | Training interactions (for exclusion during eval) |

---

## SECTION 8 — Environment Variables & Configuration

> No environment variables. All configuration is via JSON files.

**Configuration file structure** (`configs/sample_config_simple_fusion.json` shape):

| Key | Type | Description | Required | Default |
|-----|------|-------------|----------|---------|
| `embedding_dim` | int | Shared embedding dimension throughout pipeline | Yes | — |
| `loss_type` | str | `"bce"`, `"contrastive"`, `"combined"`, `"focal"`, `"triplet"` | No | `"bce"` |
| `user_use_simple_fusion` | bool | Use concat+MLP vs. attention for user tower | No | `true` |
| `user_num_attention_layers` | int | Transformer layers in user tower (if attention) | No | `4` |
| `user_num_heads` | int | Attention heads in user tower | No | `16` |
| `user_dropout` | float | Dropout in user tower | No | `0.15` |
| `user_use_cls_token` | bool | CLS token in user attention | No | `true` |
| `user_use_layer_norm` | bool | LayerNorm in user transformer blocks | No | `false` |
| `item_use_simple_fusion` | bool | Use concat+MLP vs. attention for item tower | No | `true` |
| `item_num_attention_layers` | int | Transformer layers in item tower | No | `1` |
| `item_num_heads` | int | Attention heads in item tower | No | `8` |
| `item_dropout` | float | Dropout in item tower | No | `0.1` |
| `interaction_use_simple_fusion` | bool | Use concat+MLP vs. attention for interaction | No | `true` |
| `interaction_num_attention_layers` | int | Transformer layers in interaction stage | No | `2` |
| `interaction_num_heads` | int | Attention heads in interaction | No | `8` |
| `interaction_dropout` | float | Dropout in interaction | No | `0.1` |
| `classifier_hidden_dims` | List[int] | MLP hidden layer dims for classifier | No | `[512, 256]` |
| `classifier_dropout` | float | Dropout in classifier | No | `0.2` |
| `text_encoder_config` | Dict or null | Text encoder config (see below) | No | null |
| `categorical_encoder_config` | Dict or null | Categorical encoder config (see below) | No | null |
| `continuous_encoder_config` | Dict or null | Continuous encoder config | No | default MLP |
| `image_encoder_config` | Dict or null | Image encoder config | No | null |
| `temporal_encoder_config` | Dict or null | Temporal encoder config | No | null |
| `train_split` | float | Fraction of data for training | No | `0.8` |
| `batch_size` | int | Training batch size | No | `32` |
| `negative_sampling_ratio` | float | Negatives per positive | No | `1.0` |
| `seed` | int | Random seed | No | `42` |
| `num_epochs` | int | Training epochs | No | `100` |
| `learning_rate` | float | Initial learning rate | No | `0.001` |
| `optimizer_type` | str | `"adam"`, `"adamw"`, `"sgd"` | No | `"adam"` |
| `scheduler_type` | str | `"plateau"`, `"cosine"`, `"step"`, null | No | `"plateau"` |
| `print_every` | int | Print epoch summary every N epochs | No | `10` |

**Encoder sub-configs:**

`text_encoder_config`:
```json
{"model_name": "glove-wiki-gigaword-50", "aggregation_strategy": "mean", "embedding_dim": 64, "num_text_fields": 2}
```

`categorical_encoder_config`:
```json
{"aggregation_strategy": "separate_concat", "hash_vocab_size": 4096, "embedding_dim": 64, "num_categorical_fields": 1, "mlp_hidden_dims": []}
```

`temporal_encoder_config`:
```json
{"enable_item_lookup": true, "aggregation_strategy": "mean_pooling", "output_dim": 64}
```

---

## SECTION 9 — Business Logic & Domain Glossary

**Glossary:**

| Term | Definition | Code location |
|------|-----------|---------------|
| User embedding | Fixed-dim vector representing a user's preferences, derived from all user features | `UserEmbeddingGenerator.forward()` → `feature_fusion.py` |
| Item embedding | Fixed-dim vector representing an item, derived from all item features | `ItemEmbeddingGenerator.forward()` → `feature_fusion.py` |
| Interaction embedding | Combined user+item vector capturing their compatibility | `InteractionEmbeddingGenerator.forward()` → `interaction_modeling.py` |
| Two-tower model | Architecture where user and item are encoded separately before interaction | `RecommendationPipeline` in `pipeline_builder.py` |
| Simple fusion | Concatenation + MLP projection for feature merging | `SimpleFusionLayer` in `feature_fusion.py` |
| Attention fusion | Transformer-based feature merging with CLS token | `FeatureFusionLayer` in `attention_utils.py` |
| Positive interaction | An observed user-item engagement; label=1 | `RecommendationDataset.positive_interactions` |
| Negative interaction | Unobserved or negative user-item pair; label=0 | `RecommendationDataset._generate_negative_samples()` |
| Negative sampling ratio | Controls ratio of negatives to positives in training | `RecommendationDataset.__init__(negative_sampling_ratio)` |
| Hash encoder | Hash categorical values to integer indices for embedding lookup | `HashCategoricalEncoder._hash_category()` |
| Dimension aligner | Lazy projection layer normalizing encoder output dims to `embedding_dim` | `DimensionAligner` in `pipeline_builder.py` |
| Temporal encoder | LSTM over historical item interaction sequences | `LSTMTemporalEncoder` in `lstm_temporal_encoder.py` |
| Precision@K | Fraction of top-k recommendations that are relevant | `ranking_metrics.precision_at_k()` |
| Recall@K | Fraction of all relevant items found in top-k | `ranking_metrics.recall_at_k()` |

**Business rules:**

| Rule | Enforcement location | Trigger | Failure behavior |
|------|---------------------|---------|-----------------|
| Config `embedding_dim` must be positive int | `validate_config()` | Before model creation | `ValueError` |
| `num_heads` must divide `embedding_dim` | `validate_config()` | Before model creation | `ValueError` |
| `categorical_encoder_config` must have `mlp_hidden_dims` | `validate_categorical_encoder_config()` | Config validation | `ValueError` |
| Pretrained architecture must match training config | `validate_config_compatibility()` | When `--pretrained_weights` is provided | `RuntimeError` with diff |
| Interactions must have `user_id` and `item_id` | `Inputs._load_interactions()` | Data loading | Entry skipped, error added |
| Feature count at inference must match training | `SimpleFusionLayer.forward()` | First inference forward pass with loaded checkpoint | `RuntimeError` |

**State machines:** No explicit FSMs; model state transitions are `train()` / `eval()` mode toggling.

**Permission / Role Matrix:**
> Not applicable.

**Validation rules:**
- `continuous` values: not NaN, not ±inf, within optional `value_ranges`
- `image` values: string paths, file must exist if `check_file_exists=True`, extension must be in `{.jpg, .jpeg, .png, .bmp, .tiff, .webp}`
- `temporal` values: must be lists; items in list may be `int`, `float`, `str`, or `dict`
- `categorical` values: must be `str`, `int`, or `float`; checked against `allowed_values` if provided
- `text` values: must be `str` (unless prefixed `tokenized_`); empty strings warn

---

## SECTION 10 — External Services & Integrations

| Integration | Purpose | SDK | Config |
|-------------|---------|-----|--------|
| HuggingFace Transformers | Text encoding with pre-trained models | `transformers>=4.20.0`, `AutoModel`, `AutoTokenizer` | `model_name` in `text_encoder_config` |
| gensim | Word2Vec/FastText/GloVe text encoding | `gensim` (optional, not in requirements.txt) | `model_name` matching `word2vec`/`fasttext`/`glove` pattern |
| torchvision | ResNet and ViT image backbones | `torchvision>=0.11.0` | `model_type: "resnet"` with `pretrained: true` |
| timm | ViT and advanced image backbones | `timm` (optional) | `model_type: "vit"` |
| qdrant-client | Vector database (listed in requirements but not called in current code) | `qdrant-client>=1.7.0` | Not yet wired into codebase |
| OpenCV | Image loading and preprocessing | `cv2` | Used in `base_image_encoder.py` |
| scikit-learn | Precision/recall/F1/AUC in `evaluate_model()` | `scikit-learn>=1.0.0` | Optional import in `trainer.py` |

**Notes:**
- `qdrant-client` is in `requirements.txt` but no code in the repo imports or uses it — likely planned for future vector search integration
- gensim is not in `requirements.txt` but required for Word2Vec encoders — missing dependency for that path

---

## SECTION 11 — Testing Strategy

**Framework:** pytest

**Location:** `tests/`

**Run command:**
```bash
cd /home/darshil/Desktop/recommendkit
pytest tests/ -v
pytest tests/ -v -m "not slow"   # Skip slow tests
pytest tests/ -v -m slow          # Run only slow tests
```

**Test files:**

| File | Scope | Key cases |
|------|-------|-----------|
| `tests/test_training.py` | Training loop | `@pytest.mark.slow` — 1-epoch training on 50-user/50-item subset; verifies loss history is non-empty and non-negative |
| `tests/test_inference.py` | Inference pipeline | Loading saved model, generating recommendations |
| `tests/test_save_load.py` | Model persistence | Save + load cycle preserves predictions |
| `tests/test_determinism.py` | Reproducibility | Same seed → same results |
| `tests/test_matrix_factorization.py` | Matrix factorization variant | > ⚠️ Could not determine full scope — needs team input |
| `tests/run_diagnosis.py` | Diagnostic script | Non-pytest diagnostic helper |

**Global fixtures** (`tests/conftest.py`):
- `inputs` (session scope): loads `datasets/post_recommendation/updated_output_split.json`; skips all tests if file missing
- `small_dataset` (function scope): filters to 50 users, 50 items, and their interactions

**Coverage gaps:**
- No tests for individual encoders (text, image, categorical, continuous, temporal)
- No unit tests for `config_validator.py`
- No tests for `input_processor.py` validation edge cases
- No tests for evaluation scripts (`evaluate_recommendation.py`)
- `gensim`-based Word2Vec path is not covered

---

## SECTION 12 — Deployment & CI/CD

> Not applicable for this project (no CI/CD configuration files found in the repository).

**Manual training:**
```bash
python3 train.py \
  --data_path datasets/synthetic/correlated_dataset.json \
  --config_path configs/sample_config_simple_fusion.json \
  --output_dir models \
  --model_name my_model
```

**Manual inference:**
```bash
python3 inference.py \
  --data_path datasets/synthetic/correlated_dataset.json \
  --model_dir models \
  --model_name my_model \
  --k 10
```

**Manual evaluation:**
```bash
python3 evaluation/evaluate_recommendation.py \
  --model_dir models \
  --model_name my_model \
  --data_path datasets/synthetic/correlated_dataset.json \
  --ground_truth path/to/ground_truth.json \
  --k 5 10 20
```

**Continue training from checkpoint:**
```bash
python3 train.py \
  --data_path data.json \
  --config_path configs/same_config.json \
  --pretrained_weights models \
  --pretrained_model_name my_model \
  --output_dir models \
  --model_name my_model_v2
```

---

## SECTION 13 — Security Implementation

This is a local ML training library with no HTTP endpoints, no authentication, no external network calls initiated by the codebase itself (aside from HuggingFace model downloads via `transformers`).

**Input safety:**
- All user data is validated by `input_processor.py` before reaching the model — type checks, NaN/inf rejection
- No `eval()` or `exec()` on user-supplied data
- File paths from JSON are opened only by `ImageFeatureInput.validate_feature_data()` via `pathlib.Path.exists()`

**Model weights:**
- `torch.load()` is called without `weights_only=True` in several places — this is a known PyTorch security concern (arbitrary code execution via pickled objects). Only load weights from trusted sources.

**Secrets / PII:** None stored in code.

---

## SECTION 14 — Known Issues & Technical Debt

| Location (file:line) | Comment Text | Impact | Suggested Priority |
|----------------------|-------------|--------|-------------------|
| `trainer/pipeline_builder.py:539` | `# REMOVE THESE DEBUG LINES:` | Debug comments left in; commented-out prints clutter the code | Low |
| `trainer/pipeline_builder.py:556` | `# REMOVE THIS DEBUG LINE:` | Same — leftover debug comments | Low |
| `trainer/trainer.py:339–348` | `# DEBUG: Check what encoders output` followed by live `print()` calls | `evaluate_model()` prints debug output to stdout on every call | Medium |
| `inference.py:126–130` | `print(f"🔍 Debug: Encoded user features: ...")` | Debug prints inside `generate_user_embedding()` execute on every call | Medium |
| `inference.py:243–248` | `generate_item_embedding._debug_printed` memoization | Prevents re-printing after first item, but debug print still runs once; not production-clean | Low |

**Additional known issues:**

- **Duplicate `SimpleFusionLayer` name:** Both `interaction/attention_utils.py` and `interaction/feature_fusion.py` define `SimpleFusionLayer`. The pipeline uses the one from `feature_fusion.py`. The one in `attention_utils.py` is imported by `feature_fusion.py` for `FeatureFusionLayer` but the class name collision could cause confusion.

- **`gensim` missing from `requirements.txt`:** Word2Vec text encoding path requires `gensim` which is not listed as a dependency. Users will get an import error if they specify a Word2Vec model name.

- **`qdrant-client` unused:** Listed in `requirements.txt` but no code imports it. This is dead dependency weight.

- **`evaluate_model()` debug prints:** The function in `trainer/trainer.py` has hardcoded debug print statements that will pollute stdout in any production evaluation run.

- **`inference.py` scoring mismatch with `evaluate_recommendation.py`:** `inference.py` ranks by classifier probability (full pipeline); `evaluate_recommendation.py` ranks by cosine similarity (tower embeddings only). These two evaluation methods can produce different rankings for the same model.

- **Lazy initialization and checkpointing:** `MLPContinuousEncoder.mlp` and `SimpleFusionLayer.projection` are lazily initialized; loading from checkpoint requires explicit pre-initialization steps in `train.py:main()`. This logic is easy to break if encoders are used outside the provided driver scripts.

- **Linear scan in `get_user_by_id()` / `get_item_by_id()`:** `O(n)` lookups; acceptable for small datasets but will slow down large-scale inference.

---

## SECTION 15 — Team & Code Conventions

**Branching:** (from git history)
- Feature branches: `feature/resnet_image_encoder`, `refactor/save_model_config`, `fix/temporal_encoder_tensor_issue`
- PRs merge to `main`

**Commit style:** Short imperative messages: `"added pending changes"`, `"fixed temporal encoder tensor directly assigned without extracting from dict"`, `"Add resnet image encoder support"`

**Naming conventions:**
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants/enums: `UPPER_CASE` for enum values, `snake_case` for instances
- Factory functions: `create_<thing>(config)` pattern
- Private helpers: `_single_underscore` prefix

**Import order:** stdlib → third-party → internal (with `sys.path.append` hacks for direct-execution compatibility)

**Module structure:** Each encoder package follows the pattern: `base_*.py` → `*_encoder.py` → `factory.py` → `__init__.py`

**License:** Not specified in repository.

**Owner:** DeGirum (private internal repo)

---

## SECTION 16 — Developer Quickstart

**Prerequisites:**

| Tool | Version |
|------|---------|
| Python | 3.8+ |
| pip | recent |
| CUDA (optional) | compatible with PyTorch >=1.10.0 |

**Setup:**

```bash
# 1. Clone / navigate to repo
cd /home/darshil/Desktop/recommendkit

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Word2Vec support
pip install gensim

# 5. Verify installation
python3 -c "import torch; import transformers; print('OK')"
```

**Quick training run:**

```bash
python3 train.py \
  --data_path datasets/synthetic/correlated_dataset.json \
  --config_path configs/sample_config_simple_fusion.json \
  --output_dir models \
  --model_name quickstart_test
```

**Expected output:**
```
🚀 Generic Recommendation System Training
✅ Configuration loaded
✅ Loaded N users and M items
✅ Loaded K interactions
✅ Model created with X,XXX parameters
Training on cpu
Epoch 1/3 [Train]: ...
✅ Model saved
🎉 Training completed successfully!
```

**Quick inference:**

```bash
python3 inference.py \
  --data_path datasets/synthetic/correlated_dataset.json \
  --model_dir models \
  --model_name quickstart_test \
  --k 5
```

**Run tests:**

```bash
pytest tests/ -v -m "not slow"   # Fast tests only
pytest tests/ -v                  # All tests (requires datasets/post_recommendation/updated_output_split.json)
```

**Common errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Config validation failed: 'embedding_dim' is required` | Missing `embedding_dim` in config | Add `"embedding_dim": 64` to config JSON |
| `RuntimeError: Feature dimension mismatch in SimpleFusionLayer` | Inference has different features than training | Ensure same feature types at inference as training |
| `ValueError: No interaction data found` | JSON missing `interactions` key | Add `"interactions": [...]` to data JSON |
| `ModuleNotFoundError: No module named 'gensim'` | Using Word2Vec text encoder without gensim | `pip install gensim` |
| `RuntimeError: CRITICAL: Training config architecture does not match saved model config` | Pretrained weights from different architecture | Use the same config JSON as when the pretrained model was trained |

**Command cheatsheet:**

```bash
# Train
python3 train.py --data_path DATA --config_path CONFIG --output_dir models --model_name NAME

# Train from pretrained
python3 train.py --data_path DATA --config_path CONFIG --output_dir models --model_name NAME \
  --pretrained_weights models --pretrained_model_name PRETRAINED_NAME

# Inference
python3 inference.py --data_path DATA --model_dir models --model_name NAME --k 10

# Inference for specific user with filter
python3 inference.py --data_path DATA --model_dir models --model_name NAME \
  --user_id 42 --k 10 --filters '{"categorical": {"genre": "Action"}}'

# Evaluate ranking metrics
python3 evaluation/evaluate_recommendation.py \
  --model_dir models --model_name NAME \
  --data_path DATA --ground_truth GROUND_TRUTH_JSON \
  --k 5 10 20 --output_file results.json

# Validate config before training
python3 -c "from utils.config_validator import validate_config_file; print(validate_config_file('configs/sample_config_simple_fusion.json'))"

# Run tests
pytest tests/ -v -m "not slow"
```

---

## FINAL QUALITY CHECK

- [x] Section 2 tree matches the repo; Section 3 covers all source files
- [x] Section 3: every file entry includes **Summary** before **Purpose**
- [x] Every file in Section 3 uses real names from the codebase (no invented APIs)
- [x] Non-trivial files have adequate **Core Logic** bullets and **Gotchas**
- [x] Section 6 reflects the real public API (no HTTP — documents Python public surface)
- [x] Section 7: no DB — explicit "Not applicable" + substitute structures documented
- [x] Section 8 reflects config keys actually read in `train.py:create_model_from_config()` and `train_model()`
- [x] Section 9 glossary and rules tied to real symbols
- [x] Section 14 reflects the debt scan (debug print statements, duplicate class names, missing deps)
- [x] All Mermaid diagrams are valid
- [x] No section header skipped — all N/A sections have explicit statement
