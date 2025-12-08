# Post Recommendation Dataset Analysis & Configuration

## 📊 Dataset Analysis

### Dataset Statistics
- **Users**: 476
- **Items**: 6,000 posts
- **Interactions**: 51,700
- **Avg interactions per user**: 108.61
- **Avg interactions per item**: 8.62

### Feature Analysis

#### User Features
- **Text** (2 fields):
  - `bio`: User biography text
  - `summary`: Short user summary
- **Categorical** (3 fields):
  - `country`: 1 unique value (likely all USA)
  - `gender`: 2 unique values (male/female)
  - `state`: ~100 unique values
- **Continuous** (2 fields):
  - `age`: User age
  - `income`: User income
- **Temporal** (2 fields):
  - `prev_50_posts`: List of up to 50 previously interacted post IDs
  - `last_10_session_times`: List of 10 session timestamps

#### Item Features
- **Text** (2 fields):
  - `title`: Post title
  - `description`: Post description
- **Categorical** (3 fields):
  - `category`: ~20 unique categories
  - `brand`: ~10 unique brands
  - `condition`: 3 unique values (new/refurbished/used)
- **Continuous** (3 fields):
  - `price`: Post price
  - `rating`: Post rating
  - `weight`: Post weight
- **Temporal** (2 fields):
  - `price_history`: List of 4 historical prices
  - `view_counts_daily`: List of 7 daily view counts

## 🎯 Configuration Optimizations

### Why Simple Fusion Instead of Attention?

After analyzing the dataset, we found:
1. **Short text fields**: Bio (~15 words), summary (~3 words), title (~10 words), description (~15 words)
2. **Small dataset**: 476 users - simpler models generalize better
3. **Few features per entity**: Users have ~3-4 features, items have ~4 features
4. **Attention is overkill**: For short sequences and small feature sets, attention adds complexity without benefit

**Simple fusion (concatenation + MLP) is more appropriate** because:
- It's faster to train and infer
- More stable gradients (no transformer collapse issues)
- Better generalization on small datasets
- Sufficient for the number of features we have (3-4 per entity)

### Key Configuration Choices

#### 1. **Embedding Dimension: 128** (vs 64 in baseline)
- **Rationale**: Larger dataset with rich features benefits from higher-dimensional embeddings
- **Impact**: Better representation capacity for text and categorical features

#### 2. **Simple Fusion (Not Attention)**
- **Method**: Concatenation + MLP
- **Rationale**: Text fields are short (~3-15 words), dataset is small (476 users), and we only have 3-4 features per entity
- **Impact**: Faster training, better generalization, more stable - attention would be overkill and could overfit

#### 3. **Categorical Encoder**
- **Fields**: 3 fields (country, gender, state for users; category, brand, condition for items)
- **MLP Hidden Dims**: [64] - adds non-linearity
- **Hash Vocab Size**: 8192 (vs 4096) - accommodates more unique categorical values
- **Rationale**: Multiple categorical fields need proper encoding with non-linear transformations

#### 4. **Text Encoder**
- **Max Length**: 128 (sufficient for short text fields)
- **Embedding Dim**: 128 - matches overall embedding dimension
- **Aggregation**: `separate_concat` - preserves field-specific information
- **Rationale**: Text fields are short, so 128 tokens is sufficient; preserves field distinctions

#### 5. **Temporal Encoder**
- **Bidirectional LSTM**: Enabled
- **Max Sequence Length**: 50 (matches prev_50_posts)
- **Item Lookup**: Enabled (for prev_50_posts)
- **Rationale**: Bidirectional captures both forward and backward temporal patterns

#### 6. **Training Parameters**
- **Batch Size**: 64 (vs 32) - better gradient estimates with larger dataset
- **Epochs**: 8 (vs 3) - sufficient for simpler architecture, avoids overfitting
- **Learning Rate**: 0.001 (vs 0.0005) - simpler model can handle higher learning rate
- **Dropout**: 0.1-0.2 - prevents overfitting

#### 7. **Classifier**
- **Hidden Dims**: [128, 64] - two-layer MLP for better decision boundaries
- **Dropout**: 0.2 - higher dropout in final classifier

## 📈 Expected Improvements

Compared to baseline simple fusion config:
1. **Better Feature Representation**: 128-dim embeddings capture richer text and categorical features
2. **Enhanced Categorical Encoding**: MLP layers in categorical encoder improve feature interactions
3. **Temporal Patterns**: Bidirectional LSTM better captures user behavior sequences
4. **Scalability**: Larger embeddings handle 6,000 items better
5. **Stability**: Simple fusion avoids transformer collapse and gradient issues
6. **Generalization**: Simpler architecture generalizes better on small dataset (476 users)

## 🔧 Alternative Configurations

### If Training is Too Slow:
- Reduce `embedding_dim` to 96
- Reduce `num_epochs` to 5-6
- Reduce `batch_size` to 32

### If Overfitting:
- Increase `dropout` to 0.2-0.3
- Reduce `classifier_hidden_dims` to [64]
- Reduce `embedding_dim` to 96
- Add regularization

### If Underfitting:
- Increase `num_epochs` to 10-12
- Increase `embedding_dim` to 256
- Add more hidden layers to continuous encoder
- Consider trying attention fusion (though likely overkill for short text)

## 🚀 Usage

The optimized configuration is saved at:
```
configs/post_recommendation_config.json
```

Train with:
```bash
python3 train.py \
  --data_path datasets/post_recommendation/updated_output_split.json \
  --config_path configs/post_recommendation_config.json \
  --output_dir models \
  --model_name post_recommendation_model
```

## 📝 Notes

- The dataset has no image features, so `image_encoder_config` is set to `null`
- All temporal features use item lookup for `prev_50_posts`
- The config is optimized for the specific feature structure of this dataset

