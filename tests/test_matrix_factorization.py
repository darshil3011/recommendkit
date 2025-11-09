#!/usr/bin/env python3
"""
Matrix Factorization Baseline Test
Simple MF model to verify the dataset is learnable
"""

import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

print('🔍 MATRIX FACTORIZATION BASELINE TEST')
print('='*70)

# Load dataset
with open('datasets/synthetic/simple_dataset.json', 'r') as f:
    dataset = json.load(f)

interactions = dataset['interactions']
print(f'Loaded {len(interactions)} interactions')

# Build user and item ID mappings
user_ids = sorted(set(i['user_id'] for i in interactions))
item_ids = sorted(set(i['item_id'] for i in interactions))

user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

print(f'Users: {len(user_ids)}, Items: {len(item_ids)}')

# Create positive interaction matrix
positive_interactions = set()
for interaction in interactions:
    user_idx = user_to_idx[interaction['user_id']]
    item_idx = item_to_idx[interaction['item_id']]
    positive_interactions.add((user_idx, item_idx))

print(f'Positive interactions: {len(positive_interactions)}')

# Create training data (balanced positive and negative)
training_data = []
user_item_pairs = []

# Add all positive interactions
for user_idx, item_idx in positive_interactions:
    training_data.append((user_idx, item_idx, 1.0))
    user_item_pairs.append((user_idx, item_idx))

# Add equal number of negative samples
np.random.seed(42)
num_negatives = len(positive_interactions)
sampled_negatives = 0

while sampled_negatives < num_negatives:
    user_idx = np.random.randint(0, len(user_ids))
    item_idx = np.random.randint(0, len(item_ids))
    
    if (user_idx, item_idx) not in positive_interactions:
        training_data.append((user_idx, item_idx, 0.0))
        user_item_pairs.append((user_idx, item_idx))
        sampled_negatives += 1

# Shuffle
np.random.shuffle(training_data)

print(f'Training samples: {len(training_data)} (50% positive, 50% negative)')

# Split into train/val
split_idx = int(0.8 * len(training_data))
train_data = training_data[:split_idx]
val_data = training_data[split_idx:]

print(f'Train: {len(train_data)}, Val: {len(val_data)}')

print('\n' + '='*70)
print('MATRIX FACTORIZATION MODEL')
print('='*70)

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize with small values
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)  # [batch_size, emb_dim]
        item_emb = self.item_embeddings(item_ids)  # [batch_size, emb_dim]
        
        # Dot product
        interaction = (user_emb * item_emb).sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Add biases
        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)
        
        prediction = interaction + user_b + item_b + self.global_bias
        return prediction.squeeze()

# Create model
model = MatrixFactorization(
    num_users=len(user_ids),
    num_items=len(item_ids),
    embedding_dim=32
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

print(f'Model created:')
print(f'  Users: {len(user_ids)}, Items: {len(item_ids)}')
print(f'  Embedding dim: 32')
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

print('\n' + '='*70)
print('TRAINING')
print('='*70)

num_epochs = 20
batch_size = 512

for epoch in range(num_epochs):
    model.train()
    
    # Shuffle training data
    np.random.shuffle(train_data)
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        
        user_batch = torch.LongTensor([x[0] for x in batch])
        item_batch = torch.LongTensor([x[1] for x in batch])
        label_batch = torch.FloatTensor([x[2] for x in batch])
        
        optimizer.zero_grad()
        
        predictions = model(user_batch, item_batch)
        loss = criterion(predictions, label_batch)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_preds == label_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
        num_batches += 1
    
    avg_train_loss = epoch_loss / num_batches
    avg_train_acc = epoch_acc / num_batches
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            
            user_batch = torch.LongTensor([x[0] for x in batch])
            item_batch = torch.LongTensor([x[1] for x in batch])
            label_batch = torch.FloatTensor([x[2] for x in batch])
            
            predictions = model(user_batch, item_batch)
            loss = criterion(predictions, label_batch)
            
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_preds == label_batch).float().mean()
            
            val_loss += loss.item()
            val_acc += accuracy.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    avg_val_acc = val_acc / val_batches
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')

print('\n' + '='*70)
print('TESTING ON SPECIFIC EXAMPLES')
print('='*70)

# Test on specific user-item pairs
model.eval()

# Get user and item data to check actual occupations/categories
user_data = dataset['user_data']
item_data = dataset['item_data']

# Find a software engineer
sw_eng_user = next(u for u in user_data if u['categorical']['occupation'] == 'software_engineer')
sw_eng_idx = user_to_idx[sw_eng_user['user_id']]

# Find tech and medical items
tech_item = next(i for i in item_data if i['categorical']['category'] == 'tech')
medical_item = next(i for i in item_data if i['categorical']['category'] == 'medical')

tech_idx = item_to_idx[tech_item['item_id']]
medical_idx = item_to_idx[medical_item['item_id']]

with torch.no_grad():
    # SW Engineer + Tech Item (should be positive)
    pred_pos = model(torch.tensor([sw_eng_idx]), torch.tensor([tech_idx]))
    prob_pos = torch.sigmoid(pred_pos).item()
    
    # SW Engineer + Medical Item (should be negative)
    pred_neg = model(torch.tensor([sw_eng_idx]), torch.tensor([medical_idx]))
    prob_neg = torch.sigmoid(pred_neg).item()
    
    print(f'\nSoftware Engineer (ID {sw_eng_user["user_id"]}):')
    print(f'  + Tech Item: {prob_pos:.4f} (should be ~1.0)')
    print(f'  + Medical Item: {prob_neg:.4f} (should be ~0.0)')
    print(f'  Difference: {prob_pos - prob_neg:.4f}')

# Test on doctor
doctor_user = next(u for u in user_data if u['categorical']['occupation'] == 'doctor')
doctor_idx = user_to_idx[doctor_user['user_id']]

with torch.no_grad():
    # Doctor + Medical Item (should be positive)
    pred_pos = model(torch.tensor([doctor_idx]), torch.tensor([medical_idx]))
    prob_pos = torch.sigmoid(pred_pos).item()
    
    # Doctor + Tech Item (should be negative)
    pred_neg = model(torch.tensor([doctor_idx]), torch.tensor([tech_idx]))
    prob_neg = torch.sigmoid(pred_neg).item()
    
    print(f'\nDoctor (ID {doctor_user["user_id"]}):')
    print(f'  + Medical Item: {prob_pos:.4f} (should be ~1.0)')
    print(f'  + Tech Item: {prob_neg:.4f} (should be ~0.0)')
    print(f'  Difference: {prob_pos - prob_neg:.4f}')

print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

if avg_val_acc > 0.9:
    print(f'✅ SUCCESS! MF achieved {avg_val_acc:.1%} accuracy')
    print(f'   → Dataset IS learnable!')
    print(f'   → Problem is with the complex architecture')
elif avg_val_acc > 0.7:
    print(f'⚠️  MODERATE: MF achieved {avg_val_acc:.1%} accuracy')
    print(f'   → Dataset partially learnable')
    print(f'   → May need more epochs or tuning')
else:
    print(f'❌ FAILURE: MF only achieved {avg_val_acc:.1%} accuracy')
    print(f'   → Dataset may have issues')
    print(f'   → Or MF not suitable for this task')

print('\n' + '='*70)

