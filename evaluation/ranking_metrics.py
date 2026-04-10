"""
Ranking metrics for recommendation system evaluation
Implements Precision@K, Recall@K, HR@K, and NDCG@K metrics
"""

import math
from typing import List, Set


def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Precision@K
    
    Args:
        recommended_items: List of recommended item IDs (ordered by relevance)
        relevant_items: Set of relevant item IDs (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    # Get top-k recommendations
    top_k = recommended_items[:k]
    
    # Count how many of top-k are relevant
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    
    # Precision = relevant items in top-k / k
    return relevant_in_top_k / k


def recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Recall@K
    
    Args:
        recommended_items: List of recommended item IDs (ordered by relevance)
        relevant_items: Set of relevant item IDs (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    if k == 0:
        return 0.0
    
    # Get top-k recommendations
    top_k = recommended_items[:k]
    
    # Count how many relevant items are in top-k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    
    # Recall = relevant items in top-k / total relevant items
    return relevant_in_top_k / len(relevant_items)


def hr_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Hit Rate@K — 1 if at least one relevant item is in top-K, else 0.

    This is the standard metric used in two-tower recommendation papers.
    """
    if k == 0 or len(relevant_items) == 0:
        return 0.0
    top_k = set(recommended_items[:k])
    return 1.0 if top_k & relevant_items else 0.0


def ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    DCG@K  = sum(1 / log2(i+2)) for each hit at position i in top-K
    IDCG@K = DCG of perfect ranking = sum(1 / log2(i+2)) for i in 0..min(|relevant|,K)-1
    NDCG@K = DCG@K / IDCG@K
    """
    if k == 0 or len(relevant_items) == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended_items[:k])
        if item in relevant_items
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_items), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics_for_user(
    recommended_items: List[int],
    relevant_items: Set[int],
    k_values: List[int]
) -> dict:
    """
    Compute precision@k and recall@k for multiple k values for a single user
    
    Args:
        recommended_items: List of recommended item IDs (ordered by relevance)
        relevant_items: Set of relevant item IDs (ground truth)
        k_values: List of k values to compute metrics for
        
    Returns:
        Dictionary with metrics for each k value
    """
    metrics = {}

    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(recommended_items, relevant_items, k)
        metrics[f'recall@{k}'] = recall_at_k(recommended_items, relevant_items, k)
        metrics[f'hr@{k}'] = hr_at_k(recommended_items, relevant_items, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(recommended_items, relevant_items, k)

    return metrics
