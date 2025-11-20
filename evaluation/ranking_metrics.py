"""
Ranking metrics for recommendation system evaluation
Implements precision@k and recall@k metrics
"""

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
    
    return metrics

