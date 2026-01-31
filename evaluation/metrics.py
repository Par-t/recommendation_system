"""
Ranking metrics for recommendation evaluation.

Recall@K: What fraction of relevant items appear in top-K?
NDCG@K: How well are relevant items ranked within top-K?
"""

import numpy as np


def recall_at_k(ranked_items: np.ndarray, ground_truth: set, k: int) -> float:
    """
    Recall@K: fraction of ground truth items found in top-K predictions.
    
    Args:
        ranked_items: Array of item indices, sorted by predicted score (best first)
        ground_truth: Set of item indices the user actually interacted with
        k: Number of top items to consider
        
    Returns:
        Recall score between 0 and 1
    """
    top_k = set(ranked_items[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(ranked_items: np.ndarray, ground_truth: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    Rewards relevant items appearing earlier in the ranking.
    
    Args:
        ranked_items: Array of item indices, sorted by predicted score (best first)
        ground_truth: Set of item indices the user actually interacted with
        k: Number of top items to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    top_k = ranked_items[:k]
    
    # DCG: sum of (relevance / log2(position + 2))
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # +2 because positions are 1-indexed in NDCG
    
    # IDCG: best possible DCG (all relevant items at top)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, user_items_test: dict, num_items: int, k_values: list = [10, 20]) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained NCF model
        user_items_test: Dict mapping user_idx -> set of ground truth item indices
        num_items: Total number of items
        k_values: List of K values for Recall@K and NDCG@K
        
    Returns:
        Dict of metric_name -> average score
    """
    import torch
    
    model.eval()
    all_items = torch.arange(num_items)
    
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})
    
    with torch.no_grad():
        for user_idx, ground_truth in user_items_test.items():
            # Score all items for this user
            user_tensor = torch.full((num_items,), user_idx, dtype=torch.long)
            scores = model(user_tensor, all_items).numpy()
            
            # Rank items by score (descending)
            ranked_items = np.argsort(-scores)
            
            # Compute metrics
            for k in k_values:
                results[f"recall@{k}"].append(recall_at_k(ranked_items, ground_truth, k))
                results[f"ndcg@{k}"].append(ndcg_at_k(ranked_items, ground_truth, k))
    
    # Average across users
    return {metric: np.mean(scores) for metric, scores in results.items()}
