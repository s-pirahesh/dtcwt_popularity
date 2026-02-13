"""
Evaluation Metrics Module
=========================
This module implements the core metrics defined in the Frozen Evaluation Protocol.

Metrics included:
1. Decision Layer: NDCG@K (Log-Relevance), HitRatio@K (Static Placement)
2. Diagnostic Layer: Kendall's Tau, Spearman's Rho, MAE (Sanity Check)
3. Stability Layer: RSI (Ranking Stability Index)
4. Robustness Layer: Rank Distortion (for Noise Injection test)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Union, Set

def calculate_ndcg(predicted_scores: np.ndarray, actual_views: np.ndarray, k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at K.
    
    Protocol Adherence:
        - Uses log-relevance: rel = log10(1 + actual_views) to handle heavy-tails.
        - Evaluates top-K recommendations based on predicted scores.
    
    Args:
        predicted_scores: Array of scores output by the model (WSPI/AF).
        actual_views: Array of ground truth view counts (future window).
        k: Cut-off rank.
        
    Returns:
        float: NDCG score between 0.0 and 1.0.
    """
    if len(predicted_scores) == 0 or k <= 0:
        return 0.0
        
    # 1. Define Relevance (Logarithmic to dampen viral outliers)
    relevance = np.log10(1 + actual_views)
    
    # 2. Get Top-K indices based on Predictions (Model's Ranking)
    # argsort gives ascending, so we take the last k and reverse them
    if len(predicted_scores) < k:
        k = len(predicted_scores)
    
    pred_indices = np.argsort(predicted_scores)[-k:][::-1]
    
    # 3. Calculate DCG (Discounted Cumulative Gain)
    # DCG = sum( (2^rel - 1) / log2(i + 2) )
    pred_rel = relevance[pred_indices]
    discounts = np.log2(np.arange(len(pred_rel)) + 2)
    dcg = np.sum((np.power(2, pred_rel) - 1) / discounts)
    
    # 4. Calculate IDCG (Ideal DCG)
    # Sort by actual relevance to get the ideal ranking
    ideal_indices = np.argsort(relevance)[-k:][::-1]
    ideal_rel = relevance[ideal_indices]
    idcg = np.sum((np.power(2, ideal_rel) - 1) / discounts)
    
    # 5. Final NDCG
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

def calculate_hit_rate(predicted_scores: np.ndarray, actual_views: np.ndarray, k: int = 10) -> float:
    """
    Calculates Cache Hit Ratio (CHR) @ K assuming Static Placement.
    
    Protocol Adherence:
        - Assumes Top-K items are placed in cache at the start.
        - Hit Ratio = (Sum of views of Top-K items) / (Total views of all items)
    """
    total_views = np.sum(actual_views)
    if total_views == 0:
        return 0.0
        
    # Identify Top-K items proposed by the model
    if len(predicted_scores) < k:
        k = len(predicted_scores)
        
    top_k_indices = np.argsort(predicted_scores)[-k:]
    
    # Calculate hits (views captured by these items)
    hits = np.sum(actual_views[top_k_indices])
    
    return float(hits / total_views)

def calculate_diagnostics(predicted_scores: np.ndarray, actual_views: np.ndarray) -> Dict[str, float]:
    """
    Calculates diagnostic metrics (Kendall, Spearman, MAE).
    
    Protocol Adherence:
        - Kendall Tau: For precise inversion counting.
        - Spearman Rho: For global trend correlation.
        - MAE: Sanity check only (mainly for baselines).
    """
    # Rank Correlation Metrics
    # Note: Kendall is O(N^2), might be slow for huge arrays, 
    # but acceptable for standard cache simulation sizes.
    kendall, _ = stats.kendalltau(predicted_scores, actual_views)
    spearman, _ = stats.spearmanr(predicted_scores, actual_views)
    
    # Error Metric (Sanity Check)
    # We compare normalized scores to avoid scale issues, or raw values if comparing counts.
    # Here we perform raw MAE for simplicity as requested for baselines.
    mae = np.mean(np.abs(predicted_scores - actual_views))
    
    return {
        'kendall_tau': float(kendall) if not np.isnan(kendall) else 0.0,
        'spearman_rho': float(spearman) if not np.isnan(spearman) else 0.0,
        'mae': float(mae)
    }

def calculate_rsi(top_k_t1: List[int], top_k_t2: List[int]) -> float:
    """
    Calculates Ranking Stability Index (RSI) using Jaccard Similarity.
    
    Args:
        top_k_t1: List of item indices in Top-K at time t.
        top_k_t2: List of item indices in Top-K at time t+1.
        
    Returns:
        float: Jaccard similarity (Intersection / Union).
    """
    set_t1 = set(top_k_t1)
    set_t2 = set(top_k_t2)
    
    intersection = len(set_t1.intersection(set_t2))
    union = len(set_t1.union(set_t2))
    
    if union == 0:
        return 0.0
        
    return float(intersection / union)

def calculate_rank_distortion(predicted_scores_clean: np.ndarray, 
                              predicted_scores_noisy: np.ndarray, 
                              target_index: int) -> int:
    """
    Calculates Rank Distortion for the Robustness Test.
    
    Args:
        predicted_scores_clean: Model scores before noise injection.
        predicted_scores_noisy: Model scores after noise injection.
        target_index: Index of the specific item where noise was injected.
        
    Returns:
        int: Absolute change in rank (Delta R).
    """
    # Calculate rank in clean list (Higher score = Better rank, so Rank 1 is top)
    # We use argsort twice to get ranks. 
    # Example: scores=[10, 30, 20] -> argsort=[0, 2, 1] -> argsort=[0, 2, 1] -> ranks=[2, 0, 1] (if 0 is best)
    # Here we want standard rank: 1st, 2nd, 3rd...
    
    def get_rank(scores, idx):
        # Sort descending
        sorted_indices = np.argsort(scores)[::-1]
        # Find where idx is in the sorted list
        rank = np.where(sorted_indices == idx)[0][0] + 1
        return rank

    rank_clean = get_rank(predicted_scores_clean, target_index)
    rank_noisy = get_rank(predicted_scores_noisy, target_index)
    
    return abs(rank_noisy - rank_clean)