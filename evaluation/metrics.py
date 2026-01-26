"""
Evaluation Metrics for Popularity Assessment
"""
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score


class AssessmentMetrics:
    """
    Metrics for evaluating popularity assessment methods
    """
    
    @staticmethod
    def hit_rate(ranking: List[Any], true_popular: set, top_k_ratio: float = 0.10) -> float:
        """
        Hit Rate: What fraction of truly popular items are in predicted top-K?
        
        Args:
            ranking: Ranked list of items (by predicted popularity)
            true_popular: Set of truly popular items
            top_k_ratio: Fraction of top items to consider (e.g., 0.10 = top 10%)
            
        Returns:
            Hit rate (0 to 1)
        """
        k = max(1, int(len(ranking) * top_k_ratio))
        predicted_top_k = set(ranking[:k])
        
        if len(true_popular) == 0:
            return 0.0
        
        hits = len(predicted_top_k.intersection(true_popular))
        hit_rate = hits / len(true_popular)
        
        return hit_rate
    
    @staticmethod
    def precision_at_k(ranking: List[Any], true_popular: set, top_k_ratio: float = 0.10) -> float:
        """
        Precision@K: What fraction of predicted top-K are truly popular?
        
        Args:
            ranking: Ranked list of items
            true_popular: Set of truly popular items
            top_k_ratio: Fraction of top items
            
        Returns:
            Precision (0 to 1)
        """
        k = max(1, int(len(ranking) * top_k_ratio))
        predicted_top_k = set(ranking[:k])
        
        if len(predicted_top_k) == 0:
            return 0.0
        
        tp = len(predicted_top_k.intersection(true_popular))
        precision = tp / len(predicted_top_k)
        
        return precision
    
    @staticmethod
    def false_positive_rate(ranking: List[Any], true_popular: set, 
                           top_k_ratio: float = 0.10) -> float:
        """
        False Positive Rate: What fraction of predicted top-K are NOT popular?
        
        Args:
            ranking: Ranked list of items
            true_popular: Set of truly popular items
            top_k_ratio: Fraction of top items
            
        Returns:
            FP rate (0 to 1)
        """
        k = max(1, int(len(ranking) * top_k_ratio))
        predicted_top_k = set(ranking[:k])
        
        fp = len(predicted_top_k - true_popular)
        fp_rate = fp / len(predicted_top_k)
        
        return fp_rate
    
    @staticmethod
    def ndcg_at_k(ranking: List[Any], relevance_scores: Dict[Any, float],
                  top_k_ratio: float = 0.10) -> float:
        """
        Normalized Discounted Cumulative Gain
        
        Args:
            ranking: Ranked list of items
            relevance_scores: Dictionary of item -> relevance score
            top_k_ratio: Fraction of top items
            
        Returns:
            NDCG score (0 to 1)
        """
        k = max(1, int(len(ranking) * top_k_ratio))
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(ranking[:k]):
            relevance = relevance_scores.get(item, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Compute ideal DCG (sort by relevance)
        sorted_items = sorted(relevance_scores.keys(), 
                            key=lambda x: relevance_scores[x], reverse=True)
        idcg = 0.0
        for i, item in enumerate(sorted_items[:k]):
            relevance = relevance_scores[item]
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_all_metrics(ranking: List[Any], 
                           future_accesses: Dict[Any, float],
                           top_k_ratios: List[float] = [0.05, 0.10, 0.20]) -> Dict[str, Dict]:
        """
        Compute all metrics for multiple top-K ratios
        
        Args:
            ranking: Ranked list of items
            future_accesses: Dictionary of item -> future access count
            top_k_ratios: List of top-K ratios to evaluate
            
        Returns:
            Nested dictionary: {top_k: {metric: value}}
        """
        results = {}
        
        for ratio in top_k_ratios:
            # Determine true popular items (top ratio by future accesses)
            k_true = max(1, int(len(future_accesses) * ratio))
            sorted_by_future = sorted(future_accesses.keys(),
                                     key=lambda x: future_accesses[x],
                                     reverse=True)
            true_popular = set(sorted_by_future[:k_true])
            
            # Compute metrics
            metrics = {
                'hit_rate': AssessmentMetrics.hit_rate(ranking, true_popular, ratio),
                'precision': AssessmentMetrics.precision_at_k(ranking, true_popular, ratio),
                'fp_rate': AssessmentMetrics.false_positive_rate(ranking, true_popular, ratio),
                'ndcg': AssessmentMetrics.ndcg_at_k(ranking, future_accesses, ratio),
            }
            
            # F1 score
            if metrics['precision'] + metrics['hit_rate'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['hit_rate']) / \
                               (metrics['precision'] + metrics['hit_rate'])
            else:
                metrics['f1'] = 0.0
            
            results[f'top_{int(ratio*100)}%'] = metrics
        
        return results
    
    @staticmethod
    def spearman_correlation(ranking1: List[Any], ranking2: List[Any]) -> float:
        """
        Compute Spearman rank correlation between two rankings
        
        Args:
            ranking1: First ranking
            ranking2: Second ranking
            
        Returns:
            Spearman correlation (-1 to 1)
        """
        from scipy.stats import spearmanr
        
        # Create rank dictionaries
        rank1 = {item: i for i, item in enumerate(ranking1)}
        rank2 = {item: i for i, item in enumerate(ranking2)}
        
        # Get common items
        common_items = set(rank1.keys()).intersection(set(rank2.keys()))
        
        if len(common_items) < 2:
            return 0.0
        
        # Get ranks for common items
        ranks1 = [rank1[item] for item in common_items]
        ranks2 = [rank2[item] for item in common_items]
        
        correlation, _ = spearmanr(ranks1, ranks2)
        
        return correlation if not np.isnan(correlation) else 0.0


class PredictionMetrics:
    """
    Metrics for evaluating prediction methods (secondary component)
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² Score"""
        from sklearn.metrics import r2_score as sklearn_r2
        return float(sklearn_r2(y_true, y_pred))
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all prediction metrics"""
        return {
            'mae': PredictionMetrics.mae(y_true, y_pred),
            'rmse': PredictionMetrics.rmse(y_true, y_pred),
            'mape': PredictionMetrics.mape(y_true, y_pred),
            'r2': PredictionMetrics.r2_score(y_true, y_pred),
        }
