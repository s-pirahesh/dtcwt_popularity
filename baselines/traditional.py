"""
Traditional Baseline Methods
AF, LRU, LFU, EWMA
"""
import numpy as np
from typing import List, Dict
from collections import Counter, OrderedDict


class TraditionalBaselines:
    """
    Traditional caching/popularity prediction methods
    Used as baselines for comparison
    """
    
    @staticmethod
    def access_frequency(time_series: np.ndarray) -> float:
        """
        Simple Access Frequency (AF)
        Score = Total access count
        
        Args:
            time_series: Time series of accesses
            
        Returns:
            Total access count
        """
        return float(np.sum(time_series))
    
    @staticmethod
    def lru_score(time_series: np.ndarray) -> float:
        """
        Least Recently Used (LRU) score
        Score = Recency-weighted sum
        
        Args:
            time_series: Time series of accesses
            
        Returns:
            Recency-weighted score
        """
        if len(time_series) == 0:
            return 0.0
        
        # Find last non-zero access
        nonzero_indices = np.nonzero(time_series)[0]
        
        if len(nonzero_indices) == 0:
            return 0.0
        
        # Score = 1 / time_since_last_access
        last_access = nonzero_indices[-1]
        time_since = len(time_series) - last_access
        
        score = 1.0 / (time_since + 1)
        
        return float(score)
    
    @staticmethod
    def lfu_score(time_series: np.ndarray) -> float:
        """
        Least Frequently Used (LFU) score
        Similar to AF but normalized by time window
        
        Args:
            time_series: Time series of accesses
            
        Returns:
            Average access frequency
        """
        if len(time_series) == 0:
            return 0.0
        
        return float(np.mean(time_series))
    
    @staticmethod
    def ewma_score(time_series: np.ndarray, alpha: float = 0.3) -> float:
        """
        Exponentially Weighted Moving Average (EWMA)
        Recent accesses weighted more heavily
        
        Args:
            time_series: Time series of accesses
            alpha: Smoothing parameter (0-1), higher = more weight on recent
            
        Returns:
            EWMA score
        """
        if len(time_series) == 0:
            return 0.0
        
        # Initialize with first value
        ewma = time_series[0]
        
        # Update with exponential weighting
        for value in time_series[1:]:
            ewma = alpha * value + (1 - alpha) * ewma
        
        return float(ewma)
    
    @staticmethod
    def batch_assess_all(time_series_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute all baseline scores for a list of time series
        
        Args:
            time_series_list: List of time series
            
        Returns:
            Dictionary of method_name -> scores array
        """
        results = {
            'AF': np.array([TraditionalBaselines.access_frequency(ts) 
                           for ts in time_series_list]),
            'LRU': np.array([TraditionalBaselines.lru_score(ts) 
                            for ts in time_series_list]),
            'LFU': np.array([TraditionalBaselines.lfu_score(ts) 
                            for ts in time_series_list]),
            'EWMA': np.array([TraditionalBaselines.ewma_score(ts) 
                             for ts in time_series_list]),
        }
        
        return results


class CacheSimulator:
    """
    Simple cache simulator for testing baseline performance
    """
    
    def __init__(self, cache_size: int):
        """
        Initialize cache
        
        Args:
            cache_size: Maximum number of items in cache
        """
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.access_count = Counter()
        self.hits = 0
        self.misses = 0
    
    def lru_access(self, item_id: str) -> bool:
        """
        LRU cache access
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if hit, False if miss
        """
        if item_id in self.cache:
            # Hit: move to end (most recent)
            self.cache.move_to_end(item_id)
            self.hits += 1
            return True
        else:
            # Miss: add item
            self.cache[item_id] = True
            
            # Evict if over capacity
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            
            self.misses += 1
            return False
    
    def lfu_access(self, item_id: str) -> bool:
        """
        LFU cache access
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if hit, False if miss
        """
        self.access_count[item_id] += 1
        
        if item_id in self.cache:
            self.hits += 1
            return True
        else:
            # Miss: add item
            self.cache[item_id] = True
            
            # Evict least frequently used if over capacity
            if len(self.cache) > self.cache_size:
                # Find item with minimum access count
                lfu_item = min(self.cache.keys(), 
                              key=lambda k: self.access_count[k])
                del self.cache[lfu_item]
            
            self.misses += 1
            return False
    
    def get_hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        """Reset cache statistics"""
        self.cache.clear()
        self.access_count.clear()
        self.hits = 0
        self.misses = 0
