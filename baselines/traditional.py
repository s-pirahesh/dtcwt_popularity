"""
Traditional Baseline Methods
AF, LRU, LFU, EWMA
"""
import numpy as np
from typing import List, Dict
from collections import Counter, OrderedDict


"""
Traditional Baseline Methods
Based on definitions in Baseline_Popularity_methods.docx
"""
import numpy as np
from typing import List, Dict


class TraditionalBaselines:
    """
    Traditional caching/popularity prediction methods
    """
    
    @staticmethod
    def access_frequency(time_series: np.ndarray) -> float:
        """
        Weighted Access Frequency (AF)
        Ref [34]: Assigns weights 2^-i to history.
        Recent accesses have higher weight (1, 0.5, 0.25, ...).
        """
        if len(time_series) == 0:
            return 0.0
            
        # معکوس کردن برای اینکه اندیس 0 زمان حال باشد
        reversed_ts = time_series[::-1]
        score = 0.0
        
        for i, val in enumerate(reversed_ts):
            weight = 2.0 ** (-i)
            score += weight * val
                
        return float(score)
    
    @staticmethod
    def lfu_score(time_series: np.ndarray) -> float:
        """
        Least Frequently Used (LFU)
        Ref [32] Eq(5): Mean request frequency per period.
        Formula: Sum(Requests) / Num_Periods
        """
        if len(time_series) == 0:
            return 0.0
        
        # پیاده‌سازی دقیق فرمول میانگین
        return float(np.mean(time_series))
    
    @staticmethod
    def ewma_score(time_series: np.ndarray, alpha: float = 0.2) -> float:
        """
        Exponentially Weighted Moving Average (EWMA)
        Ref [39] Eq(13): Recurrent formula for popularity.
        P(t) = alpha * R(t) + (1-alpha) * P(t-1)
        """
        if len(time_series) == 0:
            return 0.0
        
        # مقدار اولیه
        ewma = float(time_series[0])
        
        # محاسبه بازگشتی
        for val in time_series[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        
        return float(ewma)
    
    @staticmethod
    def lru_score(time_series: np.ndarray) -> float:
        """
        Least Recently Used (LRU)
        Ref [33]: Based on time since last access.
        """
        if len(time_series) == 0:
            return 0.0
        
        nonzero_indices = np.nonzero(time_series)[0]
        
        if len(nonzero_indices) == 0:
            return 0.0
        
        # فاصله از آخرین بازدید تا زمان حال (آخرین اندیس)
        last_access_idx = nonzero_indices[-1]
        current_time_idx = len(time_series) - 1
        dist = current_time_idx - last_access_idx
        
        return 1.0 / (dist + 1.0)

    @staticmethod
    def batch_assess_all(time_series_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute all baseline scores efficiently"""
        n = len(time_series_list)
        results = {
            'AF': np.zeros(n),
            'LFU': np.zeros(n),
            'EWMA': np.zeros(n),
            'LRU': np.zeros(n)
        }
        
        for i, ts in enumerate(time_series_list):
            results['AF'][i] = TraditionalBaselines.access_frequency(ts)
            results['LFU'][i] = TraditionalBaselines.lfu_score(ts)
            results['EWMA'][i] = TraditionalBaselines.ewma_score(ts)
            results['LRU'][i] = TraditionalBaselines.lru_score(ts)
            
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
