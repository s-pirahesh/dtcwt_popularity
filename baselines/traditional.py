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

        # Reverse to make index 0 the present time
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

        # Precise implementation of average formula
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

        # Initial value
        ewma = float(time_series[0])

        # Recursive calculation
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

        # Distance from last access to present time (last index)
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
    Cache Simulator that supports both Traditional (LRU/LFU)
    and Score-Based (DTCWT, Hybrid, AF) replacement policies.
    """
    
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        # cache keys -> arbitrary value (True)
        # Using OrderedDict to track insertion/access order for LRU logic
        self.cache = OrderedDict()
        
        # For LFU: track frequency history
        self.access_count = Counter()
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    def reset(self):
        self.cache.clear()
        self.access_count.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    # --- Traditional Policies ---

    def lru_access(self, item_id: str) -> bool:
        """Standard LRU Access"""
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id) # Mark as most recent
            return True
        else:
            self.misses += 1
            self.cache[item_id] = True
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False) # Remove least recent (first item)
            return False

    def lfu_access(self, item_id: str) -> bool:
        """LFU Access with LRU Tie-breaking"""
        self.access_count[item_id] += 1
        
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id) # Update for LRU tie-breaking
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.cache_size:
                # Evict item with min frequency
                # Tie-breaker: least recently used (first in OrderedDict)
                victim = min(self.cache.keys(), key=lambda k: self.access_count[k])
                del self.cache[victim]
            
            self.cache[item_id] = True
            return False

    # --- Score-Based Policy (For DTCWT/Hybrid) ---

    def access_with_score(self, item_id: str, 
                          item_score: float, 
                          current_cache_scores: Dict[str, float]) -> bool:
        """
        Generic access for Score-Based Replacement.
        Used for AF, DTCWT, Hybrid, etc.
        
        Strategy:
        - If item in cache: HIT.
        - If miss and cache full:
            Compare item_score with MIN score in cache.
            If item_score > min_cache_score: Evict min & Insert item.
            Else: Don't cache (or cache & evict immediately).
        """
        if item_id in self.cache:
            self.hits += 1
            return True
        
        self.misses += 1
        
        if len(self.cache) < self.cache_size:
            self.cache[item_id] = True
            return False
            
        # Cache is full, decide replacement based on scores
        # Find victim (item with minimum score currently in cache)
        # Note: current_cache_scores must contain scores for keys in self.cache
        if not current_cache_scores:
            # Fallback if no scores provided: Random eviction or LRU
            self.cache.popitem(last=False)
            self.cache[item_id] = True
            return False

        # Find key in cache with lowest score
        # We only look at items that are actually IN the cache
        valid_cache_items = [k for k in self.cache.keys() if k in current_cache_scores]
        
        if not valid_cache_items:
             # Safety fallback
            self.cache.popitem(last=False)
            self.cache[item_id] = True
            return False

        victim_id = min(valid_cache_items, key=lambda k: current_cache_scores[k])
        victim_score = current_cache_scores[victim_id]
        
        if item_score > victim_score:
            # Replace victim with new item
            del self.cache[victim_id]
            self.cache[item_id] = True
        
        return False
