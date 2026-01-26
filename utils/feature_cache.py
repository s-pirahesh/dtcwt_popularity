"""
Feature Caching Utilities (V3.1)
LRU cache for expensive feature computations
"""
import pickle
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional, Callable
import numpy as np


class FeatureCache:
    """
    Feature caching system for expensive computations
    
    Supports:
    - Memory caching (LRU)
    - Disk caching (pickle)
    - Automatic cache invalidation
    """
    
    def __init__(self, 
                 cache_dir: str = None,
                 memory_size: int = 1000,
                 disk_enabled: bool = False):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory for disk cache
            memory_size: LRU cache size
            disk_enabled: Enable disk caching
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_size = memory_size
        self.disk_enabled = disk_enabled
        
        if self.disk_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache (simple dict, can be upgraded to LRU)
        self._memory_cache = {}
        self._access_order = []
    
    def _get_cache_key(self, data: Any, params: dict = None) -> str:
        """
        Generate cache key from data and parameters
        
        Args:
            data: Input data (numpy array or list)
            params: Additional parameters
            
        Returns:
            Cache key (hash)
        """
        # Convert data to bytes
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()
        
        # Add parameters
        if params:
            param_str = str(sorted(params.items()))
            data_bytes += param_str.encode()
        
        # Generate hash
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Check memory cache
        if key in self._memory_cache:
            # Update access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory_cache[key]
        
        # Check disk cache
        if self.disk_enabled and self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory(key, value)
                
                return value
        
        return None
    
    def put(self, key: str, value: Any):
        """
        Put item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Add to memory cache
        self._add_to_memory(key, value)
        
        # Add to disk cache if enabled
        if self.disk_enabled and self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
    
    def _add_to_memory(self, key: str, value: Any):
        """Add item to memory cache with LRU eviction"""
        # Add to cache
        self._memory_cache[key] = value
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Evict if over capacity
        while len(self._memory_cache) > self.memory_size:
            # Remove oldest (LRU)
            oldest_key = self._access_order.pop(0)
            del self._memory_cache[oldest_key]
    
    def cached_computation(self, 
                          data: Any,
                          compute_fn: Callable,
                          params: dict = None) -> Any:
        """
        Compute with caching
        
        Args:
            data: Input data
            compute_fn: Function to compute features
            params: Additional parameters
            
        Returns:
            Computed features (from cache or newly computed)
        """
        # Generate cache key
        cache_key = self._get_cache_key(data, params)
        
        # Check cache
        cached_value = self.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Compute
        value = compute_fn(data)
        
        # Cache result
        self.put(cache_key, value)
        
        return value
    
    def clear(self):
        """Clear all caches"""
        self._memory_cache.clear()
        self._access_order.clear()
        
        if self.disk_enabled and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_cache_size(self) -> dict:
        """Get cache statistics"""
        stats = {
            'memory_items': len(self._memory_cache),
            'memory_capacity': self.memory_size,
        }
        
        if self.disk_enabled and self.cache_dir:
            disk_files = list(self.cache_dir.glob("*.pkl"))
            stats['disk_items'] = len(disk_files)
            stats['disk_size_mb'] = sum(f.stat().st_size for f in disk_files) / (1024*1024)
        
        return stats


# Global cache instance
_global_cache = None


def get_global_cache(cache_dir: str = None, 
                    memory_size: int = 1000,
                    disk_enabled: bool = False) -> FeatureCache:
    """
    Get or create global cache instance
    
    Args:
        cache_dir: Cache directory
        memory_size: Memory cache size
        disk_enabled: Enable disk caching
        
    Returns:
        Global FeatureCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = FeatureCache(
            cache_dir=cache_dir,
            memory_size=memory_size,
            disk_enabled=disk_enabled
        )
    
    return _global_cache


# Decorator for caching function results
def cache_features(cache_key_fn: Optional[Callable] = None):
    """
    Decorator for caching expensive feature computations
    
    Args:
        cache_key_fn: Optional function to generate cache key
    
    Usage:
        @cache_features()
        def compute_dtcwt_features(time_series):
            # expensive computation
            return features
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # Generate cache key
            if cache_key_fn:
                key = cache_key_fn(*args, **kwargs)
            else:
                # Use first argument as key
                if len(args) > 0:
                    key = cache._get_cache_key(args[0], kwargs)
                else:
                    key = cache._get_cache_key((), kwargs)
            
            # Check cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Compute
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(key, result)
            
            return result
        
        return wrapper
    
    return decorator
