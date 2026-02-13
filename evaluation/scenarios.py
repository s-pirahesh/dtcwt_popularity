"""
Evaluation Scenarios Module
===========================
This module handles the generation of synthetic scenarios (specifically noise injection)
to test the robustness of popularity assessment methods.

It is designed to be:
1. Dynamic: Works with any time-series dataset (numpy arrays).
2. Statistical: Selects a statistical sample of items, not just one.
3. Plug-and-Play: Can be called within the evaluation loop.
"""

import numpy as np
from typing import List, Tuple, Dict

class RobustnessScenario:
    """
    Manages the 'Noise Injection' stress test.
    """
    
    def __init__(self, sample_size: int = 50, spike_multiplier: float = 10.0):
        """
        Args:
            sample_size: Number of items to test (default 50 for statistical significance).
            spike_multiplier: Magnitude of spike relative to item's mean (e.g., 10x).
        """
        self.sample_size = sample_size
        self.spike_multiplier = spike_multiplier

    def select_stable_candidates(self, data_window: np.ndarray) -> List[int]:
        """
        Selects indices of items that are 'Low-Popularity' and 'Stable'.
        We don't want to inject noise into already viral items (it might get lost).
        We want to see if a 'dead' item suddenly jumps up due to noise.
        
        Args:
            data_window: The history window (Items x Time).
            
        Returns:
            List[int]: Indices of selected items.
        """
        num_items = data_window.shape[0]
        if num_items < self.sample_size:
            return list(range(num_items))
            
        # 1. Calculate Mean and Variance for all items
        means = np.mean(data_window, axis=1)
        variances = np.var(data_window, axis=1)
        
        # 2. Filter: We want items that have SOME views (mean > 0) but are not viral.
        # Let's pick items in the lower quartile (0-25%) of popularity, but non-zero.
        non_zero_indices = np.where(means > 0)[0]
        
        if len(non_zero_indices) < self.sample_size:
            return non_zero_indices.tolist()
            
        non_zero_means = means[non_zero_indices]
        
        # Find threshold for "Low Popularity" (e.g., median of non-zero items)
        threshold = np.percentile(non_zero_means, 50)
        
        # Candidates: Low mean AND Low variance (Stable)
        candidates = []
        for idx in non_zero_indices:
            if means[idx] <= threshold:
                candidates.append(idx)
                
        # 3. Randomly select from candidates to avoid bias
        if len(candidates) > self.sample_size:
            selected = np.random.choice(candidates, self.sample_size, replace=False)
            return selected.tolist()
        else:
            return candidates

    def inject_spike(self, time_series: np.ndarray) -> np.ndarray:
        """
        Creates a copy of the time series and injects a spike at the last timestamp.
        
        Args:
            time_series: 1D array of a single item's history.
            
        Returns:
            np.ndarray: Noisy copy of the time series.
        """
        noisy_series = time_series.copy()
        
        # Calculate spike magnitude based on history
        # Add epsilon to handle cases where mean is extremely small
        mean_val = np.mean(time_series)
        spike_val = mean_val * self.spike_multiplier
        
        # Ensure spike is at least 1 (for integer count logic)
        if spike_val < 1: 
            spike_val = 1
            
        # Add to the last time slot
        noisy_series[-1] += spike_val
        
        return noisy_series

    def measure_rank_distortion(self, 
                              scores_clean: np.ndarray, 
                              scores_noisy: np.ndarray, 
                              target_indices: List[int]) -> float:
        """
        Calculates the Average Rank Distortion across all tested items.
        
        Args:
            scores_clean: Array of scores for ALL items (before noise).
            scores_noisy: Array of scores for ALL items (after noise injection on targets).
                          *Note: In reality, we inject noise one by one or batch, 
                           but usually we compare 'Item X Clean Rank' vs 'Item X Noisy Rank'
                           within the context of the whole population.
            target_indices: Indices of items that received noise.
            
        Returns:
            float: Average change in rank (lower is better).
        """
        distortions = []
        
        # Convert scores to ranks (Higher score = Rank 1)
        # We use scipy.stats.rankdata or simple argsort. 
        # Let's use simple logic: Rank = (Number of items with score > my_score) + 1
        
        for idx in target_indices:
            # 1. Get Clean Rank
            score_c = scores_clean[idx]
            # Count how many items scored higher than this item
            rank_clean = np.sum(scores_clean > score_c) + 1
            
            # 2. Get Noisy Rank
            # CAUTION: When measuring distortion for Item X, we assume 
            # only Item X is noisy, or we compare its new score against the *clean* population
            # to see how much it jumped.
            # Comparing Noisy X against Clean Population is the standard robustness test.
            score_n = scores_noisy[idx]
            rank_noisy = np.sum(scores_clean > score_n) + 1 # Compare against clean peers
            
            # 3. Calculate Distortion
            # If rank improved from 1000 to 10, distortion is 990.
            distortions.append(abs(rank_clean - rank_noisy))
            
        if not distortions:
            return 0.0
            
        return float(np.mean(distortions))