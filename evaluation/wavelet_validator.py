# -*- coding: utf-8 -*-
"""
Wavelet Window Size Validator
Ensures proper window size for wavelet decomposition
Author: Sajjad Pirahesh
Date: February 2025
"""

import numpy as np
from typing import Tuple


class WaveletWindowValidator:
    """
    Validate the window size for a wavelet decomposition
    """
    
    @staticmethod
    def validate_window_size(window_size: int, level: int, 
                           wavelet_type: str = 'dwt') -> bool:
        """
        Check that window_size suits the wavelet
        
        Args:
            window_size: window size
            level: decomposition level
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        min_size = WaveletWindowValidator.get_minimum_size(level, wavelet_type)
        
        if window_size < min_size:
            raise ValueError(
                f"Window size ({window_size}) too small for {wavelet_type.upper()} "
                f"at level {level}. Minimum required: {min_size}"
            )
        
        return True
    
    @staticmethod
    def get_minimum_size(level: int, wavelet_type: str = 'dwt') -> int:
        """
        Minimum window_size required
        
        Args:
            level: decomposition level
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            minimum window size
        """
        if wavelet_type == 'dwt':
            # DWT: at least 2^(level+1)
            return 2 ** (level + 1)
        
        elif wavelet_type == 'dtcwt':
            # DTCWT: at least 2^(level+2) because of the Q-shift filters
            return 2 ** (level + 2)
        
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")
    
    @staticmethod
    def get_maximum_level(window_size: int, wavelet_type: str = 'dwt') -> int:
        """
        Maximum level possible for the given window_size
        
        Args:
            window_size: window size
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            maximum level
        """
        if wavelet_type == 'dwt':
            # DWT: max_level = floor(log2(window_size))
            return int(np.log2(window_size))
        
        elif wavelet_type == 'dtcwt':
            # DTCWT: max_level = floor(log2(window_size)) - 1
            return int(np.log2(window_size)) - 1
        
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")
    
    @staticmethod
    def get_optimal_level(window_size: int, wavelet_type: str = 'dwt') -> int:
        """
        Recommended level for window_size
        Conservative: 1-2 levels below the maximum
        
        Args:
            window_size: window size
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            recommended level
        """
        max_level = WaveletWindowValidator.get_maximum_level(window_size, wavelet_type)
        
        if wavelet_type == 'dwt':
            # DWT: one level below the maximum
            optimal = max_level - 1
            # at most 5, at least 2
            optimal = min(optimal, 5)
            optimal = max(optimal, 2)
        
        else:  # dtcwt
            # DTCWT: one level below the maximum
            optimal = max_level - 1
            # at most 4, at least 2
            optimal = min(optimal, 4)
            optimal = max(optimal, 2)
        
        return optimal
    
    @staticmethod
    def recommend_window_size(desired_level: int, wavelet_type: str = 'dwt') -> int:
        """
        Recommended window_size for the desired level
        
        Args:
            desired_level: desired level
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            recommended window size
        """
        min_size = WaveletWindowValidator.get_minimum_size(desired_level, wavelet_type)
        
        # Recommendation: twice the minimum, for a safer margin
        recommended = min_size * 2
        
        return recommended
    
    @staticmethod
    def get_decomposition_info(window_size: int, level: int, 
                              wavelet_type: str = 'dwt') -> dict:
        """
        Full information about the decomposition
        
        Args:
            window_size: window size
            level: decomposition level
            wavelet_type: 'dwt' or 'dtcwt'
        
        Returns:
            dict with the information
        """
        max_level = WaveletWindowValidator.get_maximum_level(window_size, wavelet_type)
        min_size = WaveletWindowValidator.get_minimum_size(level, wavelet_type)
        optimal_level = WaveletWindowValidator.get_optimal_level(window_size, wavelet_type)
        
        # Coefficient size at each level
        coeff_sizes = []
        current_size = window_size
        for i in range(level):
            current_size = (current_size + 1) // 2  # downsampling
            coeff_sizes.append(current_size)
        
        info = {
            'window_size': window_size,
            'requested_level': level,
            'max_possible_level': max_level,
            'optimal_level': optimal_level,
            'min_required_size': min_size,
            'is_valid': window_size >= min_size,
            'coefficient_sizes': coeff_sizes,
            'wavelet_type': wavelet_type,
        }
        
        return info
    
    @staticmethod
    def print_validation_report(window_size: int, level: int, 
                               wavelet_type: str = 'dwt'):
        """
        Print the validation report
        
        Args:
            window_size: window size
            level: decomposition level
            wavelet_type: 'dwt' or 'dtcwt'
        """
        info = WaveletWindowValidator.get_decomposition_info(
            window_size, level, wavelet_type
        )
        
        print("\n" + "="*70)
        print(f"WAVELET VALIDATION REPORT - {wavelet_type.upper()}")
        print("="*70)
        print(f"Window Size:           {info['window_size']}")
        print(f"Requested Level:       {info['requested_level']}")
        print(f"Maximum Possible:      {info['max_possible_level']}")
        print(f"Optimal Level:         {info['optimal_level']}")
        print(f"Minimum Required Size: {info['min_required_size']}")
        print(f"Validation:            {'✓ VALID' if info['is_valid'] else '✗ INVALID'}")
        print("\nCoefficient Sizes per Level:")
        for i, size in enumerate(info['coefficient_sizes'], 1):
            print(f"  Level {i}: {size} coefficients")
        print("="*70 + "\n")
        
        if not info['is_valid']:
            print(f"⚠️  WARNING: Window size too small!")
            print(f"   Recommended minimum: {info['min_required_size']}")
        
        if info['requested_level'] > info['optimal_level']:
            print(f"⚠️  NOTE: Using level {info['optimal_level']} is recommended")
            print(f"   for window size {window_size}")
