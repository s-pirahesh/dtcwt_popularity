"""
Method-specific configurations
تنظیمات خاص هر method برای evaluation

Author: Sajjad
Date: February 2025
"""

from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class MethodConfig:
    """تنظیمات یک method"""
    name: str
    window_days: int       # تعداد روزهای window (باید توان 2 باشد یا 7)
    min_observations: int  # حداقل تعداد مشاهدات لازم
    description: str


# تنظیمات همه methods
METHOD_CONFIGS: Dict[str, MethodConfig] = {
    # =====================================
    # گروه 1: Baselines (7 روز)
    # سریع، responsive، نیاز به data کم
    # =====================================
    'AF': MethodConfig(
        name='AF',
        window_days=7,
        min_observations=3,
        description='Access Frequency - simple baseline'
    ),
    
    'LRU': MethodConfig(
        name='LRU',
        window_days=7,
        min_observations=3,
        description='Least Recently Used'
    ),
    
    'LFU': MethodConfig(
        name='LFU',
        window_days=7,
        min_observations=3,
        description='Least Frequently Used'
    ),
    
    'EWMA': MethodConfig(
        name='EWMA',
        window_days=7,
        min_observations=3,
        description='Exponentially Weighted Moving Average'
    ),
    
    # =====================================
    # گروه 2: Statistical (32 روز = 2^5)
    # نیاز به data متوسط برای skewness/kurtosis
    # =====================================
    'Statistical': MethodConfig(
        name='Statistical',
        window_days=32,
        min_observations=10,
        description='Skewness + Kurtosis based assessment'
    ),
    
    # =====================================
    # گروه 3: Wavelet (64 روز = 2^6)
    # نیاز به signal بلند برای decomposition مناسب
    # =====================================
    'DWT+AF': MethodConfig(
        name='DWT+AF',
        window_days=64,
        min_observations=32,
        description='Discrete Wavelet Transform + AF formula'
    ),
    
    'DTCWT+AF': MethodConfig(
        name='DTCWT+AF',
        window_days=64,
        min_observations=32,
        description='Dual-Tree Complex Wavelet Transform + AF'
    ),
    
    # =====================================
    # گروه 4: Hybrid (64 روز = 2^6)
    # ترکیبی از wavelet و statistical
    # =====================================
    'Hybrid V3.0': MethodConfig(
        name='Hybrid V3.0',
        window_days=64,
        min_observations=32,
        description='DTCWT + Statistical (V3.0)'
    ),
    
    'Hybrid V3.1': MethodConfig(
        name='Hybrid V3.1',
        window_days=64,
        min_observations=32,
        description='DTCWT + Statistical + Advanced Features (V3.1)'
    ),
}


def get_method_config(method_name: str) -> MethodConfig:
    """
    دریافت config یک method
    
    Args:
        method_name: نام method
        
    Returns:
        MethodConfig
        
    Raises:
        KeyError: اگر method وجود نداشته باشد
    """
    if method_name not in METHOD_CONFIGS:
        raise KeyError(f"No config found for method: {method_name}")
    
    return METHOD_CONFIGS[method_name]


def get_window_size(method_name: str, default: int = 30) -> int:
    """
    دریافت window size یک method
    
    Args:
        method_name: نام method
        default: مقدار پیش‌فرض اگر method وجود نداشت
        
    Returns:
        تعداد روزهای window
    """
    try:
        return get_method_config(method_name).window_days
    except KeyError:
        return default


def get_min_observations(method_name: str, default: int = 10) -> int:
    """
    دریافت حداقل تعداد مشاهدات لازم
    
    Args:
        method_name: نام method
        default: مقدار پیش‌فرض
        
    Returns:
        حداقل تعداد مشاهدات
    """
    try:
        return get_method_config(method_name).min_observations
    except KeyError:
        return default


def list_methods_by_window_size() -> Dict[int, list]:
    """
    دسته‌بندی methods بر اساس window size
    
    Returns:
        Dict: {window_size: [method_names]}
    """
    grouped = {}
    
    for name, config in METHOD_CONFIGS.items():
        window = config.window_days
        if window not in grouped:
            grouped[window] = []
        grouped[window].append(name)
    
    return grouped


def validate_configs():
    """
    اعتبارسنجی configs
    بررسی می‌کند که همه window_days توان 2 یا 7 باشند
    """
    errors = []
    
    # مقادیر مجاز: 7 و توان‌های 2
    allowed_values = [7] + [2**i for i in range(3, 10)]  # 7, 8, 16, 32, 64, 128, 256, 512
    
    for name, config in METHOD_CONFIGS.items():
        # بررسی window_days
        if config.window_days not in allowed_values:
            errors.append(
                f"{name}: window_days={config.window_days} "
                f"باید 7 یا توان 2 باشد (مثلاً 8, 16, 32, 64, 128)"
            )
        
        # بررسی min_observations
        if config.min_observations < 1:
            errors.append(
                f"{name}: min_observations={config.min_observations} "
                f"باید >= 1 باشد"
            )
        
        # بررسی منطقی: min_observations نباید بیشتر از window_days باشد
        if config.min_observations > config.window_days:
            errors.append(
                f"{name}: min_observations={config.min_observations} "
                f"بیشتر از window_days={config.window_days}!"
            )
    
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))
    
    # چاپ آمار
    print("=" * 70)
    print("METHOD CONFIGS VALIDATION")
    print("=" * 70)
    
    grouped = list_methods_by_window_size()
    for window_size in sorted(grouped.keys()):
        methods = grouped[window_size]
        print(f"\nWindow {window_size} days ({len(methods)} methods):")
        for method in methods:
            config = METHOD_CONFIGS[method]
            print(f"  • {method:<20} min_obs={config.min_observations:>3}")
    
    print("\n" + "=" * 70)
    print(f"✓ All {len(METHOD_CONFIGS)} method configs validated successfully")
    print("=" * 70 + "\n")


# اجرای validation هنگام import
validate_configs()


if __name__ == '__main__':
    # تست
    print("\nTesting method_configs...\n")
    
    # تست get_window_size
    print("Window sizes:")
    print(f"  AF: {get_window_size('AF')} days")
    print(f"  Statistical: {get_window_size('Statistical')} days")
    print(f"  DWT+AF: {get_window_size('DWT+AF')} days")
    print(f"  DTCWT+AF: {get_window_size('DTCWT+AF')} days")
    
    # تست get_method_config
    print("\nMethod config for DWT+AF:")
    config = get_method_config('DWT+AF')
    print(f"  Name: {config.name}")
    print(f"  Window: {config.window_days} days")
    print(f"  Min obs: {config.min_observations}")
    print(f"  Description: {config.description}")
    
    print("\n✓ Tests passed!")
