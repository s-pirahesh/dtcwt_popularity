# methods/base_method.py
from abc import ABC, abstractmethod
import numpy as np


class BaseMethod(ABC):
    """
    کلاس پایه انتزاعی برای تمام روش‌های ارزیابی محبوبیت
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def assess(self, time_series: np.ndarray) -> float:
        """
        محاسبه امتیاز محبوبیت برای یک سری زمانی مشخص

        Args:
            time_series: آرایه‌ای از تعداد دسترسی‌ها در طول زمان
        Returns:
            امتیاز محبوبیت (عدد اعشاری)
        """
        pass

    def get_name(self) -> str:
        return self.name