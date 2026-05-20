# -*- coding: utf-8 -*-
"""
Time Slot Utilities
Generic time handling برای datasets با granularity مختلف

Author: Sajjad
Date: February 2025
"""

from datetime import timedelta, datetime
from typing import Optional
import pandas as pd


class TimeSlotHelper:
    """
    Helper برای کار با time slots generic
    
    هر dataset granularity خودش را دارد:
      - MovieLens daily: 1 slot = 1 day
      - MovieLens weekly: 1 slot = 1 week  
      - Youku: 1 slot = 5 minutes
      - NYC Yellow Taxi: 1 slot = 15 minutes
    """
    
    def __init__(self, time_granularity: str = 'daily', 
                 slot_duration_minutes: Optional[int] = None):
        """
        Args:
            time_granularity: 'daily', 'hourly', 'minute', 'weekly', 'custom'
            slot_duration_minutes: برای 'custom' مثلاً 5 برای 5-minute slots
        """
        self.granularity = time_granularity
        self.custom_minutes = slot_duration_minutes
        
        # محاسبه یک slot duration
        self._slot_delta = self._calculate_slot_duration()
    
    def _calculate_slot_duration(self) -> timedelta:
        """محاسبه مدت زمان یک slot"""
        if self.granularity == 'daily':
            return timedelta(days=1)
        elif self.granularity == 'hourly':
            return timedelta(hours=1)
        elif self.granularity == 'minute':
            return timedelta(minutes=1)
        elif self.granularity == 'weekly':
            return timedelta(days=7)
        elif self.granularity == 'monthly':
            return timedelta(days=30)  # تقریبی
        elif self.granularity == 'custom':
            if self.custom_minutes is None:
                raise ValueError("slot_duration_minutes must be set for custom granularity")
            return timedelta(minutes=self.custom_minutes)
        else:
            raise ValueError(f"Unknown time_granularity: {self.granularity}")
    
    def slots_to_timedelta(self, n_slots: int) -> timedelta:
        """
        تبدیل تعداد slots به timedelta
        
        Args:
            n_slots: تعداد slots
            
        Returns:
            timedelta object
            
        Example:
            daily: 7 slots → 7 days
            5min: 12 slots → 60 minutes = 1 hour
        """
        return self._slot_delta * n_slots
    
    def days_to_slots(self, days: int) -> int:
        """
        Convert a window size expressed in DAYS to the equivalent number
        of time-slots for this dataset's granularity.

        Examples:
          daily  granularity: 7 days → 7 slots
          hourly granularity: 7 days → 168 slots
          5-min  granularity: 7 days → 2016 slots

        Args:
            days: window size in calendar days (from MethodConfig.window_days)

        Returns:
            int: equivalent number of time-slots
        """
        day_seconds  = 24 * 3600
        slot_seconds = self._slot_delta.total_seconds()
        return max(1, int(days * day_seconds / slot_seconds))

    def count_slots(self, start: datetime, end: datetime) -> int:
        """
        محاسبه تعداد slots بین دو تاریخ
        
        Args:
            start: تاریخ شروع
            end: تاریخ پایان
            
        Returns:
            تعداد slots
            
        Example:
            daily: 2023-01-01 to 2023-01-08 → 7 slots
            5min: 10:00 to 11:00 → 12 slots
        """
        total_seconds = (end - start).total_seconds()
        slot_seconds = self._slot_delta.total_seconds()
        
        return int(total_seconds / slot_seconds)
    
    def add_slots(self, dt: datetime, n_slots: int) -> datetime:
        """
        اضافه کردن n slots به تاریخ
        
        Args:
            dt: تاریخ
            n_slots: تعداد slots برای اضافه کردن
            
        Returns:
            تاریخ جدید
        """
        return dt + self.slots_to_timedelta(n_slots)
    
    def get_unit_name(self, plural: bool = False) -> str:
        """
        دریافت نام واحد زمانی برای نمایش
        
        Args:
            plural: آیا جمع باشد؟
            
        Returns:
            نام واحد
            
        Example:
            daily: "day" or "days"
            5min: "5-minute slot" or "5-minute slots"
        """
        if self.granularity == 'daily':
            return 'days' if plural else 'day'
        elif self.granularity == 'hourly':
            return 'hours' if plural else 'hour'
        elif self.granularity == 'minute':
            return 'minutes' if plural else 'minute'
        elif self.granularity == 'weekly':
            return 'weeks' if plural else 'week'
        elif self.granularity == 'monthly':
            return 'months' if plural else 'month'
        elif self.granularity == 'custom':
            unit = f"{self.custom_minutes}-minute slot"
            return unit + 's' if plural else unit
        else:
            return 'slots' if plural else 'slot'
    
    def format_window_size(self, n_slots: int) -> str:
        """
        فرمت کردن window size برای نمایش
        
        Args:
            n_slots: تعداد slots
            
        Returns:
            رشته فرمت شده
            
        Example:
            daily, 30 → "30 days"
            5min, 288 → "288 slots (24 hours)"
        """
        unit = self.get_unit_name(plural=(n_slots > 1))
        base = f"{n_slots} {unit}"
        
        # اضافه کردن توضیح اگر custom
        if self.granularity == 'custom':
            total_hours = (n_slots * self.custom_minutes) / 60
            if total_hours >= 24:
                days = total_hours / 24
                base += f" ({days:.1f} days)"
            else:
                base += f" ({total_hours:.1f} hours)"
        
        return base


def create_time_helper(config) -> TimeSlotHelper:
    """
    ساخت TimeSlotHelper از config
    
    Args:
        config: EvaluationConfig
        
    Returns:
        TimeSlotHelper instance
    """
    return TimeSlotHelper(
        time_granularity=config.time_granularity,
        slot_duration_minutes=config.slot_duration_minutes
    )
