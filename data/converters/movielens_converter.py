"""
MovieLens Dataset Converter
تبدیل فایل ratings.csv به فرمت استاندارد

Dataset: MovieLens (ml-32m)
Source: https://grouplens.org/datasets/movielens/
Size: 32M ratings, 87K movies, 200K users
Period: 1995-2023

Structure:
    Input: ratings.csv
        - userId,movieId,rating,timestamp
        - timestamp: Unix timestamp (seconds since 1970)
        - rating: 0.5 to 5.0 stars
    
    Output: Standard CSV
        - timestamp: datetime
        - item_id: movieId
        - count: 1 per rating (or aggregated)
        - rating: original rating value (optional)
        - user_id: userId (optional)
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from . import BaseConverter, ConverterFactory


class MovieLensConverter(BaseConverter):
    """
    Converter برای دیتاست MovieLens
    
    تبدیل ratings.csv به فرمت استاندارد برای تحلیل محبوبیت
    
    گزینه‌های تبدیل:
        - keep_rating: نگهداری ستون rating
        - keep_user: نگهداری ستون userId
        - aggregate_by: تجمیع بر اساس ('hour', 'day', 'week')
        - min_rating: حداقل rating برای فیلتر (مثلاً >= 3.0)
    """
    
    def __init__(self, 
                 keep_rating: bool = True,
                 keep_user: bool = False,
                 aggregate_by: Optional[str] = None,
                 min_rating: Optional[float] = None,
                 **kwargs):
        """
        مقداردهی اولیه
        
        Args:
            keep_rating: نگهداری ستون rating در خروجی
            keep_user: نگهداری ستون userId در خروجی
            aggregate_by: تجمیع زمانی ('hour', 'day', 'week', None)
            min_rating: حداقل rating برای فیلتر (None = همه)
            **kwargs: پارامترهای BaseConverter
        """
        super().__init__(**kwargs)
        self.keep_rating = keep_rating
        self.keep_user = keep_user
        self.aggregate_by = aggregate_by
        self.min_rating = min_rating
    
    def _convert_single_file(self, 
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        تبدیل فایل ratings.csv
        
        Args:
            file_path: مسیر ratings.csv
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame استاندارد
        """
        self.log(f"خواندن فایل MovieLens: {file_path.name}")
        
        # ==========================================
        # مرحله 1: خواندن CSV
        # ==========================================
        df = pd.read_csv(
            file_path,
            # ستون‌های مورد نیاز
            usecols=['userId', 'movieId', 'rating', 'timestamp'],
            # نوع داده‌ها برای سرعت بیشتر
            dtype={
                'userId': 'int32',
                'movieId': 'int32',
                'rating': 'float32',
                'timestamp': 'int64'
            }
        )
        
        self.log(f"  ✓ {len(df):,} رکورد خوانده شد")
        
        # ==========================================
        # مرحله 2: فیلتر rating (اختیاری)
        # ==========================================
        if self.min_rating is not None:
            before_len = len(df)
            df = df[df['rating'] >= self.min_rating]
            self.log(f"  ✓ فیلتر rating >= {self.min_rating}: "
                    f"{len(df):,} رکورد ({len(df)/before_len*100:.1f}%)")
        
        # ==========================================
        # مرحله 3: تبدیل timestamp
        # ==========================================
        self.log("  تبدیل Unix timestamp به datetime...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # ==========================================
        # مرحله 4: ساخت ستون‌های استاندارد
        # ==========================================
        # item_id = movieId
        df['item_id'] = df['movieId'].astype(str)
        
        # count = 1 (هر rating یک دسترسی)
        df['count'] = 1
        
        # ==========================================
        # مرحله 5: تجمیع زمانی (اختیاری)
        # ==========================================
        if self.aggregate_by:
            df = self._aggregate_temporal(df)
        
        # ==========================================
        # مرحله 6: انتخاب ستون‌های خروجی
        # ==========================================
        output_columns = ['timestamp', 'item_id', 'count']
        
        # اضافه کردن ستون‌های اختیاری
        if self.keep_rating:
            output_columns.append('rating')
        
        if self.keep_user:
            df['user_id'] = df['userId'].astype(str)
            output_columns.append('user_id')
        
        df_output = df[output_columns].copy()
        
        # ==========================================
        # مرحله 7: مرتب‌سازی نهایی
        # ==========================================
        df_output = df_output.sort_values('timestamp').reset_index(drop=True)
        
        self.log(f"  ✓ تبدیل کامل: {len(df_output):,} رکورد")
        
        return df_output
    
    def _aggregate_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تجمیع بر اساس بازه زمانی
        
        مثال: تجمیع rating های یک فیلم در یک ساعت/روز/هفته
        
        Args:
            df: DataFrame با timestamp دقیق
            
        Returns:
            DataFrame تجمیع شده
        """
        self.log(f"  تجمیع بر اساس: {self.aggregate_by}")
        
        # گرد کردن timestamp
        freq_map = {
            'hour': 'H',
            'day': 'D',
            'week': 'W'
        }
        freq = freq_map.get(self.aggregate_by, 'D')
        
        df['time_bucket'] = df['timestamp'].dt.floor(freq)
        
        # تجمیع
        agg_dict = {
            'count': 'sum',  # مجموع count ها
        }
        
        # اگر rating را نگه می‌داریم، میانگین بگیریم
        if self.keep_rating and 'rating' in df.columns:
            agg_dict['rating'] = 'mean'
        
        # اگر user را نگه می‌داریم، اولین یکی را بگیریم
        if self.keep_user and 'userId' in df.columns:
            # تعداد کاربران یکتا در این بازه
            df['user_count'] = df.groupby(['time_bucket', 'item_id'])['userId'].transform('nunique')
            agg_dict['user_count'] = 'first'
        
        df_agg = df.groupby(['time_bucket', 'item_id'], as_index=False).agg(agg_dict)
        
        # تغییر نام time_bucket به timestamp
        df_agg.rename(columns={'time_bucket': 'timestamp'}, inplace=True)
        
        self.log(f"  ✓ از {len(df):,} به {len(df_agg):,} رکورد تجمیع شد")
        
        return df_agg
    
    def get_statistics_extended(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        آمار تکمیلی برای MovieLens
        
        Args:
            df: DataFrame تبدیل شده
            
        Returns:
            دیکشنری آمار
        """
        stats = {
            'total_ratings': len(df),
            'unique_movies': df['item_id'].nunique(),
            'time_span': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        }
        
        if 'rating' in df.columns:
            stats['avg_rating'] = df['rating'].mean()
            stats['rating_distribution'] = df['rating'].value_counts().to_dict()
        
        if 'user_id' in df.columns:
            stats['unique_users'] = df['user_id'].nunique()
            stats['avg_ratings_per_user'] = len(df) / df['user_id'].nunique()
            stats['avg_ratings_per_movie'] = len(df) / df['item_id'].nunique()
        
        return stats


# ==========================================
# ثبت در Factory
# ==========================================
ConverterFactory.register('movielens', MovieLensConverter)


# ==========================================
# استفاده نمونه
# ==========================================
if __name__ == '__main__':
    """
    مثال‌های استفاده:
    
    # حالت پایه: فقط timestamp, item_id, count
    converter = MovieLensConverter()
    df = converter.convert(
        input_path='data/raw/movielens/ratings.csv',
        output_path='data/datasets/movielens_basic.csv'
    )
    
    # با نگهداری rating
    converter = MovieLensConverter(keep_rating=True)
    df = converter.convert(
        input_path='data/raw/movielens/ratings.csv',
        output_path='data/datasets/movielens_with_rating.csv'
    )
    
    # با تجمیع روزانه
    converter = MovieLensConverter(
        aggregate_by='day',
        keep_rating=True
    )
    df = converter.convert(
        input_path='data/raw/movielens/ratings.csv',
        output_path='data/datasets/movielens_daily.csv'
    )
    
    # فقط rating های بالا (>= 4.0)
    converter = MovieLensConverter(
        min_rating=4.0,
        keep_rating=True
    )
    df = converter.convert(
        input_path='data/raw/movielens/ratings.csv',
        output_path='data/datasets/movielens_high_rated.csv'
    )
    
    # کامل با همه گزینه‌ها
    converter = MovieLensConverter(
        keep_rating=True,
        keep_user=True,
        aggregate_by='hour',
        min_rating=3.0,
        verbose=True
    )
    df = converter.convert(
        input_path='data/raw/movielens/ratings.csv',
        output_path='data/datasets/movielens_complete.csv'
    )
    """
    
    # تست سریع
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'movielens_converted.csv'
        
        converter = MovieLensConverter(keep_rating=True, verbose=True)
        df = converter.convert(input_file, output_file)
        
        print("\nنمونه خروجی:")
        print(df.head(10))
        print("\nآمار:")
        stats = converter.get_statistics_extended(df)
        for key, value in stats.items():
            print(f"  {key}: {value}")
