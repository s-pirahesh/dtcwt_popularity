"""
Generate Sample Datasets for Testing
Creates synthetic datasets with realistic patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_youtube_like_data(n_videos=500, n_days=60, output_path=None):
    """
    Generate YouTube-like viewing dataset
    
    Args:
        n_videos: Number of videos
        n_days: Number of days
        output_path: Path to save CSV
        
    Returns:
        DataFrame with generated data
    """
    print(f"Generating YouTube-like dataset...")
    print(f"  Videos: {n_videos}, Days: {n_days}")
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    
    # Video categories with different patterns
    categories = {
        'viral': 0.05,      # 5% viral videos
        'trending': 0.15,   # 15% trending
        'stable': 0.50,     # 50% stable
        'declining': 0.20,  # 20% declining
        'random': 0.10,     # 10% random
    }
    
    video_id = 0
    
    for category, ratio in categories.items():
        n_category = int(n_videos * ratio)
        
        for _ in range(n_category):
            video_id += 1
            
            if category == 'viral':
                # Viral: sudden spike then decline
                base_views = np.random.randint(50, 200)
                spike_day = np.random.randint(5, 20)
                spike_magnitude = np.random.uniform(5, 15)
                
                for i, date in enumerate(dates):
                    if i < spike_day:
                        views = base_views * (1.1 ** i)
                    elif i == spike_day:
                        views = base_views * spike_magnitude
                    else:
                        decay = 0.85 ** (i - spike_day)
                        views = base_views * spike_magnitude * decay
                    
                    views = int(views + np.random.normal(0, views * 0.1))
                    
                    data.append({
                        'timestamp': date,
                        'video_id': f'video_{video_id:04d}',
                        'view_count': max(0, views),
                        'category': category
                    })
            
            elif category == 'trending':
                # Trending: steady growth
                base_views = np.random.randint(100, 500)
                growth_rate = np.random.uniform(1.02, 1.08)
                
                for i, date in enumerate(dates):
                    views = base_views * (growth_rate ** i)
                    views = int(views + np.random.normal(0, views * 0.05))
                    
                    data.append({
                        'timestamp': date,
                        'video_id': f'video_{video_id:04d}',
                        'view_count': max(0, views),
                        'category': category
                    })
            
            elif category == 'stable':
                # Stable: constant with small variations
                avg_views = np.random.randint(200, 1000)
                
                for i, date in enumerate(dates):
                    views = avg_views + np.random.normal(0, avg_views * 0.1)
                    
                    data.append({
                        'timestamp': date,
                        'video_id': f'video_{video_id:04d}',
                        'view_count': max(0, int(views)),
                        'category': category
                    })
            
            elif category == 'declining':
                # Declining: started high, decreasing
                initial_views = np.random.randint(1000, 3000)
                decay_rate = np.random.uniform(0.92, 0.98)
                
                for i, date in enumerate(dates):
                    views = initial_views * (decay_rate ** i)
                    views = int(views + np.random.normal(0, views * 0.05))
                    
                    data.append({
                        'timestamp': date,
                        'video_id': f'video_{video_id:04d}',
                        'view_count': max(0, views),
                        'category': category
                    })
            
            else:  # random
                # Random: no clear pattern
                for i, date in enumerate(dates):
                    views = np.random.exponential(300)
                    
                    data.append({
                        'timestamp': date,
                        'video_id': f'video_{video_id:04d}',
                        'view_count': max(0, int(views)),
                        'category': category
                    })
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print(f"  Generated {len(df)} records")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    
    return df


def generate_movielens_like_data(n_movies=300, n_days=90, output_path=None):
    """
    Generate MovieLens-like rating dataset
    
    Args:
        n_movies: Number of movies
        n_days: Number of days
        output_path: Path to save CSV
        
    Returns:
        DataFrame with generated data
    """
    print(f"\nGenerating MovieLens-like dataset...")
    print(f"  Movies: {n_movies}, Days: {n_days}")
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    
    for movie_id in range(1, n_movies + 1):
        # Movie characteristics
        popularity = np.random.choice(['blockbuster', 'popular', 'niche', 'cult'])
        release_day = np.random.randint(0, min(30, n_days))
        
        for i, date in enumerate(dates):
            if i < release_day:
                ratings = 0
            else:
                days_since_release = i - release_day
                
                if popularity == 'blockbuster':
                    # High initial ratings, slowly declining
                    base = 200
                    ratings = base * (0.95 ** days_since_release)
                elif popularity == 'popular':
                    # Steady ratings
                    ratings = np.random.randint(50, 150)
                elif popularity == 'niche':
                    # Low but steady
                    ratings = np.random.randint(10, 40)
                else:  # cult
                    # Slow growth over time
                    ratings = 20 + days_since_release * 2
                
                ratings = int(ratings + np.random.normal(0, ratings * 0.2))
            
            if ratings > 0:
                data.append({
                    'timestamp': date,
                    'movie_id': f'movie_{movie_id:04d}',
                    'rating_count': max(0, ratings),
                    'popularity': popularity
                })
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print(f"  Generated {len(df)} records")
    
    return df


def main():
    """Generate all sample datasets"""
    
    print("="*60)
    print("  SAMPLE DATA GENERATOR")
    print("="*60)
    
    # Create datasets directory
    data_dir = Path('data/datasets')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    youtube_df = generate_youtube_like_data(
        n_videos=500,
        n_days=60,
        output_path=data_dir / 'sample_youtube.csv'
    )
    
    movielens_df = generate_movielens_like_data(
        n_movies=300,
        n_days=90,
        output_path=data_dir / 'sample_movielens.csv'
    )
    
    print("\n" + "="*60)
    print("SAMPLE DATASETS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  1. data/datasets/sample_youtube.csv")
    print(f"  2. data/datasets/sample_movielens.csv")
    print("\nYou can now run experiments with these datasets!")
    print("\nExample:")
    print("  python experiments/exp1_assessment_comparison.py")
    print("="*60)


if __name__ == '__main__':
    main()
