# -*- coding: utf-8 -*-
"""
Results Visualizer - Graphical Analysis
Create plots and visualizations from saved results
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# تنظیمات matplotlib برای Unicode
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ResultsVisualizer:
    """
    رسم نمودارها از نتایج ذخیره شده
    
    Features:
    - Temporal evolution plots
    - Method comparison plots  
    - Stratum comparison plots
    - Ranking evolution heatmaps
    - Cold-start analysis plots
    """
    
    def __init__(self, analyzer, output_dir: Optional[Path] = None):
        """
        Args:
            analyzer: ResultsAnalyzer instance
            output_dir: دایرکتوری ذخیره نمودارها (None = run_dir/visualization)
        """
        self.analyzer = analyzer
        
        if output_dir is None:
            self.output_dir = analyzer.run_dir / 'visualization'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # تنظیمات نمودار
        self.figsize = (12, 6)
        self.dpi = 300
        self.style = 'seaborn-v0_8-darkgrid'
        
        # رنگ‌ها
        self.colors = sns.color_palette("husl", n_colors=10)
    
    def plot_temporal_evolution(self, 
                                methods: List[str],
                                metric: str = 'mae',
                                save: bool = True,
                                show: bool = False) -> plt.Figure:
        """
        رسم تکامل زمانی یک معیار برای چند روش
        
        Args:
            methods: لیست نام روش‌ها
            metric: نام معیار
            save: ذخیره نمودار
            show: نمایش نمودار
        
        Returns:
            matplotlib Figure
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, method in enumerate(methods):
            try:
                evolution = self.analyzer.get_temporal_evolution(method, metric)
                ax.plot(evolution['date'], evolution[metric], 
                       label=method, linewidth=2, color=self.colors[i])
            except Exception as e:
                print(f"Warning: Could not plot {method}: {e}")
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Temporal Evolution - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'temporal_evolution_{metric}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_method_comparison(self,
                              filter_top_percent: Optional[float] = None,
                              filter_stratum: Optional[str] = None,
                              metrics: List[str] = None,
                              save: bool = True,
                              show: bool = False) -> plt.Figure:
        """
        رسم مقایسه روش‌ها
        
        Args:
            filter_top_percent: فیلتر top-k%
            filter_stratum: فیلتر stratum
            metrics: لیست معیارها (None = همه)
            save: ذخیره
            show: نمایش
        
        Returns:
            matplotlib Figure
        """
        # مقایسه روش‌ها
        comparison = self.analyzer.compare_methods(
            filter_top_percent=filter_top_percent,
            filter_stratum=filter_stratum
        )
        
        if len(comparison) == 0:
            print("No data to plot")
            return None
        
        # انتخاب معیارها
        if metrics is None:
            metrics = ['spearman', 'mae', 'rmse', 'ndcg']
        
        # رسم
        plt.style.use(self.style)
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric not in comparison.columns:
                continue
            
            # مرتب‌سازی
            if metric in ['spearman', 'kendall', 'ndcg', 'coverage']:
                # بالاتر بهتر
                data = comparison.sort_values(metric, ascending=False)
            else:
                # پایین‌تر بهتر
                data = comparison.sort_values(metric, ascending=True)
            
            # رسم
            bars = ax.barh(data['method'], data[metric], color=self.colors[:len(data)])
            
            # برچسب‌ها
            ax.set_xlabel(metric.upper(), fontsize=11)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            # مقادیر روی میله‌ها
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}',
                       ha='left', va='center', fontsize=9)
        
        # عنوان کلی
        title_parts = ['Method Comparison']
        if filter_top_percent:
            title_parts.append(f'(Top {filter_top_percent}%)')
        if filter_stratum:
            title_parts.append(f'({filter_stratum})')
        
        fig.suptitle(' '.join(title_parts), fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            filename = 'method_comparison'
            if filter_top_percent:
                filename += f'_top{int(filter_top_percent)}'
            if filter_stratum:
                filename += f'_{filter_stratum}'
            filepath = self.output_dir / f'{filename}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_stratum_comparison(self,
                               methods: List[str],
                               metric: str = 'spearman_corr',
                               save: bool = True,
                               show: bool = False) -> plt.Figure:
        """
        رسم مقایسه عملکرد در strata مختلف
        
        Args:
            methods: لیست نام روش‌ها
            metric: نام معیار
            save: ذخیره
            show: نمایش
        
        Returns:
            matplotlib Figure
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        strata_names = ['cold_start', 'low', 'medium', 'high']
        x = np.arange(len(strata_names))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            try:
                stratum_comp = self.analyzer.get_stratum_comparison(method, metric)
                
                # مرتب‌سازی بر اساس ترتیب strata
                stratum_comp['order'] = stratum_comp['stratum_name'].map({
                    'cold_start': 0, 'low': 1, 'medium': 2, 'high': 3
                })
                stratum_comp = stratum_comp.sort_values('order')
                
                # ایجاد array کامل با NaN برای strata های خالی
                values = np.full(len(strata_names), np.nan)
                errors = np.full(len(strata_names), np.nan)
                
                # پر کردن مقادیر موجود
                for _, row in stratum_comp.iterrows():
                    stratum_idx = int(row['order'])
                    if stratum_idx < len(strata_names):
                        values[stratum_idx] = row['mean']
                        errors[stratum_idx] = row['std']
                
                # رسم فقط strata های غیر-NaN
                valid_mask = ~np.isnan(values)
                if valid_mask.any():
                    ax.bar(x[valid_mask] + i * width, values[valid_mask], width, 
                          label=method, yerr=errors[valid_mask], capsize=3,
                          color=self.colors[i], alpha=0.8)
            
            except Exception as e:
                print(f"Warning: Could not plot {method}: {e}")
        
        ax.set_xlabel('Stratum', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Performance by Stratum - {metric.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(strata_names)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'stratum_comparison_{metric}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_ranking_evolution_heatmap(self,
                                      method_name: str,
                                      top_n_items: int = 50,
                                      save: bool = True,
                                      show: bool = False) -> plt.Figure:
        """
        رسم heatmap تکامل ranking
        
        Args:
            method_name: نام روش
            top_n_items: تعداد آیتم‌های بالا
            save: ذخیره
            show: نمایش
        
        Returns:
            matplotlib Figure
        """
        df = self.analyzer.load_detailed_scores(method_name)
        
        # انتخاب top-N items بر اساس میانگین actual_count
        top_items = df.groupby('item_id')['actual_count'].mean().nlargest(top_n_items).index
        df = df[df['item_id'].isin(top_items)]
        
        # ایجاد pivot table (item × window)
        pivot = df.pivot_table(
            index='item_id',
            columns='window_id',
            values='rank_predicted',
            aggfunc='first'
        )
        
        # رسم
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(min(20, len(pivot.columns)/2), min(15, len(pivot)/3)))
        
        sns.heatmap(pivot, cmap='RdYlGn_r', cbar_kws={'label': 'Predicted Rank'},
                   ax=ax, linewidths=0)
        
        ax.set_xlabel('Window ID', fontsize=12)
        ax.set_ylabel('Item ID', fontsize=12)
        ax.set_title(f'Ranking Evolution - {method_name}\n(Top {top_n_items} Items)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'ranking_heatmap_{method_name}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_summary_report(self,
                            filter_top_percent: Optional[float] = None,
                            filter_stratum: Optional[str] = None,
                            save: bool = True,
                            show: bool = False):
        """
        ایجاد گزارش خلاصه با چند نمودار
        
        Args:
            filter_top_percent: فیلتر top-k%
            filter_stratum: فیلتر stratum
            save: ذخیره
            show: نمایش
        """
        methods = self.analyzer.available_methods
        
        if len(methods) == 0:
            print("No methods available")
            return
        
        print("\nGenerating summary report...")
        
        # 1. Temporal evolution (MAE)
        print("  1. Temporal evolution (MAE)...")
        self.plot_temporal_evolution(methods, metric='mae', save=save, show=show)
        
        # 2. Method comparison
        print("  2. Method comparison...")
        self.plot_method_comparison(
            filter_top_percent=filter_top_percent,
            filter_stratum=filter_stratum,
            save=save,
            show=show
        )
        
        # 3. Stratum comparison (برای اولین روش)
        print("  3. Stratum comparison...")
        self.plot_stratum_comparison(methods[:3], metric='spearman_corr', save=save, show=show)
        
        print("\n✓ Summary report created successfully!")
        print(f"  Output directory: {self.output_dir}")