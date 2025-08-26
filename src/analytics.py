#!/usr/bin/env python3
"""
Analytics Script for Multilingual App Reviews Analytics

This script loads the processed Parquet dataset, computes key performance indicators (KPIs),
and generates insightful visualizations saved to the reports/figures directory.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import seaborn as sns
from typing import Optional, Dict, List

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_paths():
    """Setup and validate data directory paths."""
    project_root = Path(__file__).parent.parent
    processed_data_path = project_root / "data" / "processed"
    figures_path = project_root / "reports" / "figures"
    
    # Create figures directory if it doesn't exist
    figures_path.mkdir(parents=True, exist_ok=True)
    
    return processed_data_path, figures_path

def load_processed_data(processed_data_path: Path, filename: str = "cleaned_reviews.parquet") -> pd.DataFrame:
    """Load the processed dataset from Parquet file."""
    file_path = processed_data_path / filename
    
    if not file_path.exists():
        # Try CSV fallback
        csv_path = processed_data_path / filename.replace('.parquet', '.csv')
        if csv_path.exists():
            print(f"Parquet file not found, loading CSV from {csv_path}")
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Processed dataset not found at {file_path} or {csv_path}. "
                f"Please run the data preprocessing script first: python src/data_prep.py"
            )
    
    print(f"Loading processed dataset from {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def compute_weekly_average_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly average ratings over time."""
    # Find date and rating columns
    date_col = None
    rating_col = None
    
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
        if 'rating' in col.lower() or 'score' in col.lower():
            rating_col = col
    
    if date_col is None or rating_col is None:
        print("Warning: Could not find date or rating columns for weekly analysis")
        return pd.DataFrame()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Remove rows with invalid dates or ratings
    valid_data = df.dropna(subset=[date_col, rating_col])
    
    if len(valid_data) == 0:
        print("Warning: No valid date/rating data found")
        return pd.DataFrame()
    
    # Group by week and compute average rating
    valid_data['week'] = valid_data[date_col].dt.to_period('W')
    weekly_ratings = valid_data.groupby('week')[rating_col].agg(['mean', 'count', 'std']).reset_index()
    weekly_ratings['week'] = weekly_ratings['week'].dt.start_time
    
    print(f"Computed weekly ratings for {len(weekly_ratings)} weeks")
    return weekly_ratings

def compute_language_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of reviews by language."""
    # Find language column
    language_col = None
    for col in df.columns:
        if 'lang' in col.lower():
            language_col = col
            break
    
    if language_col is None:
        print("Warning: No language column found")
        return pd.DataFrame()
    
    lang_dist = df[language_col].value_counts().reset_index()
    lang_dist.columns = ['language', 'count']
    lang_dist['percentage'] = (lang_dist['count'] / len(df)) * 100
    
    print(f"Found {len(lang_dist)} unique languages")
    return lang_dist

def compute_app_ratings_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rating summary statistics by app."""
    # Find app and rating columns
    app_col = None
    rating_col = None
    
    for col in df.columns:
        if 'app' in col.lower() and 'name' in col.lower():
            app_col = col
        elif 'app' in col.lower():
            app_col = col
        if 'rating' in col.lower() or 'score' in col.lower():
            rating_col = col
    
    if app_col is None or rating_col is None:
        print("Warning: Could not find app or rating columns")
        return pd.DataFrame()
    
    app_ratings = df.groupby(app_col)[rating_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    app_ratings.columns = [app_col, 'review_count', 'avg_rating', 'median_rating', 
                          'rating_std', 'min_rating', 'max_rating']
    app_ratings = app_ratings.sort_values('avg_rating', ascending=False)
    
    print(f"Computed ratings for {len(app_ratings)} apps")
    return app_ratings

def plot_weekly_ratings_trend(weekly_ratings: pd.DataFrame, figures_path: Path):
    """Generate and save weekly ratings trend plot."""
    if weekly_ratings.empty:
        print("Skipping weekly ratings plot - no data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot average rating with error bars
    ax.errorbar(weekly_ratings['week'], weekly_ratings['mean'], 
               yerr=weekly_ratings['std'], marker='o', linewidth=2, 
               capsize=5, capthick=2, label='Weekly Average Rating')
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Average Rating')
    ax.set_title('Weekly Average Rating Trend', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = figures_path / 'weekly_ratings_trend.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved weekly ratings trend plot to {output_path}")

def plot_language_distribution(lang_dist: pd.DataFrame, figures_path: Path):
    """Generate and save language distribution plot."""
    if lang_dist.empty:
        print("Skipping language distribution plot - no data available")
        return
    
    # Plot top 15 languages
    top_languages = lang_dist.head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    bars = ax1.bar(range(len(top_languages)), top_languages['count'])
    ax1.set_xlabel('Language')
    ax1.set_ylabel('Number of Reviews')
    ax1.set_title('Top 15 Languages by Review Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(top_languages)))
    ax1.set_xticklabels(top_languages['language'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Pie chart for top 10
    top_10 = lang_dist.head(10)
    others_count = lang_dist.iloc[10:]['count'].sum()
    
    if others_count > 0:
        pie_data = list(top_10['count']) + [others_count]
        pie_labels = list(top_10['language']) + ['Others']
    else:
        pie_data = top_10['count']
        pie_labels = top_10['language']
    
    ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Language Distribution (Top 10 + Others)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = figures_path / 'language_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved language distribution plot to {output_path}")

def plot_app_ratings_comparison(app_ratings: pd.DataFrame, figures_path: Path):
    """Generate and save app ratings comparison plot."""
    if app_ratings.empty:
        print("Skipping app ratings plot - no data available")
        return
    
    # Plot top 20 apps by review count
    top_apps = app_ratings.head(20)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Average rating by app
    bars1 = ax1.barh(range(len(top_apps)), top_apps['avg_rating'])
    ax1.set_xlabel('Average Rating')
    ax1.set_ylabel('App')
    ax1.set_title('Average Rating by App (Top 20 by Review Count)', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(top_apps)))
    ax1.set_yticklabels(top_apps.iloc[:, 0], fontsize=8)  # First column is app name
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}', ha='left', va='center')
    
    # Review count by app
    bars2 = ax2.barh(range(len(top_apps)), top_apps['review_count'])
    ax2.set_xlabel('Number of Reviews')
    ax2.set_ylabel('App')
    ax2.set_title('Review Count by App (Top 20)', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(top_apps)))
    ax2.set_yticklabels(top_apps.iloc[:, 0], fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    
    output_path = figures_path / 'app_ratings_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved app ratings comparison plot to {output_path}")

def plot_rating_distribution(df: pd.DataFrame, figures_path: Path):
    """Generate and save overall rating distribution plot."""
    # Find rating column
    rating_col = None
    for col in df.columns:
        if 'rating' in col.lower() or 'score' in col.lower():
            rating_col = col
            break
    
    if rating_col is None:
        print("Skipping rating distribution plot - no rating column found")
        return
    
    valid_ratings = df[rating_col].dropna()
    
    if len(valid_ratings) == 0:
        print("Skipping rating distribution plot - no valid ratings found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(valid_ratings, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(valid_ratings, vert=True)
    ax2.set_ylabel('Rating')
    ax2.set_title('Rating Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {valid_ratings.mean():.2f}\n'
    stats_text += f'Median: {valid_ratings.median():.2f}\n'
    stats_text += f'Std: {valid_ratings.std():.2f}\n'
    stats_text += f'Count: {len(valid_ratings)}'
    
    ax2.text(1.1, 0.7, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    output_path = figures_path / 'rating_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved rating distribution plot to {output_path}")

def generate_summary_report(df: pd.DataFrame, weekly_ratings: pd.DataFrame, 
                          lang_dist: pd.DataFrame, app_ratings: pd.DataFrame, 
                          figures_path: Path):
    """Generate a text summary report of key findings."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MULTILINGUAL APP REVIEWS ANALYTICS - SUMMARY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 20)
    report_lines.append(f"Total reviews: {len(df):,}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append(f"Date range: {df.select_dtypes(include=['datetime']).min().min()} to {df.select_dtypes(include=['datetime']).max().max()}")
    report_lines.append("")
    
    # Language statistics
    if not lang_dist.empty:
        report_lines.append("LANGUAGE DISTRIBUTION")
        report_lines.append("-" * 20)
        report_lines.append(f"Total languages: {len(lang_dist)}")
        report_lines.append("Top 5 languages:")
        for i, row in lang_dist.head(5).iterrows():
            report_lines.append(f"  {row['language']}: {row['count']:,} reviews ({row['percentage']:.1f}%)")
        report_lines.append("")
    
    # App statistics
    if not app_ratings.empty:
        report_lines.append("APP PERFORMANCE")
        report_lines.append("-" * 15)
        report_lines.append(f"Total apps analyzed: {len(app_ratings)}")
        report_lines.append("Top 5 apps by average rating:")
        for i, row in app_ratings.head(5).iterrows():
            app_name = row.iloc[0]  # First column is app name
            report_lines.append(f"  {app_name}: {row['avg_rating']:.2f} stars ({row['review_count']} reviews)")
        report_lines.append("")
    
    # Weekly trends
    if not weekly_ratings.empty:
        report_lines.append("RATING TRENDS")
        report_lines.append("-" * 13)
        overall_trend = "increasing" if weekly_ratings['mean'].iloc[-1] > weekly_ratings['mean'].iloc[0] else "decreasing"
        report_lines.append(f"Overall trend: {overall_trend}")
        report_lines.append(f"Average rating range: {weekly_ratings['mean'].min():.2f} - {weekly_ratings['mean'].max():.2f}")
        report_lines.append("")
    
    # Find rating column for overall stats
    rating_col = None
    for col in df.columns:
        if 'rating' in col.lower() or 'score' in col.lower():
            rating_col = col
            break
    
    if rating_col is not None:
        valid_ratings = df[rating_col].dropna()
        report_lines.append("OVERALL RATING STATISTICS")
        report_lines.append("-" * 25)
        report_lines.append(f"Mean rating: {valid_ratings.mean():.2f}")
        report_lines.append(f"Median rating: {valid_ratings.median():.2f}")
        report_lines.append(f"Standard deviation: {valid_ratings.std():.2f}")
        report_lines.append(f"Rating range: {valid_ratings.min():.1f} - {valid_ratings.max():.1f}")
        report_lines.append("")
    
    report_lines.append("GENERATED VISUALIZATIONS")
    report_lines.append("-" * 24)
    report_lines.append("• weekly_ratings_trend.png - Weekly average rating over time")
    report_lines.append("• language_distribution.png - Distribution of reviews by language")
    report_lines.append("• app_ratings_comparison.png - App performance comparison")
    report_lines.append("• rating_distribution.png - Overall rating distribution")
    report_lines.append("")
    report_lines.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = figures_path / 'analytics_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Generated summary report: {report_path}")
    print("\nKEY FINDINGS:")
    print(report_text)

def main():
    """Main analytics pipeline."""
    parser = argparse.ArgumentParser(description="Analyze multilingual app reviews dataset")
    parser.add_argument("--input", "-i", type=str, default="cleaned_reviews.parquet",
                        help="Input Parquet filename (default: cleaned_reviews.parquet)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        processed_data_path, figures_path = setup_paths()
        
        # Load processed dataset
        df = load_processed_data(processed_data_path, args.input)
        
        print("\n" + "="*50)
        print("Starting analytics pipeline...")
        print("="*50)
        
        # Compute KPIs
        print("\n1. Computing weekly average ratings...")
        weekly_ratings = compute_weekly_average_rating(df)
        
        print("\n2. Computing language distribution...")
        lang_dist = compute_language_distribution(df)
        
        print("\n3. Computing app ratings summary...")
        app_ratings = compute_app_ratings_summary(df)
        
        # Generate visualizations
        print("\n4. Generating visualizations...")
        plot_weekly_ratings_trend(weekly_ratings, figures_path)
        plot_language_distribution(lang_dist, figures_path)
        plot_app_ratings_comparison(app_ratings, figures_path)
        plot_rating_distribution(df, figures_path)
        
        # Generate summary report
        print("\n5. Generating summary report...")
        generate_summary_report(df, weekly_ratings, lang_dist, app_ratings, figures_path)
        
        print("\n" + "="*50)
        print("Analytics pipeline completed successfully!")
        print("="*50)
        print(f"Visualizations saved to: {figures_path}")
        
    except Exception as e:
        print(f"Error during analytics: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()