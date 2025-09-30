#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quality Score Analysis and Visualization

This script analyzes quality analysis results from DQA outputs,
calculates averages and variance, and creates candlestick plots
grouped by book names.
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import matplotlib.patches as mpatches

def extract_book_name(filename):
    """Extract book name from filename."""
    # Remove the _quality_analysis.json suffix
    name = filename.replace('_quality_analysis.json', '')
    
    # Handle special cases first
    if re.match(r'^\d{14}_\d+[A-Za-z]*$', name):
        timestamp_part = name[:14]
        return f"Scan Collection {timestamp_part}"
    
    if 'Gule_Polaad' in name:
        return "Gule Polaad"
    
    special_patterns = {
        'Gujarat_Ehtihasic_Prasang': 'Gujarat Ehtihasic Prasang',
        'Gata_Zarna_Gani': 'Gata Zarna Gani',
        'Sathina_Sahityanu_Digdarshan': 'Sathina Sahityanu Digdarshan'
    }
    
    for pattern, clean_name in special_patterns.items():
        if pattern in name:
            return clean_name
    
    if 'Sahajanand Swami' in name:
        return 'Sahajanand Swami'
    
    if 'Himalay No Pravas' in name:
        if '1941' in name:
            return 'Himalay No Pravas 1941'
        elif '1962' in name:
            return 'Himalay No Pravas 1962'
        else:
            return 'Himalay No Pravas'
    
    # Standard book_page format
    if '_' in name:
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            book_part, page_part = parts
            if (page_part.isdigit() or re.match(r'\d+[A-Za-z]*$', page_part)):
                return book_part.replace('_', ' ')
    
    return name.replace('_', ' ')

def extract_numeric_rating(rating_str):
    """Extract numeric value from rating string like '3.5/5 (Good)'."""
    if not rating_str:
        return None
    
    # Extract number before /5
    match = re.search(r'(\d+(?:\.\d+)?)/5', rating_str)
    if match:
        return float(match.group(1))
    return None

def extract_metric_ratings(metric_ratings):
    """Extract numeric values from metric ratings."""
    metrics = {}
    for metric, rating in metric_ratings.items():
        numeric_value = extract_numeric_rating(rating)
        if numeric_value is not None:
            metrics[metric] = numeric_value
    return metrics

def load_quality_data(dqa_outputs_dir):
    """Load all quality analysis data."""
    dqa_outputs_dir = Path(dqa_outputs_dir)
    if not dqa_outputs_dir.exists():
        print(f"❌ Directory not found: {dqa_outputs_dir}")
        return {}
    
    quality_data = []
    
    print(f"📂 Loading quality data from: {dqa_outputs_dir}")
    
    json_files = list(dqa_outputs_dir.glob("*_quality_analysis.json"))
    print(f"📊 Found {len(json_files)} quality analysis files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract metadata
            original_image = data['image_metadata']['original_image']
            book_name = extract_book_name(json_file.name)
            
            # Extract crop ratings
            crops_data = []
            overall_ratings = []
            all_metrics = defaultdict(list)
            
            for crop in data.get('crops', []):
                crop_quality = crop.get('quality_analysis', {})
                
                # Overall rating
                overall_rating = extract_numeric_rating(crop_quality.get('overallRating', ''))
                if overall_rating is not None:
                    overall_ratings.append(overall_rating)
                
                # Individual metrics
                summary = crop_quality.get('summary', {})
                metric_ratings = summary.get('metricRatings', {})
                metrics = extract_metric_ratings(metric_ratings)
                
                for metric, value in metrics.items():
                    all_metrics[metric].append(value)
                
                crops_data.append({
                    'crop_index': crop.get('crop_index', 0),
                    'overall_rating': overall_rating,
                    'metrics': metrics
                })
            
            # Calculate statistics
            if overall_ratings:
                image_stats = {
                    'image_name': original_image,
                    'book_name': book_name,
                    'file_path': str(json_file),
                    'total_crops': len(overall_ratings),
                    'avg_rating': np.mean(overall_ratings),
                    'min_rating': np.min(overall_ratings),
                    'max_rating': np.max(overall_ratings),
                    'std_rating': np.std(overall_ratings),
                    'median_rating': np.median(overall_ratings),
                    'all_ratings': overall_ratings,
                    'crops_data': crops_data
                }
                
                # Add metric averages
                for metric, values in all_metrics.items():
                    if values:
                        image_stats[f'avg_{metric.lower().replace(" & ", "_").replace(" ", "_")}'] = np.mean(values)
                        image_stats[f'std_{metric.lower().replace(" & ", "_").replace(" ", "_")}'] = np.std(values)
                
                quality_data.append(image_stats)
                
        except Exception as e:
            print(f"⚠️  Error processing {json_file.name}: {e}")
            continue
    
    print(f"✅ Loaded quality data for {len(quality_data)} images")
    return quality_data

def create_candlestick_plot_by_book(quality_data, output_dir):
    """Create candlestick plots grouped by book."""
    if not quality_data:
        print("❌ No quality data to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(quality_data)
    
    # Group by book and sort
    book_groups = df.groupby('book_name')
    
    # Sort books by average rating (descending)
    book_avg_ratings = df.groupby('book_name')['avg_rating'].mean().sort_values(ascending=False)
    
    print(f"📊 Creating plots for {len(book_groups)} books...")
    
    # Create main candlestick plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Prepare data for plotting
    plot_data = []
    x_labels = []
    x_positions = []
    book_boundaries = []
    current_x = 0
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(book_groups)))
    book_colors = {}
    
    for i, (book_name, book_data) in enumerate(book_groups):
        book_data = book_data.sort_values('image_name')  # Sort images within book
        book_colors[book_name] = colors[i]
        book_start = current_x
        
        for _, row in book_data.iterrows():
            plot_data.append({
                'x': current_x,
                'open': row['min_rating'],
                'high': row['max_rating'],
                'low': row['min_rating'],
                'close': row['avg_rating'],
                'median': row['median_rating'],
                'book_name': book_name,
                'image_name': row['image_name'],
                'avg_rating': row['avg_rating'],
                'std_rating': row['std_rating']
            })
            
            x_labels.append(row['image_name'].replace('.png', '').replace('.jpg', '')[:15])
            x_positions.append(current_x)
            current_x += 1
        
        book_boundaries.append((book_start, current_x - 1, book_name))
    
    # Plot candlesticks
    for point in plot_data:
        x = point['x']
        low = point['low']
        high = point['high']
        avg = point['close']
        median = point['median']
        book_name = point['book_name']
        
        color = book_colors[book_name]
        
        # Candlestick body (min to max)
        ax.plot([x, x], [low, high], color=color, linewidth=2, alpha=0.7)
        
        # Average point
        ax.scatter(x, avg, color=color, s=60, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Median line
        ax.plot([x-0.2, x+0.2], [median, median], color='red', linewidth=2, alpha=0.8)
        
        # Error bars for standard deviation
        if point['std_rating'] > 0:
            ax.errorbar(x, avg, yerr=point['std_rating'], fmt='none', 
                       ecolor='gray', alpha=0.5, capsize=3)
    
    # Customize plot
    ax.set_xlim(-0.5, len(plot_data) - 0.5)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel('Images (Grouped by Book)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Rating (0-5 scale)', fontsize=12, fontweight='bold')
    ax.set_title('Document Quality Analysis by Book\n(Candlestick: Min-Max Range, Dot: Average, Red Line: Median)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add book boundary lines and labels
    for start, end, book_name in book_boundaries:
        # Vertical separator
        if start > 0:
            ax.axvline(start - 0.5, color='black', linestyle='--', alpha=0.3)
        
        # Book label
        mid_point = (start + end) / 2
        ax.text(mid_point, 5.3, book_name.replace(' ', '\n'), 
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=book_colors[book_name], alpha=0.3))
    
    # Remove x-axis labels (too crowded)
    ax.set_xticks([])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='gray', alpha=0.7, label='Min-Max Range'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, label='Average Rating'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Median Rating'),
        plt.Line2D([0], [0], color='gray', alpha=0.5, label='Std Deviation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plot_file = output_dir / "quality_analysis_by_book_candlestick.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Candlestick plot saved: {plot_file}")
    
    # Create summary statistics plot
    create_book_summary_plot(df, output_dir)
    
    plt.show()
    
    return df

def create_book_summary_plot(df, output_dir):
    """Create summary statistics plot by book."""
    # Calculate book-level statistics
    book_stats = df.groupby('book_name').agg({
        'avg_rating': ['mean', 'std', 'count'],
        'std_rating': 'mean'
    }).round(3)
    
    book_stats.columns = ['avg_rating_mean', 'avg_rating_std', 'image_count', 'avg_std_within_image']
    book_stats = book_stats.sort_values('avg_rating_mean', ascending=True)
    
    # Create horizontal bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot 1: Average rating by book
    y_pos = np.arange(len(book_stats))
    bars = ax1.barh(y_pos, book_stats['avg_rating_mean'], 
                   xerr=book_stats['avg_rating_std'],
                   alpha=0.7, capsize=5)
    
    # Color bars by rating
    for i, (bar, rating) in enumerate(zip(bars, book_stats['avg_rating_mean'])):
        if rating >= 4.0:
            bar.set_color('green')
        elif rating >= 3.0:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name.replace(' ', '\n') for name in book_stats.index], fontsize=8)
    ax1.set_xlabel('Average Quality Rating', fontweight='bold')
    ax1.set_title('Average Quality Rating by Book', fontweight='bold')
    ax1.set_xlim(0, 5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (rating, std) in enumerate(zip(book_stats['avg_rating_mean'], book_stats['avg_rating_std'])):
        ax1.text(rating + 0.1, i, f'{rating:.2f}±{std:.2f}', 
                va='center', fontsize=8)
    
    # Plot 2: Image count by book
    bars2 = ax2.barh(y_pos, book_stats['image_count'], alpha=0.7, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([name.replace(' ', '\n') for name in book_stats.index], fontsize=8)
    ax2.set_xlabel('Number of Images Analyzed', fontweight='bold')
    ax2.set_title('Image Count by Book', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(book_stats['image_count']):
        ax2.text(count + 0.5, i, str(int(count)), va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_file = output_dir / "quality_summary_by_book.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"📊 Summary plot saved: {summary_file}")
    
    plt.show()
    
    # Print statistics
    print(f"\n📈 QUALITY ANALYSIS SUMMARY:")
    print(f"{'='*60}")
    print(f"{'Book Name':<30} {'Avg Rating':<12} {'Std Dev':<10} {'Images':<8}")
    print(f"{'-'*60}")
    
    for book_name, stats in book_stats.iterrows():
        print(f"{book_name[:29]:<30} {stats['avg_rating_mean']:<12.2f} {stats['avg_rating_std']:<10.2f} {int(stats['image_count']):<8}")

def main():
    """Main function to analyze quality scores."""
    project_root = Path(__file__).parent
    dqa_outputs_dir = project_root / "DQA_data" / "dqa_outputs"
    output_dir = project_root / "quality_analysis_plots"
    
    print("📊 Quality Score Analysis and Visualization")
    print("=" * 60)
    
    # Load quality data
    quality_data = load_quality_data(dqa_outputs_dir)
    
    if not quality_data:
        print("❌ No quality data found")
        return
    
    # Create visualizations
    df = create_candlestick_plot_by_book(quality_data, output_dir)
    
    # Save processed data
    data_file = output_dir / "quality_analysis_data.csv"
    df.to_csv(data_file, index=False)
    print(f"💾 Quality data saved: {data_file}")
    
    print(f"\n✅ Analysis complete! Check {output_dir} for visualizations.")

if __name__ == "__main__":
    main() 