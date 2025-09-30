#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Quality Score Visualization with Zoom/Pan Navigation

Creates a large-scale interactive plot sorted by book-wise mean rating (low to high)
with proper navigation controls for handling 1000+ images across 26 books.
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

def load_quality_data(dqa_outputs_dir):
    """Load all quality analysis data."""
    dqa_outputs_dir = Path(dqa_outputs_dir)
    if not dqa_outputs_dir.exists():
        print(f"❌ Directory not found: {dqa_outputs_dir}")
        return []
    
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
            overall_ratings = []
            crop_coordinates = []
            
            for crop in data.get('crops', []):
                crop_quality = crop.get('quality_analysis', {})
                
                # Overall rating
                overall_rating = extract_numeric_rating(crop_quality.get('overallRating', ''))
                if overall_rating is not None:
                    overall_ratings.append(overall_rating)
                
                # Crop coordinates
                coords = crop.get('crop_coordinates', {})
                if coords:
                    crop_coordinates.append(coords)
            
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
                    'crop_coordinates': crop_coordinates
                }
                
                quality_data.append(image_stats)
                
        except Exception as e:
            print(f"⚠️  Error processing {json_file.name}: {e}")
            continue
    
    print(f"✅ Loaded quality data for {len(quality_data)} images")
    return quality_data

def create_interactive_plotly_visualization(quality_data, output_dir):
    """Create interactive Plotly visualization with zoom/pan controls."""
    if not quality_data:
        print("❌ No quality data to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(quality_data)
    
    # Calculate book-level statistics and sort by mean rating (low to high)
    book_stats = df.groupby('book_name').agg({
        'avg_rating': ['mean', 'std', 'count'],
        'std_rating': 'mean'
    }).round(3)
    
    book_stats.columns = ['book_mean', 'book_std', 'image_count', 'avg_within_std']
    book_stats = book_stats.sort_values('book_mean', ascending=True)  # Low to high
    
    print(f"📊 Creating interactive visualization for {len(book_stats)} books...")
    print(f"📈 Book ranking (worst to best):")
    for i, (book, stats) in enumerate(book_stats.iterrows(), 1):
        print(f"  {i:2d}. {book:<30} {stats['book_mean']:.2f} ± {stats['book_std']:.2f} ({int(stats['image_count'])} images)")
    
    # Prepare data for plotting in sorted order
    plot_data = []
    x_position = 0
    book_boundaries = []
    
    # Color palette for books
    colors = px.colors.qualitative.Set3 * 3  # Repeat colors if needed
    book_colors = {}
    
    for i, (book_name, book_data) in enumerate(df.groupby('book_name')):
        # Sort by book mean rating
        book_mean = book_stats.loc[book_name, 'book_mean']
        book_data = book_data.sort_values('image_name')  # Sort images within book
        book_colors[book_name] = colors[i % len(colors)]
        
        book_start = x_position
        
        for _, row in book_data.iterrows():
            plot_data.append({
                'x': x_position,
                'image_name': row['image_name'].replace('.png', '').replace('.jpg', ''),
                'book_name': book_name,
                'avg_rating': row['avg_rating'],
                'min_rating': row['min_rating'],
                'max_rating': row['max_rating'],
                'std_rating': row['std_rating'],
                'median_rating': row['median_rating'],
                'total_crops': row['total_crops'],
                'book_mean': book_mean,
                'all_ratings': row['all_ratings']
            })
            x_position += 1
        
        book_boundaries.append({
            'start': book_start,
            'end': x_position - 1,
            'book_name': book_name,
            'book_mean': book_mean,
            'image_count': len(book_data)
        })
    
    # Sort plot_data by book mean rating
    plot_data_sorted = []
    x_pos = 0
    
    for book_name in book_stats.index:  # Already sorted by mean rating
        book_data_points = [p for p in plot_data if p['book_name'] == book_name]
        for point in book_data_points:
            point['x'] = x_pos
            plot_data_sorted.append(point)
            x_pos += 1
    
    plot_df = pd.DataFrame(plot_data_sorted)
    
    # Create main interactive plot
    fig = go.Figure()
    
    # Add traces for each book
    for book_name in book_stats.index:
        book_data = plot_df[plot_df['book_name'] == book_name]
        
        # Min-Max range (as error bars)
        fig.add_trace(go.Scatter(
            x=book_data['x'],
            y=book_data['avg_rating'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=book_data['max_rating'] - book_data['avg_rating'],
                arrayminus=book_data['avg_rating'] - book_data['min_rating'],
                visible=True,
                color='lightgray',
                thickness=1
            ),
            mode='markers',
            marker=dict(
                size=8,
                color=book_colors[book_name],
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            name=f"{book_name} (μ={book_stats.loc[book_name, 'book_mean']:.2f})",
            text=[f"Image: {name}<br>Book: {book}<br>Avg: {avg:.2f}<br>Range: {min_r:.1f}-{max_r:.1f}<br>Std: {std:.2f}<br>Crops: {crops}" 
                  for name, book, avg, min_r, max_r, std, crops in 
                  zip(book_data['image_name'], book_data['book_name'], book_data['avg_rating'], 
                      book_data['min_rating'], book_data['max_rating'], book_data['std_rating'], book_data['total_crops'])],
            hovertemplate='%{text}<extra></extra>',
            legendgroup=book_name
        ))
        
        # Add median line for each point
        fig.add_trace(go.Scatter(
            x=book_data['x'],
            y=book_data['median_rating'],
            mode='markers',
            marker=dict(
                symbol='line-ew',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name=f"{book_name} Median",
            showlegend=False,
            text=[f"Median: {med:.2f}" for med in book_data['median_rating']],
            hovertemplate='%{text}<extra></extra>',
            legendgroup=book_name
        ))
    
    # Add book boundary lines and annotations
    for i, boundary in enumerate(book_boundaries):
        if i > 0:  # Don't add line before first book
            fig.add_vline(
                x=boundary['start'] - 0.5,
                line=dict(color='black', width=1, dash='dash'),
                opacity=0.5
            )
        
        # Add book name annotation
        mid_x = (boundary['start'] + boundary['end']) / 2
        fig.add_annotation(
            x=mid_x,
            y=5.3,
            text=f"{boundary['book_name']}<br>μ={boundary['book_mean']:.2f}",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor=book_colors[boundary['book_name']],
            opacity=0.7,
            bordercolor='black',
            borderwidth=1
        )
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text="📊 Document Quality Analysis by Book (Sorted: Worst → Best)<br><sub>Interactive: Zoom, Pan, Click Legend to Toggle Books</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Images (Grouped by Book, Sorted by Book Mean Rating)",
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False
        ),
        yaxis=dict(
            title="Quality Rating (1-5 Scale)",
            range=[0.5, 5.5],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=False
        ),
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(r=200),  # Extra space for legend
        height=800,
        width=1400,
        plot_bgcolor='white'
    )
    
    # Add navigation instructions
    fig.add_annotation(
        text="📱 Navigation: Drag to pan • Scroll to zoom • Double-click to reset • Click legend to toggle books",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, color='blue'),
        bgcolor='lightblue',
        opacity=0.8
    )
    
    # Save interactive plot
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    html_file = output_dir / "interactive_quality_analysis_sorted.html"
    fig.write_html(str(html_file))
    print(f"📊 Interactive plot saved: {html_file}")
    
    # Also save static version
    png_file = output_dir / "interactive_quality_analysis_sorted.png"
    fig.write_image(str(png_file), width=1400, height=800, scale=2)
    print(f"📊 Static plot saved: {png_file}")
    
    return fig, book_stats

def create_book_summary_dashboard(book_stats, output_dir):
    """Create a summary dashboard of book statistics."""
    
    # Create horizontal bar chart sorted by mean rating
    fig = go.Figure()
    
    y_pos = list(range(len(book_stats)))
    colors = ['red' if mean < 3.0 else 'orange' if mean < 4.0 else 'green' 
              for mean in book_stats['book_mean']]
    
    fig.add_trace(go.Bar(
        x=book_stats['book_mean'],
        y=[name.replace(' ', '<br>') for name in book_stats.index],
        orientation='h',
        marker=dict(color=colors, opacity=0.8),
        error_x=dict(
            type='data',
            array=book_stats['book_std'],
            visible=True
        ),
        text=[f"{mean:.2f}±{std:.2f}" for mean, std in 
              zip(book_stats['book_mean'], book_stats['book_std'])],
        textposition='outside',
        hovertemplate='Book: %{y}<br>Mean Rating: %{x:.2f}<br>Std Dev: %{error_x.array:.2f}<br>Images: %{customdata}<extra></extra>',
        customdata=book_stats['image_count']
    ))
    
    fig.update_layout(
        title="📈 Book Quality Rankings (Sorted: Worst → Best)",
        xaxis_title="Average Quality Rating",
        yaxis_title="Books",
        height=max(600, len(book_stats) * 25),
        width=1000,
        margin=dict(l=200),
        plot_bgcolor='white'
    )
    
    # Save summary dashboard
    summary_html = output_dir / "book_summary_dashboard.html"
    fig.write_html(str(summary_html))
    print(f"📊 Book summary dashboard saved: {summary_html}")
    
    return fig

def main():
    """Main function to create interactive quality visualizations."""
    project_root = Path(__file__).parent
    dqa_outputs_dir = project_root / "DQA_data" / "dqa_outputs"
    output_dir = project_root / "interactive_quality_plots"
    
    print("📊 Interactive Quality Score Visualization")
    print("=" * 60)
    
    # Load quality data
    quality_data = load_quality_data(dqa_outputs_dir)
    
    if not quality_data:
        print("❌ No quality data found")
        return
    
    # Create interactive visualizations
    main_fig, book_stats = create_interactive_plotly_visualization(quality_data, output_dir)
    summary_fig = create_book_summary_dashboard(book_stats, output_dir)
    
    # Save processed data
    data_file = output_dir / "interactive_quality_data.csv"
    df = pd.DataFrame(quality_data)
    df.to_csv(data_file, index=False)
    print(f"💾 Quality data saved: {data_file}")
    
    print(f"\n✅ Interactive visualization complete!")
    print(f"🌐 Open these files in your browser:")
    print(f"   - Main Plot: {output_dir}/interactive_quality_analysis_sorted.html")
    print(f"   - Summary:   {output_dir}/book_summary_dashboard.html")

if __name__ == "__main__":
    main() 