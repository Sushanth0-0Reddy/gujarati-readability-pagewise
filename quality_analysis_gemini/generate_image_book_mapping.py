#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image to Book Name Mapping Generator

This script creates an Excel file with all image paths/names 
and their corresponding book names in separate columns.
"""

import os
import re
from pathlib import Path
import pandas as pd

def extract_book_name(filename):
    """
    Extract clean book name from filename by removing page numbers and suffixes.
    
    Args:
        filename: Image filename
    
    Returns:
        Clean book name as string
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Handle special cases first
    
    # Pattern 1: Date-based scans (e.g., "20221219133422_00061")
    if re.match(r'^\d{14}_\d+[A-Za-z]*$', name):
        timestamp_part = name[:14]  # Extract full YYYYMMDDHHMMSS
        return f"Scan Collection {timestamp_part}"
    
    # Pattern 2: Gule Polaad special format
    if 'Gule_Polaad' in name:
        return "Gule Polaad"
    
    # Pattern 3: Other special underscore cases
    special_patterns = {
        'Gujarat_Ehtihasic_Prasang': 'Gujarat Ehtihasic Prasang',
        'Gata_Zarna_Gani': 'Gata Zarna Gani', 
        'Ba_Bapuni_Shili_Chayama': 'Ba Bapuni Shili Chayama',
        'Gandhijini_Apeksa': 'Gandhijini Apeksa',
        'Ane_Bhaumitika': 'Ane Bhaumitika',
        'Sathina_Sahityanu_Digdarshan': 'Sathina Sahityanu Digdarshan'
    }
    
    for pattern, clean_name in special_patterns.items():
        if pattern in name:
            return clean_name
    
    # Pattern 4: Sahajanand Swami format (e.g., "Sahajanand Swami M-064")
    if 'Sahajanand Swami' in name:
        return 'Sahajanand Swami'
    
    # Pattern 5: Sona na Vruksho long format
    if 'Sona na Vruksho' in name:
        return 'Sona na Vruksho Manilal H Patel'
    
    # Pattern 6: Thakkarbapa format (e.g., "Thakkarbapa-1-015")
    if 'Thakkarbapa' in name:
        return 'Thakkarbapa'
    
    # Pattern 7: Volume format
    if name.startswith('Vol'):
        return 'Volume Collection'
    
    # Pattern 8: Page split format (_p1, _p2)
    if re.search(r'_p[12]$', name):
        base_name = re.sub(r'_p[12]$', '', name)
        return extract_book_name(base_name + '.dummy')
    
    # Pattern 9: Standard book_page format (e.g., "Vibhavana_197")
    if '_' in name:
        # Split on last underscore to handle names with multiple underscores
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            book_part, page_part = parts
            
            # Check if page part looks like a page number
            if (page_part.isdigit() or 
                re.match(r'\d+[A-Za-z]*$', page_part) or 
                re.match(r'[A-Za-z]*\d+$', page_part)):
                
                # Clean up the book part
                clean_book = book_part.replace('_', ' ')
                
                # Handle some specific cases
                if clean_book == 'Himalay No Pravas':
                    # Look for year in the page part or original filename
                    if '1941' in name:
                        return 'Himalay No Pravas 1941'
                    elif '1962' in name:
                        return 'Himalay No Pravas 1962'
                    else:
                        return 'Himalay No Pravas'
                
                return clean_book
    
    # Pattern 10: Hyphen-based format (e.g., "title4", "files", etc.)
    if name in ['title4', 'files', 'image_checker', 'verifier']:
        return f"Utility File ({name})"
    
    # Default: treat entire name as book name (clean up underscores)
    return name.replace('_', ' ')

def generate_image_book_mapping(images_dir, output_file):
    """
    Generate Excel file with image paths and corresponding book names.
    
    Args:
        images_dir: Path to the images directory
        output_file: Path for the output Excel file
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ Directory not found: {images_dir}")
        return
    
    # Image file extensions to consider
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}
    
    # List to store mapping data
    mapping_data = []
    
    print(f"📂 Analyzing images in: {images_dir}")
    
    # Process all image files
    for image_file in sorted(images_dir.iterdir()):
        if image_file.is_file():
            if image_file.suffix.lower() in image_extensions:
                book_name = extract_book_name(image_file.name)
                mapping_data.append({
                    'Image Name': image_file.name,
                    'Image Path': str(image_file),
                    'Book Name': book_name
                })
            elif image_file.suffix.lower() in {'.txt', '.sh'}:
                # Handle non-image files
                book_name = extract_book_name(image_file.name)
                mapping_data.append({
                    'Image Name': image_file.name,
                    'Image Path': str(image_file),
                    'Book Name': book_name
                })
    
    # Create DataFrame
    df = pd.DataFrame(mapping_data)
    
    # Sort by book name, then by image name for better organization
    df = df.sort_values(['Book Name', 'Image Name'])
    
    print(f"📊 Processing complete:")
    print(f"   Total files: {len(df)}")
    print(f"   Unique books: {df['Book Name'].nunique()}")
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main mapping sheet
        df.to_excel(writer, sheet_name='Image to Book Mapping', index=False)
        
        # Summary by book
        book_summary = df.groupby('Book Name').agg({
            'Image Name': 'count',
            'Image Path': 'first'  # Just to get first example
        }).rename(columns={'Image Name': 'File Count', 'Image Path': 'Sample Path'})
        book_summary = book_summary.reset_index()
        book_summary = book_summary.sort_values('File Count', ascending=False)
        book_summary.to_excel(writer, sheet_name='Books Summary', index=False)
        
        # Book names only (alphabetical)
        unique_books = sorted(df['Book Name'].unique())
        books_only_df = pd.DataFrame({'Book Name': unique_books})
        books_only_df.to_excel(writer, sheet_name='Book Names Only', index=False)
    
    print(f"📋 Excel file created: {output_file}")
    print(f"   Sheets: 'Image to Book Mapping', 'Books Summary', 'Book Names Only'")
    print(f"   Total entries: {len(df)}")

def preview_mapping(df, num_samples=10):
    """Preview the mapping data."""
    print(f"\n📄 Sample Mapping (First {num_samples} entries):")
    print(df[['Image Name', 'Book Name']].head(num_samples).to_string(index=False))
    
    print(f"\n📊 Statistics:")
    print(f"   Total Images: {len(df)}")
    print(f"   Unique Books: {df['Book Name'].nunique()}")
    
    print(f"\n📈 Top 10 Books by File Count:")
    book_counts = df['Book Name'].value_counts().head(10)
    for i, (book, count) in enumerate(book_counts.items(), 1):
        print(f"   {i:2d}. {book:<40} ({count:3d} files)")

def main():
    """Main function to generate image-book mapping Excel file."""
    # Set up paths
    project_root = Path(__file__).parent
    images_dir = project_root / "DQA_data" / "sampled_images"
    output_file = project_root / "image_book_mapping.xlsx"
    
    print("📚 Image to Book Name Mapping Generator")
    print("=" * 60)
    
    # Generate the mapping
    try:
        generate_image_book_mapping(images_dir, output_file)
        
        # Load and preview the data
        df = pd.read_excel(output_file, sheet_name='Image to Book Mapping')
        preview_mapping(df)
        
        print(f"\n✅ Image-Book mapping saved to: {output_file}")
        print(f"📋 Use this file to see which book each image belongs to!")
        
    except Exception as e:
        print(f"❌ Error creating Excel file: {e}")
        print("   Make sure you have pandas and openpyxl installed:")
        print("   pip install pandas openpyxl")

if __name__ == "__main__":
    main() 