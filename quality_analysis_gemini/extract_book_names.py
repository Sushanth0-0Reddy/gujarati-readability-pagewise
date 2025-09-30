#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Book Name Extractor

This script analyzes all image files and extracts unique book names 
by removing page numbers, suffixes, and other identifiers.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

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

def get_all_book_names(images_dir):
    """
    Extract all unique book names from the images directory.
    
    Args:
        images_dir: Path to the images directory
    
    Returns:
        Dictionary with book names and their file counts
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ Directory not found: {images_dir}")
        return {}
    
    # Image file extensions to consider
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}
    
    # Dictionary to store book -> count
    books_count = defaultdict(int)
    all_files = []
    
    print(f"📂 Analyzing images in: {images_dir}")
    
    # Process all image files
    for image_file in images_dir.iterdir():
        if image_file.is_file():
            if image_file.suffix.lower() in image_extensions:
                book_name = extract_book_name(image_file.name)
                books_count[book_name] += 1
                all_files.append((image_file.name, book_name))
            elif image_file.suffix.lower() in {'.txt', '.sh'}:
                # Handle non-image files
                book_name = extract_book_name(image_file.name)
                books_count[book_name] += 1
                all_files.append((image_file.name, book_name))
    
    print(f"📊 Processing complete:")
    print(f"   Total files: {sum(books_count.values())}")
    print(f"   Unique books: {len(books_count)}")
    
    return dict(books_count), all_files

def main():
    """Main function to extract and display all book names."""
    # Set up paths
    project_root = Path(__file__).parent
    images_dir = project_root / "DQA_data" / "sampled_images"
    
    print("📚 Comprehensive Book Name Extractor")
    print("=" * 60)
    
    # Get all book names
    books_count, all_files = get_all_book_names(images_dir)
    
    if not books_count:
        print("❌ No files found or directory doesn't exist")
        return
    
    # Sort books alphabetically
    sorted_books = sorted(books_count.items())
    
    print(f"\n📖 EXHAUSTIVE LIST OF ALL BOOK NAMES:")
    print(f"Total Unique Books: {len(sorted_books)}")
    print(f"Total Files: {sum(books_count.values())}")
    print("\n" + "=" * 60)
    
    # Print all unique book names
    for i, (book_name, count) in enumerate(sorted_books, 1):
        print(f"{i:3d}. {book_name}")
    
    print("\n" + "=" * 60)
    print(f"\n📊 DETAILED BREAKDOWN (with file counts):")
    
    # Sort by count (descending) for detailed breakdown
    sorted_by_count = sorted(books_count.items(), key=lambda x: x[1], reverse=True)
    
    for i, (book_name, count) in enumerate(sorted_by_count, 1):
        print(f"{i:3d}. {book_name:<50} ({count:3d} files)")
    
    # Save to file
    output_file = project_root / "all_book_names.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EXHAUSTIVE LIST OF ALL BOOK NAMES\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Unique Books: {len(sorted_books)}\n")
        f.write(f"Total Files: {sum(books_count.values())}\n")
        f.write(f"Generated: {__file__}\n\n")
        
        f.write("ALPHABETICAL LIST:\n")
        f.write("-" * 30 + "\n")
        for i, (book_name, count) in enumerate(sorted_books, 1):
            f.write(f"{i:3d}. {book_name}\n")
        
        f.write(f"\nDETAILED BREAKDOWN (by file count):\n")
        f.write("-" * 40 + "\n")
        for i, (book_name, count) in enumerate(sorted_by_count, 1):
            f.write(f"{i:3d}. {book_name:<50} ({count:3d} files)\n")
    
    print(f"\n✅ Complete list saved to: {output_file}")

if __name__ == "__main__":
    main() 