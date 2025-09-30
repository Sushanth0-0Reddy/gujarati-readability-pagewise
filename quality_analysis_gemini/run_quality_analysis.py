#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple runner script for Document Quality Analysis

Usage:
    python run_quality_analysis.py
    
This script will:
1. Process all images in DQA_data/sampled_images/
2. Generate 5 random 40% crops per image
3. Analyze each crop using Gemini 2.5 Flash
4. Save all crop analyses in a single JSON file per image to DQA_data/dqa_outputs/
"""

import sys
from pathlib import Path

# Add current directory to path so we can import the analyzer
sys.path.append(str(Path(__file__).parent))

# Import the main analyzer
from document_quality_analyzer import main

if __name__ == "__main__":
    print("🚀 Starting Document Quality Analysis...")
    print("This will analyze images and generate quality reports.")
    print("Press Ctrl+C to stop at any time.\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}") 