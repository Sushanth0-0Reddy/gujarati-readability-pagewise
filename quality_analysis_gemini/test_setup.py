#!/usr/bin/env python3

"""
Test script to verify document quality analyzer setup
"""

import sys
from pathlib import Path

print("🧪 Testing Document Quality Analyzer Setup")
print("=" * 50)

# Test 1: Check Python version
print(f"✅ Python version: {sys.version}")

# Test 2: Import standard libraries
try:
    import json, os, time, asyncio, random
    from datetime import timedelta
    from pathlib import Path
    print("✅ Standard libraries imported successfully")
except ImportError as e:
    print(f"❌ Standard library import failed: {e}")
    sys.exit(1)

# Test 3: Import third-party libraries
try:
    from tqdm import tqdm
    from PIL import Image
    from dotenv import load_dotenv
    from tenacity import retry, wait_exponential, stop_after_attempt
    print("✅ Third-party libraries imported successfully")
except ImportError as e:
    print(f"❌ Third-party library import failed: {e}")
    print("   Install missing packages with: pip install tqdm pillow python-dotenv tenacity")
    sys.exit(1)

# Test 4: Import Google Cloud libraries
try:
    from google.genai import Client as GeminiClient
    from google.genai.types import HttpOptions, Part, GenerateContentConfig
    print("✅ Google Cloud AI libraries imported successfully")
except ImportError as e:
    print(f"❌ Google Cloud AI library import failed: {e}")
    print("   Install with: pip install google-genai")
    sys.exit(1)

# Test 5: Check directory structure
project_root = Path(__file__).parent.parent
input_dir = project_root / "DQA_data" / "sampled_images"
output_dir = project_root / "DQA_data" / "dqa_outputs"

if input_dir.exists():
    input_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"✅ Input directory exists with {len(input_files)} images")
else:
    print(f"❌ Input directory not found: {input_dir}")

if output_dir.exists():
    print(f"✅ Output directory exists: {output_dir}")
else:
    print(f"⚠️  Output directory will be created: {output_dir}")

# Test 6: Check credentials
possible_credentials = [
    "vision-projects-463307-service-account.json",
    "vision-projects-463307-a8a8a88fc2c1.json",
    "service-account.json",
    "credentials.json"
]

credentials_found = False
for cred_file in possible_credentials:
    path = project_root / cred_file
    if path.exists():
        print(f"✅ Credentials file found: {cred_file}")
        credentials_found = True
        break

if not credentials_found:
    print("⚠️  No credentials file found. Ensure you have a Google Cloud service account JSON file")

# Test 7: Test image processing
try:
    from PIL import Image
    import io
    
    # Create a simple test image
    test_img = Image.new('RGB', (100, 100), 'white')
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG')
    print("✅ Image processing test successful")
except Exception as e:
    print(f"❌ Image processing test failed: {e}")

print("\n" + "=" * 50)
print("🎯 Setup Summary:")
print(f"   - Python version: {sys.version.split()[0]}")
print(f"   - Required libraries: {'✅ All present' if True else '❌ Missing some'}")
print(f"   - Input images: {len(input_files) if 'input_files' in locals() else 0}")
print(f"   - Credentials: {'✅ Found' if credentials_found else '⚠️  Not found'}")
print("\n💡 If all tests pass, you can run:")
print("   python3 quality_analysis/document_quality_analyzer.py")
print("   python3 quality_analysis/run_quality_analysis.py") 