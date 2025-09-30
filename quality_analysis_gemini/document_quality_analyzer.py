#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Add project root to Python path
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

"""
Document Quality Analysis Script

This script analyzes document image quality using Gemini 2.5 Flash.
For each input image, it generates 5 random 40% crops and analyzes quality.
All crop analyses are saved in a single JSON file per image.
"""

import json
import os
import sys
import time
import asyncio
import random
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import io
from tenacity import retry, wait_exponential, stop_after_attempt

from google.genai import Client as GeminiClient
from google.genai.types import HttpOptions, Part, GenerateContentConfig, ThinkingConfig

# Load environment variables
load_dotenv()

# Configuration
MODEL_ID = "gemini-2.5-pro"
API_TIMEOUT_MILLISECONDS = 300000  # 5 minutes
TEMPERATURE = 0.0
REASONING_TOKENS = 128

# Rate limiting configuration (using same retry config as experiment_config.yaml but with 32 workers)
MAX_CONCURRENT_REQUESTS = 32
REQUEST_DELAY = 1  # Same as experiment_config.yaml
RETRY_DELAY_BASE = 10  # Same as experiment_config.yaml
NUM_WORKERS = 32

# Cloud configuration
DEFAULT_PROJECT_ID = "vision-projects-463307"
DEFAULT_REGION = "us-central1"
DEFAULT_CREDENTIALS_FILE = "vision-projects-463307-service-account.json"

# Quality analysis prompt - Enhanced Critical Analysis with Resolution and Color Fidelity Assessment
QUALITY_ANALYSIS_PROMPT = """Role: You are an extremely critical and meticulous expert AI assistant specializing in document digitization and image quality analysis. You have very high standards and are known for being unforgiving in your assessments.

Task: Your goal is to rigorously and critically evaluate the quality of a provided scanned document image. Be pessimistic and look for every possible flaw, imperfection, or issue. Based on your analysis, you must generate a response in a single JSON format. Analyze the user's image based on the examples provided below.

Critical Evaluation Standards

    Any visible blur, softness, or lack of crisp edges should be heavily penalized.

    Even minor shadows, uneven lighting, or contrast issues are significant problems.

    Small amounts of noise, artifacts, or imperfections are unacceptable.

    Any geometric distortion, skew, or perspective issues are major flaws.

    Incomplete capture, including inconsistent or tight margins, is a critical failure.

    Be extremely demanding about text sharpness and uniformity.

    Resolution (DPI) is a key metric; low resolution is a major flaw regardless of clarity.

    For color documents, Color Fidelity (accuracy, bleed, moiré) is critical.

Examples

Example 1: Very Poor Quality Scan

Input Description: "The image is a severely degraded scan with major distortions, extensive blur, heavy shadows, missing content, and extreme pixelation."

Required JSON Output:
{
  "overallRating": "1/5 (Very Poor)",
  "detailedAnalysis": {
    "clarity": "The text is completely illegible due to severe blur and distortion. Character edges are completely lost, making OCR impossible. This is utterly unacceptable for any digital preservation purpose.",
    "contrast": "Contrast is virtually non-existent. Text blends into the background in most areas, creating an unusable document with no clear distinction between content and page.",
    "noiseAndArtifacts": "Extensive digital artifacts, compression issues, dust, stains, and noise completely overwhelm the document content. Multiple large obstructions render significant portions unreadable.",
    "geometricDistortion": "Severe perspective distortion and extreme skew make the document appear warped and unnatural. Lines are curved and text orientation is chaotic.",
    "illuminationAndShadows": "Extremely uneven lighting with heavy shadows, bright hotspots, and complete darkness in some areas. Lighting quality is completely unacceptable.",
    "completenessAndMargins": "Major portions of the document are cropped out or missing entirely. What remains shows no concept of proper margins. This is a fundamentally incomplete and poorly framed capture.",
    "textLineThickness": "Text line thickness is completely inconsistent with broken characters, missing strokes, and irregular line weights throughout the document.",
    "resolution": "Extremely low resolution, likely screen-grab quality below 100 DPI. The image is a pixelated mess, completely unsuitable for any purpose, including basic on-screen viewing.",
    "colorFidelity": "Catastrophic color failure. If this were a color document, the scan quality indicates there would be extreme color shifts, oversaturation, and severe bleed, rendering colors unrecognizable and obstructive."
  },
  "summary": {
    "justificationParagraph": "This scan represents a complete failure in document digitization. Every quality metric fails catastrophically, with severe blur, low resolution, distortion, missing content, and lighting issues making it completely unusable for any purpose. This is an example of how not to scan documents.",
    "metricRatings": {
      "Clarity": "1/5",
      "Contrast": "1/5",
      "Noise & Artifacts": "1/5",
      "Geometric Distortion": "1/5",
      "Illumination & Shadows": "1/5",
      "Completeness & Margins": "1/5",
      "Text Line Thickness": "1/5",
      "Resolution": "1/5",
      "Color Fidelity": "1/5"
    }
  }
}

Example 2: Poor Quality Scan (Mixed Flaws)

Input Description: "A smartphone photo of a document. The center is fairly clear but the page is heavily skewed, one side is dark with shadows, and content is cut off."

Required JSON Output:
{
  "overallRating": "2/5 (Poor)",
  "detailedAnalysis": {
    "clarity": "Clarity is inconsistent. The center of the document is somewhat readable but suffers from softness. The edges are significantly blurred due to focus and lighting issues, making OCR unreliable.",
    "contrast": "Contrast is passable in the well-lit center (3/5), but it degrades severely in the shadowed areas, where text becomes nearly unreadable (1/5).",
    "noiseAndArtifacts": "Visible digital noise and smudging, typical of a low-quality phone camera, detract from the document's professionalism and readability.",
    "geometricDistortion": "Severe perspective distortion is the most critical failure here. The document is heavily skewed, indicating it was photographed at a steep angle, which is unacceptable.",
    "illuminationAndShadows": "A prominent shadow cast by the phone or hand covers a significant portion of the document, creating areas of poor visibility. This is a major lighting flaw.",
    "completenessAndMargins": "Content is clearly cropped at the bottom edge, and the skewed perspective means margins are completely non-uniform. The capture is incomplete.",
    "textLineThickness": "Line thickness varies significantly across the document, appearing thinner in bright spots and bloated or blurred in darker areas.",
    "resolution": "Resolution is likely low-to-moderate from a phone's sensor, but the effective resolution is ruined by the blur and distortion. It fails to meet archival standards.",
    "colorFidelity": "Not assessed as the source appears monochrome. However, any color document scanned with such poor technique would exhibit severe fidelity issues and color shifts in the shadows."
  },
  "summary": {
    "justificationParagraph": "This scan is poor overall, primarily due to catastrophic geometric distortion and major lighting flaws. While the text in the center has some degree of clarity, the combination of skew, shadows, and incomplete capture makes it unsuitable for any professional use. The few acceptable areas do not compensate for the critical failures elsewhere.",
    "metricRatings": {
      "Clarity": "2/5",
      "Contrast": "2/5",
      "Noise & Artifacts": "2/5",
      "Geometric Distortion": "1/5",
      "Illumination & Shadows": "1/5",
      "Completeness & Margins": "2/5",
      "Text Line Thickness": "2/5",
      "Resolution": "2/5",
      "Color Fidelity": "2/5"
    }
  }
}

Example 3: Fair Quality Scan (Mixed Flaws)

Input Description: "A flatbed scan that is very sharp, but the contrast is washed out and there are dust particles on the scanner glass."

Required JSON Output:
{
  "overallRating": "3/5 (Fair)",
  "detailedAnalysis": {
    "clarity": "Text is exceptionally sharp and well-defined. Character edges are crisp, which is a strong point of this scan.",
    "contrast": "The primary flaw is poor contrast. The text appears faded and washed-out against a grayish background, suggesting incorrect scanner settings. This significantly impairs readability.",
    "noiseAndArtifacts": "Multiple dust specks and a few small stray lines are visible across the document, indicating a dirty scanner bed. While not obstructing text, they are unacceptable for a clean digital copy.",
    "geometricDistortion": "The geometric alignment is excellent. The document is perfectly flat and squared with no discernible skew or distortion.",
    "illuminationAndShadows": "Lighting is perfectly even across the entire document, which is a major advantage of a flatbed scanner. There are no shadows or hotspots.",
    "completenessAndMargins": "The entire document is captured with generous and even margins. The framing is very well done.",
    "textLineThickness": "Text line thickness is perfectly uniform due to the high clarity and good source material.",
    "resolution": "The scan boasts a high resolution, evident from the sharp character details. This would be excellent for OCR if not for the contrast issues.",
    "colorFidelity": "N/A (monochrome document)."
  },
  "summary": {
    "justificationParagraph": "This scan is a frustrating mix of excellent and poor qualities. While it excels in clarity, resolution, and geometry, it is severely undermined by poor contrast and visible artifacts from a dirty scanner. The washed-out appearance makes it only 'Fair' and not suitable for professional use without significant post-processing. It's a case of good hardware used improperly.",
    "metricRatings": {
      "Clarity": "5/5",
      "Contrast": "2/5",
      "Noise & Artifacts": "2/5",
      "Geometric Distortion": "5/5",
      "Illumination & Shadows": "5/5",
      "Completeness & Margins": "5/5",
      "Text Line Thickness": "5/5",
      "Resolution": "5/5",
      "Color Fidelity": "N/A"
    }
  }
}

Example 4: Good Quality Scan (Minor Flaws)

Input Description: "A high-quality scan that is nearly perfect, but has a very faint shadow along one edge and is slightly misaligned."

Required JSON Output:
{
  "overallRating": "4/5 (Good)",
  "detailedAnalysis": {
    "clarity": "Text clarity is excellent. All characters are sharp and perfectly defined, making it ideal for OCR processing.",
    "contrast": "The contrast between the black text and white background is outstanding, with no faded or washed-out areas.",
    "noiseAndArtifacts": "The image is pristine and free of any dust, stains, or digital noise. The background is clean and uniform.",
    "geometricDistortion": "A minor but noticeable rotational skew is present. The document is not perfectly aligned to the horizontal axis, which prevents it from being a perfect 5/5 scan.",
    "illuminationAndShadows": "The lighting is even across 95% of the document, but a very faint, soft shadow is visible along the left margin, indicating a minor imperfection in the lighting setup.",
    "completenessAndMargins": "The full document is captured with good, consistent margins on all sides.",
    "textLineThickness": "Text line thickness is perfectly consistent throughout the document.",
    "resolution": "Resolution is high and entirely suitable for archival and reproduction purposes. No pixelation is visible.",
    "colorFidelity": "N/A (monochrome document)."
  },
  "summary": {
    "justificationParagraph": "This is a very good quality scan that meets almost all professional standards. Its excellent clarity, contrast, and resolution are commendable. However, it is held back from perfection by a minor but definite geometric skew and a faint shadow. While suitable for almost any purpose, these small flaws prevent it from achieving a top rating.",
    "metricRatings": {
      "Clarity": "5/5",
      "Contrast": "5/5",
      "Noise & Artifacts": "5/5",
      "Geometric Distortion": "3/5",
      "Illumination & Shadows": "4/5",
      "Completeness & Margins": "5/5",
      "Text Line Thickness": "5/5",
      "Resolution": "5/5",
      "Color Fidelity": "N/A"
    }
  }
}

Example 5: Excellent Quality Scan

Input Description: "A pristine, high-resolution scan with perfect clarity, even lighting, no distortion, and complete capture."

Required JSON Output:
{
  "overallRating": "5/5 (Excellent)",
  "detailedAnalysis": {
    "clarity": "The text is exceptionally sharp and clear with perfectly defined character edges. This represents the gold standard for OCR-ready documents with no visible softness or blur.",
    "contrast": "Outstanding contrast between text and background with deep blacks and pure whites. Text stands out crisply against the background.",
    "noiseAndArtifacts": "The image is completely pristine with no visible dust, stains, artifacts, or digital noise. The scan quality is flawless.",
    "geometricDistortion": "Perfect geometric accuracy with no skew, distortion, or perspective issues. The document is perfectly aligned and flat.",
    "illuminationAndShadows": "Absolutely even illumination across the entire document with no shadows, hotspots, or lighting variations. Professional-grade lighting quality.",
    "completenessAndMargins": "Complete capture of all document content with ideal, uniform margins. Every element is fully preserved with perfect framing.",
    "textLineThickness": "Perfect uniformity in text line thickness with consistent stroke weights throughout. This reflects optimal scanning conditions.",
    "resolution": "High-resolution scan (estimated 400-600 DPI or higher), providing exceptional detail suitable for professional archival, print reproduction, and flawless OCR. This is the benchmark for preservation.",
    "colorFidelity": "Perfect color fidelity. Colors are rendered with absolute accuracy, true to the original source, with no perceptible shift, bleed, or artifacts. Meets the highest standards for graphic and historical preservation."
  },
  "summary": {
    "justificationParagraph": "This scan represents the absolute pinnacle of document digitization quality. Every aspect, including resolution and color fidelity, is flawless, meeting the highest professional standards for archival preservation and OCR processing. This is a benchmark example of perfect scanning technique.",
    "metricRatings": {
      "Clarity": "5/5",
      "Contrast": "5/5",
      "Noise & Artifacts": "5/5",
      "Geometric Distortion": "5/5",
      "Illumination & Shadows": "5/5",
      "Completeness & Margins": "5/5",
      "Text Line Thickness": "5/5",
      "Resolution": "5/5",
      "Color Fidelity": "5/5"
    }
  }
}

Your Task

Now, critically analyze the image provided by the user with extreme attention to detail and high standards. Look for every possible flaw or imperfection. Be pessimistic and unforgiving in your assessment. Based on your analysis, generate a single JSON object with the same structure as the examples above. Return ONLY the JSON, no additional text."""

def get_vertex_client() -> GeminiClient:
    """Set up Vertex AI client with proper credentials."""
    # Set up Google Cloud credentials - try config file first, then fallbacks
    possible_credentials = [
        DEFAULT_CREDENTIALS_FILE,
        "vision-projects-463307-service-account.json",
        "vision-projects-463307-a8a8a88fc2c1.json",
        "service-account.json",
        "credentials.json"
    ]
    
    credentials_path = None
    for cred_file in possible_credentials:
        path = str(project_root / cred_file)
        if os.path.exists(path):
            credentials_path = path
            break
    
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        print(f"Using credentials file: {os.path.basename(credentials_path)}")
    else:
        print("⚠️  No credentials file found. Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable")

    # Set default project and region if not set
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_CLOUD_PROJECT"] = DEFAULT_PROJECT_ID
    if not os.getenv("GOOGLE_CLOUD_REGION"):
        os.environ["GOOGLE_CLOUD_REGION"] = DEFAULT_REGION

    return GeminiClient(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_REGION"),
        http_options=HttpOptions(api_version="v1", timeout=API_TIMEOUT_MILLISECONDS),
    )

def image_to_part(image: Image.Image) -> Part:
    """Convert PIL Image to Part object for Gemini API."""
    # Convert RGBA/LA/P to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        # Create a white background
        rgb_image = Image.new('RGB', image.size, 'white')
        if image.mode == 'RGBA':
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        else:
            rgb_image.paste(image)
        image = rgb_image

    # Convert image to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=95)
    img_bytes = img_buffer.getvalue()

    return Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

def generate_random_crop(image: Image.Image, crop_percentage: float = 0.4) -> tuple[Image.Image, dict]:
    """
    Generate a random crop of the specified percentage from the image.
    
    Args:
        image: PIL Image to crop
        crop_percentage: Percentage of original image to keep (0.4 = 40%)
    
    Returns:
        Tuple of (Cropped PIL Image, crop coordinates dict)
    """
    width, height = image.size
    
    # Calculate crop dimensions
    crop_width = int(width * crop_percentage)
    crop_height = int(height * crop_percentage)
    
    # Calculate maximum possible start positions
    max_x = width - crop_width
    max_y = height - crop_height
    
    # Generate random start position
    start_x = random.randint(0, max_x) if max_x > 0 else 0
    start_y = random.randint(0, max_y) if max_y > 0 else 0
    
    # Calculate end positions
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    
    # Crop the image
    cropped = image.crop((start_x, start_y, end_x, end_y))
    
    # Return coordinates as well
    coordinates = {
        "crop_box": [start_x, start_y, end_x, end_y],
        "top_left": [start_x, start_y],
        "bottom_right": [end_x, end_y],
        "width": crop_width,
        "height": crop_height,
        "center": [start_x + crop_width // 2, start_y + crop_height // 2],
        "percentage_of_original": crop_percentage
    }
    
    return cropped, coordinates

@retry(wait=wait_exponential(multiplier=RETRY_DELAY_BASE, max=120), stop=stop_after_attempt(4))
async def analyze_image_quality_async(client: GeminiClient, image: Image.Image, semaphore: asyncio.Semaphore) -> dict:
    """
    Analyze document quality of an image using Gemini API.
    
    Args:
        client: Gemini client
        image: PIL Image to analyze
        semaphore: Asyncio semaphore for rate limiting
    
    Returns:
        Dict containing quality analysis JSON
    """
    async with semaphore:
        try:
            # Add delay between requests to respect rate limits
            await asyncio.sleep(REQUEST_DELAY)
            
            # Convert image to part
            image_part = image_to_part(image)
            contents = [image_part, QUALITY_ANALYSIS_PROMPT]
            
            # Make API call (note: generate_content is not async, so we run it in executor)
            loop = asyncio.get_event_loop()
            
            # Prepare thinking configuration
            thinking_config = ThinkingConfig(thinking_budget=REASONING_TOKENS)
            
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=GenerateContentConfig(
                        temperature=TEMPERATURE,
                        thinking_config=thinking_config,
                    )
                )
            )
            
            response_text = response.text.strip() if response.text else ""
            
            # Try to parse JSON from response
            try:
                # Remove any markdown formatting if present
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()
                
                quality_data = json.loads(response_text)
                return quality_data
            except json.JSONDecodeError as e:
                print(f"    ❌ Failed to parse JSON response: {e}")
                print(f"    Raw response: {response_text[:200]}...")
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text
                }
            
        except Exception as e:
            print(f"    ❌ Quality analysis error: {e}")
            raise

async def process_image_async(image_path: Path, output_dir: Path, client: GeminiClient, semaphore: asyncio.Semaphore) -> bool:
    """
    Process a single image: generate 5 random crops and analyze each.
    Save all crop analyses in a single JSON file.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        client: Gemini client
        semaphore: Asyncio semaphore for rate limiting
    
    Returns:
        True if successful, False otherwise
    """
    image_name = image_path.stem
    print(f"🔍 [{image_name}] Processing image...")
    
    try:
        # Load the image
        with Image.open(image_path) as img:
            img = img.copy()  # Create a copy to avoid issues with context manager
            
        print(f"📏 [{image_name}] Image size: {img.size}")
        
        # Initialize output structure for all crops
        image_analysis = {
            "image_metadata": {
                "original_image": image_path.name,
                "original_size": img.size,
                "crop_percentage": 40.0,
                "total_crops": 5,
                "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "crops": []
        }
        
        # Generate 5 random crops
        crops_processed = 0
        for crop_idx in range(5):
            try:
                # Generate random crop (40% of original)
                cropped_img, crop_coordinates = generate_random_crop(img, crop_percentage=0.4)
                
                print(f"✂️  [{image_name}] Crop {crop_idx + 1}: {cropped_img.size} at {crop_coordinates['top_left']}")
                
                # Analyze quality
                quality_result = await analyze_image_quality_async(client, cropped_img, semaphore)
                
                # Add crop-specific metadata
                crop_data = {
                    "crop_index": crop_idx + 1,
                    "cropped_size": cropped_img.size,
                    "crop_coordinates": crop_coordinates,
                    "quality_analysis": quality_result
                }
                
                # Add to crops list
                image_analysis["crops"].append(crop_data)
                
                crops_processed += 1
                print(f"✅ [{image_name}] Crop {crop_idx + 1} analyzed")
                
            except Exception as e:
                print(f"❌ [{image_name}] Failed to process crop {crop_idx + 1}: {e}")
                # Add error entry for failed crop
                error_data = {
                    "crop_index": crop_idx + 1,
                    "cropped_size": None,
                    "crop_coordinates": None,
                    "quality_analysis": {
                        "error": f"Failed to analyze crop: {str(e)}"
                    }
                }
                image_analysis["crops"].append(error_data)
                continue
        
        # Save single output file with all crops
        output_filename = f"{image_name}_quality_analysis.json"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(image_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📊 [{image_name}] Completed: {crops_processed}/5 crops processed and saved to {output_filename}")
        return crops_processed > 0
        
    except Exception as e:
        print(f"❌ [{image_name}] Failed to process image: {e}")
        return False

async def image_worker(file_queue: asyncio.Queue, semaphore: asyncio.Semaphore,
                      output_dir: Path, worker_id: int, global_pbar: tqdm):
    """Worker that processes images from the queue."""
    client = get_vertex_client()  # Create a new client instance for each worker

    while True:
        try:
            image_path = await file_queue.get()
            if image_path is None:  # Shutdown signal
                break

            success = await process_image_async(image_path, output_dir, client, semaphore)

            if success:
                # Update progress bar
                global_pbar.update(1)

            file_queue.task_done()

        except Exception as e:
            print(f"Error processing image: {e}")
            file_queue.task_done()

async def process_all_images_async(input_dir: Path, output_dir: Path):
    """
    Process all images in the input directory with parallel workers.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Get all image files from input directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))

    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return

    print(f"🎯 Found {len(image_files)} images to process")
    print(f"📋 Each image will generate 5 random crops (40% of original)")
    print(f"📄 Total expected outputs: {len(image_files)} JSON files (1 per image with 5 crops each)")

    # Create queue and semaphore
    file_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Add files to queue
    for image_file in image_files:
        await file_queue.put(image_file)

    # Create progress bar
    progress_bar = tqdm(total=len(image_files), desc="Processing images")

    # Create workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = asyncio.create_task(
            image_worker(
                file_queue, semaphore, output_dir, i+1, progress_bar
            )
        )
        workers.append(worker)

    # Wait for all files to be processed
    await file_queue.join()

    # Shutdown workers
    for _ in range(NUM_WORKERS):
        await file_queue.put(None)

    # Wait for workers to finish
    await asyncio.gather(*workers)

    # Close progress bar
    progress_bar.close()

    # Final summary
    total_json_files = len(list(output_dir.glob("*_quality_analysis.json")))
    print(f"\n{'='*60}")
    print(f"🏁 Processing Summary:")
    print(f"   Images processed: {len(image_files)}")
    print(f"   JSON files created: {total_json_files}")
    print(f"   Expected outputs: {len(image_files)} files (1 per image with 5 crops each)")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}")

def main():
    """Main function to run document quality analysis."""
    
    # Set up paths
    input_dir = project_root / "DQA_data" / "sampled_images"
    output_dir = project_root / "DQA_data" / "dqa_outputs"
    
    print("🚀 Document Quality Analyzer")
    print("=" * 60)
    print(f"🖼️  Input directory: {input_dir}")
    print(f"📤 Output directory: {output_dir}")
    print(f"🤖 Model: {MODEL_ID}")
    print(f"🎛️  Rate limiting: {MAX_CONCURRENT_REQUESTS} concurrent requests, {REQUEST_DELAY}s delay")
    print(f"👥 Workers: {NUM_WORKERS} parallel workers")
    print()

    start_time = time.time()

    # Process all images
    try:
        asyncio.run(process_all_images_async(input_dir, output_dir))
        
        # Calculate and print total time taken
        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n✨ Processing complete!")
        print(f"⏱️  Total time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main() 