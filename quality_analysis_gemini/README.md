# Document Quality Analysis

This tool analyzes the quality of scanned document images using Google's Gemini 2.5 Flash model. For each input image, it generates multiple random crops and evaluates various quality metrics.

## Features

- **Automated Quality Assessment**: Uses AI to evaluate document scan quality
- **Multiple Crop Analysis**: Generates 5 random 40% crops per image for comprehensive analysis
- **Structured Output**: Provides detailed JSON reports with quality metrics
- **Parallel Processing**: Handles multiple images concurrently for efficiency
- **Comprehensive Metrics**: Analyzes clarity, contrast, noise, distortion, lighting, and completeness

## Setup

### Prerequisites

1. **Google Cloud Credentials**: Ensure you have a valid service account JSON file
2. **Required Dependencies**: Install from the main project requirements

### Directory Structure

```
DQA_data/
├── sampled_images/     # Input images go here
└── dqa_outputs/        # Generated quality reports
```

## Usage

### Basic Usage

```bash
# From the quality_analysis directory
python document_quality_analyzer.py

# Or use the runner script
python run_quality_analysis.py
```

### Input Requirements

- Place your document images in: `/root/sarvam/akshar-experiments-pipeline/akshar-ocr-experiments/DQA_data/sampled_images/`
- Supported formats: PNG, JPG, JPEG, TIF, TIFF, WEBP, BMP

### Output

For each input image, the script generates 1 JSON file in `/root/sarvam/akshar-experiments-pipeline/akshar-ocr-experiments/DQA_data/dqa_outputs/`:

```
original_image_quality_analysis.json
```

Each file contains quality analysis for all 5 random crops of that image.

## Sample Output Format

Each JSON file contains all 5 crop analyses:

```json
{
  "image_metadata": {
    "original_image": "document.jpg",
    "original_size": [2480, 3508],
    "crop_percentage": 40.0,
    "total_crops": 5,
    "analysis_timestamp": "2024-07-30 17:30:45"
  },
  "crops": [
    {
      "crop_index": 1,
      "cropped_size": [992, 1403],
      "quality_analysis": {
        "overallRating": "4/5 (Good)",
        "detailedAnalysis": {
          "clarity": "Text is sharp and clear with minimal artifacts...",
          "contrast": "Good contrast between text and background...",
          "noiseAndArtifacts": "Minor dust spots present but not affecting readability...",
          "geometricDistortion": "Slight rotation detected but easily correctable...",
          "illuminationAndShadows": "Even lighting with minor shadows in corners...",
          "completeness": "All content visible, no cropping detected...",
          "textLineThickness": "Consistent line thickness throughout..."
        },
        "summary": {
          "justificationParagraph": "The document shows good quality...",
          "metricRatings": {
            "Clarity": "4/5",
            "Contrast": "4/5",
            "Noise & Artifacts": "3.5/5",
            "Geometric Distortion": "3/5",
            "Illumination & Shadows": "4/5",
            "Completeness": "5/5",
            "Text Line Thickness": "4/5"
          }
        }
      }
    },
    {
      "crop_index": 2,
      "cropped_size": [992, 1403],
      "quality_analysis": {
        "overallRating": "3/5 (Fair)",
        "detailedAnalysis": {
          "clarity": "Some blurring present in text areas...",
          "contrast": "Moderate contrast, readable but not optimal...",
          "noiseAndArtifacts": "Several dust spots and minor artifacts...",
          "geometricDistortion": "Minimal skew detected...",
          "illuminationAndShadows": "Uneven lighting with slight shadows...",
          "completeness": "Full content visible...",
          "textLineThickness": "Generally consistent with minor variations..."
        },
        "summary": {
          "justificationParagraph": "Acceptable quality with room for improvement...",
          "metricRatings": {
            "Clarity": "3/5",
            "Contrast": "3/5",
            "Noise & Artifacts": "2.5/5",
            "Geometric Distortion": "4/5",
            "Illumination & Shadows": "3/5",
            "Completeness": "5/5",
            "Text Line Thickness": "3.5/5"
          }
        }
      }
    }
    // ... crops 3, 4, and 5 follow the same structure
  ]
}
```

## Configuration

Key settings in `document_quality_analyzer.py`:

```python
MODEL_ID = "gemini-2.5-flash"           # AI model to use
MAX_CONCURRENT_REQUESTS = 5             # Parallel API calls
REQUEST_DELAY = 1                       # Seconds between requests
NUM_WORKERS = 3                         # Worker processes
```

## Quality Metrics

The analysis evaluates:

1. **Clarity**: Text sharpness and readability
2. **Contrast**: Text-background distinction
3. **Noise & Artifacts**: Dust, smudges, digital artifacts
4. **Geometric Distortion**: Skew, rotation, warping
5. **Illumination & Shadows**: Lighting uniformity
6. **Completeness**: Content cropping and visibility
7. **Text Line Thickness**: Font consistency

Each metric is rated on a 1-5 scale with detailed explanations.

## Troubleshooting

### Common Issues

1. **Credentials Error**: Ensure Google Cloud service account JSON is in the project root
2. **No Images Found**: Check that images are in the correct input directory
3. **API Rate Limits**: Reduce `MAX_CONCURRENT_REQUESTS` if getting rate limit errors
4. **Memory Issues**: Reduce `NUM_WORKERS` for large images

### Error Messages

- `❌ Input directory not found`: Create the DQA_data/sampled_images directory
- `⚠️ No credentials file found`: Place your service account JSON in the project root
- `❌ Failed to parse JSON response`: The AI response wasn't valid JSON (usually retries automatically)

## Performance

- **Processing Time**: ~10-15 seconds per crop (depends on image size and API response time)
- **Expected Output**: 1 JSON file per input image (containing 5 crop analyses)
- **Resource Usage**: Moderate CPU, low memory, network-dependent

## Integration

This tool is designed to work with the broader Akshar OCR pipeline and can be integrated with:

- OCR quality assessment workflows
- Document preprocessing pipelines
- Batch document analysis systems
- Quality control processes 