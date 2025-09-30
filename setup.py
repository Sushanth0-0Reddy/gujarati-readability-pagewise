#!/usr/bin/env python3
"""
Setup script for Gujarati Readability Classification - Page-wise Training System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="gujarati-readability-pagewise",
    version="1.0.0",
    author="Sarvam AI",
    author_email="team@sarvam.ai",
    description="Page-wise readability classification system for Gujarati documents using deep learning and XGBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarvam-ai/gujarati-readability-pagewise",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "viz": [
            "plotly>=5.15.0",
            "dash>=2.10.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "plotly>=5.15.0",
            "dash>=2.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "extract-features=scripts.extract_pagewise_features:main",
            "train-xgboost=scripts.train_pagewise_xgboost:main",
            "predict-book=scripts.predict_single_book:main",
            "plot-predictions=scripts.plot_prediction_distribution_general:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    keywords=[
        "machine-learning",
        "computer-vision", 
        "document-analysis",
        "readability",
        "gujarati",
        "nlp",
        "ocr",
        "xgboost",
        "deep-learning",
        "transformers",
        "vision-language-model",
    ],
    project_urls={
        "Bug Reports": "https://github.com/sarvam-ai/gujarati-readability-pagewise/issues",
        "Source": "https://github.com/sarvam-ai/gujarati-readability-pagewise",
        "Documentation": "https://github.com/sarvam-ai/gujarati-readability-pagewise/blob/main/README.md",
    },
)
