#!/bin/bash
# Render build script

set -e  # Exit on error

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || echo "NLTK download optional, continuing..."

echo "Build completed successfully!"
