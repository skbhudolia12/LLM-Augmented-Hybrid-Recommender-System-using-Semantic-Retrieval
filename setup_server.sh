#!/bin/bash
set -e

echo "=================================================="
echo " Setting up I2P-BERT Environment on A100 Server"
echo "=================================================="

# Assuming conda environment is already activated
echo "Installing dependencies into the active environment..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "IMPORTANT: Please check the .env file to ensure your TMDB_API_KEY is correct."
else
    echo ".env file already exists."
fi

echo "=================================================="
echo " Setup complete!"
echo " To run the pipeline:"
echo "   python -m scripts.run_pipeline"
echo "=================================================="
