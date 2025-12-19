#!/bin/bash
# Setup script for Python virtual environment
# Run this script to install all dependencies

set -e

cd "$(dirname "$0")"

echo "🐍 Setting up Python virtual environment..."
echo ""

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.13..."
    /opt/homebrew/bin/python3.13 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "Python version: $(python --version)"
echo ""

# Upgrade pip
echo "📦 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies - use requirements-all.txt (consolidated)
echo ""
echo "📦 Installing all dependencies..."
echo "   (This includes: backend, ML training, orchestrator, and all Google Cloud SDKs)"
echo ""

if [ -f "requirements-all.txt" ]; then
    pip install -r requirements-all.txt
elif [ -f "pyproject.toml" ]; then
    # Alternative: install using pyproject.toml with all optional dependencies
    pip install -e ".[all]"
else
    # Fallback: install from individual requirements files
    echo "Installing from individual requirements files..."
    pip install -r ml_pipeline/trainer/requirements.txt
    pip install -r backend/requirements.txt
    pip install -r ml_pipeline/orchestrator/requirements.txt
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Installed packages:"
pip list | head -20
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run training, use:"
echo "  ./run_training.sh --sample-size 300"
echo ""
echo "Or install using pyproject.toml:"
echo "  pip install -e '.[all]'"
