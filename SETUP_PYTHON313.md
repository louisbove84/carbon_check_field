# Python 3.13 Setup Instructions

## Issue
Python 3.13 requires:
1. **GEOS library** for `shapely` (system dependency)
2. **Newer pydantic** version (2.5.3 is too old for Python 3.13)

## Solution

### Step 1: Install GEOS (required for shapely)
```bash
brew install geos
```

### Step 2: Install Python dependencies
The `requirements-all.txt` has been updated with Python 3.13-compatible versions:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
source venv/bin/activate
pip install -r requirements-all.txt
```

### Step 3: Run training
```bash
python ml_pipeline/tools/test_training.py --sample-size 300 --min-samples-per-crop 15
```

## Alternative: Use Python 3.11 (matches Dockerfiles)

If you prefer to match the Dockerfiles exactly (which use Python 3.11):

```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Create venv with Python 3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-all.txt
```

## Note
The Dockerfiles use Python 3.11, so using Python 3.11 locally ensures exact compatibility with production.
