# Python Virtual Environment Setup

## ✅ What's Been Done

1. **Created a fresh virtual environment** using Python 3.13.9 (from Homebrew)
2. **Location**: `carbon_check_field/venv/`
3. **Python version**: 3.13.9

## 🚀 Next Steps

Run the setup script to install all dependencies:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
./setup_venv.sh
```

This will:
- Activate the virtual environment
- Upgrade pip, setuptools, and wheel
- Install all ML pipeline dependencies

## 📝 Using the Virtual Environment

**Activate the environment:**
```bash
source venv/bin/activate
```

**Run training:**
```bash
# Option 1: Use the helper script
./run_training.sh --sample-size 300

# Option 2: Activate venv manually
source venv/bin/activate
python ml_pipeline/tools/test_training.py --sample-size 300
```

**Deactivate when done:**
```bash
deactivate
```

## 🔍 Troubleshooting

If you encounter SSL certificate errors:
1. Make sure you're connected to the internet
2. Try running: `pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package>`

If Python still crashes:
- The virtual environment uses Python 3.13.9 from Homebrew
- This is a fresh installation that should avoid the previous crashes
- If issues persist, we may need to check for system-level conflicts

## 📊 Viewing Results

After training completes, view TensorBoard:
```bash
source venv/bin/activate
tensorboard --logdir test_output/tensorboard_logs
```

The confusion matrix will show all classes including 'Other' if it's in your BigQuery data.
