"""
Test script to verify the model evaluation module works correctly.
This generates a sample confusion matrix and other visualizations locally.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model_evaluation import run_comprehensive_evaluation

# Set up matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

print("=" * 70)
print("🧪 TESTING MODEL EVALUATION MODULE")
print("=" * 70)
print()

# Create synthetic dataset with realistic crop classification features
np.random.seed(42)
n_samples = 200
n_features = 20

# Generate features (simulating NDVI stats, elevation, lat/long encoding, etc.)
X = np.random.randn(n_samples, n_features)
X[:, 0] = np.random.uniform(0.3, 0.9, n_samples)  # NDVI mean (0-1 range)
X[:, 1] = np.random.uniform(0.05, 0.2, n_samples)  # NDVI std
X[:, 2] = np.random.uniform(0, 500, n_samples)  # Elevation

# Create labels (5 crops)
crops = ['Corn', 'Soybeans', 'Alfalfa', 'Winter Wheat', 'Cotton']
y = np.random.choice(crops, size=n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])

# Add some structure to make classification possible
# Corn: high NDVI mean, medium elevation
corn_mask = y == 'Corn'
X[corn_mask, 0] += 0.1  # Higher NDVI
X[corn_mask, 2] = np.random.uniform(200, 400, np.sum(corn_mask))

# Soybeans: medium NDVI, similar to corn (will cause some confusion)
soy_mask = y == 'Soybeans'
X[soy_mask, 0] += 0.05
X[soy_mask, 2] = np.random.uniform(150, 350, np.sum(soy_mask))

# Alfalfa: very high NDVI, low elevation
alfalfa_mask = y == 'Alfalfa'
X[alfalfa_mask, 0] += 0.15
X[alfalfa_mask, 2] = np.random.uniform(0, 200, np.sum(alfalfa_mask))

# Winter Wheat: medium NDVI, high elevation
wheat_mask = y == 'Winter Wheat'
X[wheat_mask, 2] = np.random.uniform(300, 500, np.sum(wheat_mask))

# Cotton: low NDVI, very low elevation
cotton_mask = y == 'Cotton'
X[cotton_mask, 0] -= 0.1
X[cotton_mask, 2] = np.random.uniform(0, 100, np.sum(cotton_mask))

# Create feature names
feature_names = [
    'ndvi_mean', 'ndvi_std', 'elevation_m',
    'ndvi_min', 'ndvi_max', 'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
    'ndvi_early', 'ndvi_late',
    'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
    'elevation_bin_0', 'elevation_bin_1', 'elevation_bin_2', 'elevation_bin_3',
    'ndvi_derived_1', 'ndvi_derived_2'
]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Dataset:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {len(feature_names)}")
print(f"   Classes: {len(crops)}")
print()

# Create and train model
print("🤖 Training RandomForest model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"   Training accuracy: {train_score:.2%}")
print(f"   Test accuracy: {test_score:.2%}")
print()

# Run comprehensive evaluation
print("🔍 Running comprehensive evaluation...")
output_dir = './test_evaluation_output'
os.makedirs(output_dir, exist_ok=True)

try:
    eval_results = run_comprehensive_evaluation(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir
    )
    
    print()
    print("=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📁 Output directory: {output_dir}")
    print()
    print("Generated files:")
    for key, value in eval_results.items():
        if 'path' in key.lower():
            if value and os.path.exists(value):
                file_size = os.path.getsize(value) / 1024  # KB
                print(f"   ✅ {os.path.basename(value)} ({file_size:.1f} KB)")
            else:
                print(f"   ⚠️  {key}: {value}")
    
    print()
    print("📊 Advanced Metrics Summary:")
    if 'advanced_metrics' in eval_results:
        metrics = eval_results['advanced_metrics']['overall_metrics']
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
        print(f"   Matthews Corr: {metrics['matthews_corrcoef']:.3f}")
        print(f"   Macro F1: {metrics['macro_avg']['f1_score']:.2%}")
        print(f"   Weighted F1: {metrics['weighted_avg']['f1_score']:.2%}")
    
    print()
    print("🎉 All evaluation functions working correctly!")
    print()
    print(f"💡 To view the confusion matrix:")
    print(f"   open {output_dir}/confusion_matrix_enhanced.png")
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ EVALUATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

