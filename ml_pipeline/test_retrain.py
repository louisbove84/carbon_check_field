#!/usr/bin/env python3

"""
Test Script for Retraining Pipeline
====================================
Validates the retraining pipeline with dry-run options.
"""

import sys
import argparse
from auto_retrain_model import (
    load_data_from_bigquery,
    check_training_data_quality,
    engineer_features,
    train_local_model,
    save_model_to_gcs,
    deploy_model_to_vertex_ai,
    test_prediction
)

def test_data_loading():
    """Test BigQuery data loading"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        df = load_data_from_bigquery()
        print(f"✅ Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Crops: {df['crop'].unique().tolist()}")
        return df
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_data_quality(df):
    """Test data quality checks"""
    print("\n" + "="*60)
    print("TEST 2: Data Quality")
    print("="*60)
    
    try:
        quality_report = check_training_data_quality(df)
        print(f"✅ Quality report generated")
        print(f"   Total samples: {quality_report['total_samples']}")
        print(f"   Balanced: {quality_report['is_balanced']}")
        return quality_report
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_feature_engineering(df):
    """Test feature engineering"""
    print("\n" + "="*60)
    print("TEST 3: Feature Engineering")
    print("="*60)
    
    try:
        df_enhanced, feature_cols = engineer_features(df.copy())
        print(f"✅ Features engineered")
        print(f"   Original features: {df.shape[1]}")
        print(f"   Enhanced features: {df_enhanced.shape[1]}")
        print(f"   Feature list: {len(feature_cols)} features")
        print(f"   New features: {[f for f in feature_cols if f not in df.columns]}")
        return df_enhanced, feature_cols
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None


def test_model_training(df, feature_cols):
    """Test model training"""
    print("\n" + "="*60)
    print("TEST 4: Model Training")
    print("="*60)
    
    try:
        pipeline, metrics = train_local_model(df, feature_cols)
        print(f"✅ Model trained (Pipeline with scaler + classifier)")
        print(f"   Train accuracy: {metrics['train_accuracy']:.2%}")
        print(f"   Test accuracy: {metrics['test_accuracy']:.2%}")
        print(f"   Train samples: {metrics['n_train_samples']}")
        print(f"   Test samples: {metrics['n_test_samples']}")
        return pipeline, metrics
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None, None


def test_model_prediction(pipeline, df, feature_cols):
    """Test model prediction using pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Model Prediction")
    print("="*60)
    
    try:
        # Get a test sample (pipeline handles scaling automatically)
        test_sample = df[feature_cols].iloc[0:1]  # DataFrame format for pipeline
        prediction = pipeline.predict(test_sample)[0]
        probabilities = pipeline.predict_proba(test_sample)[0]
        
        # Get classifier from pipeline for classes
        classifier = pipeline.named_steps['classifier']
        
        print(f"✅ Prediction successful")
        print(f"   Actual crop: {df.iloc[0]['crop']}")
        print(f"   Predicted crop: {prediction}")
        print(f"   Confidence: {max(probabilities):.2%}")
        print(f"   All probabilities: {dict(zip(classifier.classes_, probabilities))}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def run_full_test(args):
    """Run complete test pipeline"""
    print("\n" + "="*60)
    print("🧪 RETRAINING PIPELINE TEST SUITE")
    print("="*60)
    
    # Test 1: Load data
    df = test_data_loading()
    if df is None or len(df) == 0:
        print("\n❌ FAILED: Cannot load data")
        return False
    
    # Test 2: Check quality
    quality_report = test_data_quality(df)
    if quality_report is None:
        print("\n❌ FAILED: Data quality check failed")
        return False
    
    # Test 3: Engineer features
    df_enhanced, feature_cols = test_feature_engineering(df)
    if df_enhanced is None or feature_cols is None:
        print("\n❌ FAILED: Feature engineering failed")
        return False
    
    # Test 4: Train model
    pipeline, metrics = test_model_training(df_enhanced, feature_cols)
    if pipeline is None:
        print("\n❌ FAILED: Model training failed")
        return False
    
    # Test 5: Test prediction
    prediction_ok = test_model_prediction(pipeline, df_enhanced, feature_cols)
    if not prediction_ok:
        print("\n❌ FAILED: Model prediction failed")
        return False
    
    # Optional: Save to GCS (if requested)
    if args.save_model:
        print("\n" + "="*60)
        print("TEST 6: Save to GCS (Optional)")
        print("="*60)
        try:
            from auto_retrain_model import BUCKET_NAME
            model_path = save_model_to_gcs(pipeline, feature_cols, BUCKET_NAME)
            print(f"✅ Model saved to {model_path}")
        except Exception as e:
            print(f"⚠️  Save failed (non-critical): {e}")
    
    # Optional: Deploy to Vertex AI (if requested)
    if args.deploy_model:
        print("\n" + "="*60)
        print("TEST 7: Deploy to Vertex AI (Optional)")
        print("="*60)
        try:
            from auto_retrain_model import ENDPOINT_ID
            endpoint = deploy_model_to_vertex_ai(None, ENDPOINT_ID)
            print(f"✅ Model deployed to endpoint: {endpoint.name}")
        except Exception as e:
            print(f"⚠️  Deploy failed (non-critical): {e}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   • Training samples: {quality_report['total_samples']}")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   • Crops: {list(quality_report['crops'].keys())}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test retraining pipeline')
    parser.add_argument('--save-model', action='store_true', 
                        help='Save model to GCS (requires write permissions)')
    parser.add_argument('--deploy-model', action='store_true',
                        help='Deploy model to Vertex AI (requires permissions)')
    
    args = parser.parse_args()
    
    success = run_full_test(args)
    
    if not success:
        sys.exit(1)
    
    print("\n💡 Next steps:")
    print("   1. Deploy Cloud Function: ./deploy_retrain_function.sh")
    print("   2. Test deployed function: curl <function-url>")
    print("   3. Set up Cloud Scheduler for monthly retraining")


if __name__ == '__main__':
    main()

