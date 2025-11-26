#!/bin/bash
# Manual Pipeline Testing Script
# ================================
# Tests the complete ML pipeline locally before deploying

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"

echo "============================================================"
echo "🧪 MANUAL ML PIPELINE TESTING"
echo "============================================================"
echo ""

# Check if user wants to test specific steps
if [ "$1" == "holdout" ]; then
    echo "📊 TEST 1: Creating Holdout Test Set"
    echo "------------------------------------------------------------"
    python3 << 'EOF'
from model_evaluation import create_or_load_holdout_set
df = create_or_load_holdout_set(force_recreate=False)
print(f"\n✅ Holdout set: {len(df)} samples")
print(f"Crops: {df['crop'].value_counts().to_dict()}")
EOF
    exit 0
fi

if [ "$1" == "collection" ]; then
    echo "📥 TEST 2: Triggering Data Collection"
    echo "------------------------------------------------------------"
    echo "Sending HTTP POST to Cloud Function..."
    curl -X POST \
      https://$REGION-$PROJECT_ID.cloudfunctions.net/monthly-data-collection
    
    echo ""
    echo ""
    echo "📋 Viewing logs (last 50 lines):"
    echo "------------------------------------------------------------"
    sleep 5
    gcloud functions logs read monthly-data-collection \
      --region=$REGION \
      --limit=50
    exit 0
fi

if [ "$1" == "retrain" ]; then
    echo "🤖 TEST 3: Triggering Model Retraining"
    echo "------------------------------------------------------------"
    echo "Sending HTTP POST to Cloud Function..."
    curl -X POST \
      https://$REGION-$PROJECT_ID.cloudfunctions.net/auto-retrain-model
    
    echo ""
    echo ""
    echo "📋 Viewing logs (last 200 lines):"
    echo "------------------------------------------------------------"
    sleep 10
    gcloud functions logs read auto-retrain-model \
      --region=$REGION \
      --limit=200
    exit 0
fi

if [ "$1" == "endpoint" ]; then
    echo "🎯 TEST 4: Testing Endpoint Predictions"
    echo "------------------------------------------------------------"
    python3 test_endpoint.py
    exit 0
fi

if [ "$1" == "metrics" ]; then
    echo "📊 TEST 5: Viewing Metrics in BigQuery"
    echo "------------------------------------------------------------"
    
    echo ""
    echo "Recent Model Performance:"
    echo "------------------------------------------------------------"
    bq query --use_legacy_sql=false --format=pretty \
    "SELECT 
      model_type,
      ROUND(accuracy, 3) as accuracy,
      ROUND(corn_f1, 3) as corn_f1,
      ROUND(soybeans_f1, 3) as soybeans_f1,
      ROUND(alfalfa_f1, 3) as alfalfa_f1,
      ROUND(winter_wheat_f1, 3) as winter_wheat_f1,
      evaluation_time
    FROM \`ml-pipeline-477612.crop_ml.model_performance\`
    ORDER BY evaluation_time DESC
    LIMIT 5"
    
    echo ""
    echo "Deployment History:"
    echo "------------------------------------------------------------"
    bq query --use_legacy_sql=false --format=pretty \
    "SELECT 
      deployment_time,
      deployment_decision,
      ROUND(accuracy, 3) as accuracy,
      training_samples
    FROM \`ml-pipeline-477612.crop_ml.deployment_history\`
    ORDER BY deployment_time DESC
    LIMIT 5"
    
    exit 0
fi

# If no argument or "all", run all tests
echo "🔄 Running ALL tests in sequence..."
echo ""

# Test 1: Holdout Set
echo "============================================================"
echo "📊 TEST 1: Creating/Loading Holdout Test Set"
echo "============================================================"
python3 << 'EOF'
from model_evaluation import create_or_load_holdout_set
df = create_or_load_holdout_set(force_recreate=False)
print(f"\n✅ Holdout set: {len(df)} samples")
print(f"Crops: {df['crop'].value_counts().to_dict()}")
EOF

echo ""
read -p "Press Enter to continue to Test 2..."
echo ""

# Test 2: Data Collection
echo "============================================================"
echo "📥 TEST 2: Manual Data Collection"
echo "============================================================"
echo "⚠️  WARNING: This will collect 400 new samples from Earth Engine"
echo "   and add them to BigQuery. This is billable."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    curl -X POST \
      https://$REGION-$PROJECT_ID.cloudfunctions.net/monthly-data-collection
    
    echo ""
    echo "⏳ Waiting 10 seconds for logs..."
    sleep 10
    
    echo ""
    echo "📋 Recent logs:"
    gcloud functions logs read monthly-data-collection \
      --region=$REGION \
      --limit=50
else
    echo "⏭️  Skipping data collection test"
fi

echo ""
read -p "Press Enter to continue to Test 3..."
echo ""

# Test 3: Model Retraining
echo "============================================================"
echo "🤖 TEST 3: Manual Model Retraining with Evaluation"
echo "============================================================"
echo "⚠️  WARNING: This will train a new model and potentially deploy it."
echo "   Training takes 5-10 minutes and is billable."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    curl -X POST \
      https://$REGION-$PROJECT_ID.cloudfunctions.net/auto-retrain-model
    
    echo ""
    echo "⏳ Training in progress... This will take 5-10 minutes."
    echo "   You can check progress in Cloud Console > Cloud Functions > auto-retrain-model > Logs"
    echo ""
    echo "   Or run: gcloud functions logs read auto-retrain-model --region=$REGION --limit=200"
    echo ""
else
    echo "⏭️  Skipping retraining test"
fi

echo ""
echo "============================================================"
echo "✅ TESTING COMPLETE"
echo "============================================================"
echo ""
echo "📋 Next Steps:"
echo "1. Check BigQuery metrics:"
echo "   ./test_pipeline.sh metrics"
echo ""
echo "2. Test endpoint predictions:"
echo "   ./test_pipeline.sh endpoint"
echo ""
echo "3. View function logs:"
echo "   gcloud functions logs read auto-retrain-model --region=$REGION --limit=200"
echo ""
echo "4. Setup automated scheduling:"
echo "   See DEPLOYMENT_AND_TESTING.md (Step 3)"
echo ""

