"""
ML Pipeline Orchestrator
========================
Main entry point for the automated ML pipeline.

Steps:
1. Collect data from Earth Engine → BigQuery
2. Retrain model with new data
3. Evaluate and deploy if quality gates pass

Usage:
    python main.py              # Run full pipeline
    python main.py --step=1     # Run only data collection
    python main.py --step=2     # Run only retraining
    python main.py --step=3     # Run only evaluation
"""

import logging
import sys
import argparse
from datetime import datetime

# Import pipeline steps
import collect_data
import retrain
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(step=None):
    """
    Run the complete ML pipeline or a specific step.
    
    Args:
        step: Optional step number (1, 2, or 3). If None, runs all steps.
    
    Returns:
        dict with pipeline results
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("🌾 AUTOMATED ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Started at: {start_time.isoformat()}")
    logger.info("")
    
    results = {
        'status': 'success',
        'start_time': start_time.isoformat(),
        'steps_completed': []
    }
    
    try:
        # Step 1: Data Collection
        if step is None or step == 1:
            logger.info("=" * 70)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 70)
            collection_result = collect_data.collect_training_data()
            results['data_collection'] = collection_result
            results['steps_completed'].append('data_collection')
            
            if collection_result['status'] != 'success':
                raise Exception(f"Data collection failed: {collection_result.get('error')}")
            
            logger.info(f"✅ Collected {collection_result['samples_collected']} samples")
            logger.info("")
        
        # Step 2: Model Retraining
        if step is None or step == 2:
            logger.info("=" * 70)
            logger.info("STEP 2: MODEL RETRAINING")
            logger.info("=" * 70)
            retrain_result = retrain.retrain_model()
            results['retraining'] = retrain_result
            results['steps_completed'].append('retraining')
            
            if retrain_result['status'] != 'success':
                raise Exception(f"Retraining failed: {retrain_result.get('error')}")
            
            logger.info(f"✅ Model trained with {retrain_result['training_samples']} samples")
            logger.info(f"   Accuracy: {retrain_result['metrics']['test_accuracy']:.2%}")
            logger.info("")
        
        # Step 3: Evaluation & Deployment
        if step is None or step == 3:
            logger.info("=" * 70)
            logger.info("STEP 3: EVALUATION & DEPLOYMENT")
            logger.info("=" * 70)
            eval_result = evaluate.evaluate_and_deploy()
            results['evaluation'] = eval_result
            results['steps_completed'].append('evaluation')
            
            if eval_result['status'] != 'success':
                raise Exception(f"Evaluation failed: {eval_result.get('error')}")
            
            if eval_result['deployed']:
                logger.info("✅ New model DEPLOYED to production")
            else:
                logger.warning("⛔ New model BLOCKED - did not pass quality gates")
                for reason in eval_result.get('reasons', []):
                    logger.warning(f"   {reason}")
            logger.info("")
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds() / 60
        results['duration_minutes'] = round(duration, 2)
        results['end_time'] = datetime.now().isoformat()
        
        logger.info("=" * 70)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Steps completed: {', '.join(results['steps_completed'])}")
        logger.info("")
        
        return results
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.error(f"❌ Pipeline failed after {duration:.1f} minutes")
        logger.error(f"Error: {str(e)}", exc_info=True)
        
        return {
            'status': 'error',
            'error': str(e),
            'duration_minutes': round(duration, 2),
            'steps_completed': results['steps_completed']
        }


# Flask app for Cloud Run
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def run():
    """HTTP endpoint for Cloud Run."""
    # Get step parameter if provided
    step = request.args.get('step', None)
    if step:
        step = int(step)
    
    # Run pipeline
    result = run_pipeline(step=step)
    
    # Return JSON response
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(result), status_code


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # For local testing
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], 
                       help='Run specific step only (1=collect, 2=retrain, 3=evaluate)')
    parser.add_argument('--port', type=int, default=8080, 
                       help='Port for Flask server (default: 8080)')
    args = parser.parse_args()
    
    if '--step' in sys.argv:
        # Run specific step directly (for testing)
        result = run_pipeline(step=args.step)
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        # Start Flask server
        app.run(host='0.0.0.0', port=args.port)
