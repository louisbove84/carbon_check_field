"""
ML Pipeline Orchestrator - HTTP Endpoint
=========================================
Lightweight Flask app that triggers the ML pipeline.
"""

import logging
from flask import Flask, jsonify, request
import numpy as np
import orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


@app.route('/', methods=['POST'])
def run():
    """Trigger the complete ML pipeline (data collection + training + deployment)."""
    result = orchestrator.run_pipeline()
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(_to_jsonable(result)), status_code


@app.route('/train', methods=['POST'])
def train():
    """Trigger only training (skip data collection and deployment)."""
    result = orchestrator.run_training_only()
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(_to_jsonable(result)), status_code


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=8080)

