"""
Skew Audit Job - HTTP Endpoint
==============================
Flask app that triggers the data skew audit.
Designed to be called by Cloud Scheduler monthly.
"""

import logging
from flask import Flask, jsonify, request
import numpy as np
import skew_detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)


def _to_jsonable(value):
    """Convert numpy types to JSON serializable types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@app.route('/', methods=['POST'])
def run():
    """Trigger the skew audit pipeline."""
    result = skew_detector.run_skew_audit()
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(_to_jsonable(result)), status_code


@app.route('/audit', methods=['POST'])
def audit():
    """Alias for the main endpoint."""
    result = skew_detector.run_skew_audit()
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(_to_jsonable(result)), status_code


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ml-skew-audit'}), 200


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=8080)
