"""
ML Pipeline Orchestrator - HTTP Endpoint
=========================================
Lightweight Flask app that triggers the ML pipeline.
"""

import logging
from flask import Flask, jsonify, request
import orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def run():
    """Trigger the ML pipeline."""
    result = orchestrator.run_pipeline()
    status_code = 200 if result['status'] == 'success' else 500
    return jsonify(result), status_code


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=8080)

