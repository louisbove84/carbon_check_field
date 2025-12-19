#!/usr/bin/env python3
"""
Simple script to run only the data collection step.
This clears BigQuery and collects fresh balanced data with buffer logic.
"""

import sys
import os
import logging
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import orchestrator functions
from orchestrator import load_config, export_earth_engine_data

def main():
    """Run data collection only."""
    logger.info("=" * 70)
    logger.info("🌍 EARTH ENGINE DATA COLLECTION ONLY")
    logger.info("=" * 70)
    logger.info("This will:")
    logger.info("  1. Clear existing BigQuery data (if overwrite_table: true)")
    logger.info("  2. Collect balanced samples with 150m buffer")
    logger.info("  3. Export to BigQuery")
    logger.info("")
    
    try:
        # Run data collection
        result = export_earth_engine_data()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ DATA COLLECTION COMPLETE")
        logger.info("=" * 70)
        logger.info(json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
