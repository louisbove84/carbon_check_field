"""
Cloud Function Entry Point for Automated Model Retraining
===========================================================
This file imports and exposes the retrain_model function
for Cloud Functions deployment.
"""

from auto_retrain_model import retrain_model

# Cloud Functions will call this function
# It's imported from auto_retrain_model.py
__all__ = ['retrain_model']

