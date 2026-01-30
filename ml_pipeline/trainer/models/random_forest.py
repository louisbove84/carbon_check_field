"""
Random Forest Model
===================
Random Forest classifier implementation using scikit-learn.
Wraps sklearn Pipeline with StandardScaler + RandomForestClassifier.
"""

import os
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with StandardScaler preprocessing.
    
    This is the primary explainable model in the pipeline.
    Provides feature importance scores and probability estimates.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)
            feature_names: Optional list of feature names for importance tracking
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_names = feature_names
        
        self._pipeline: Optional[Pipeline] = None
        self._classes: Optional[List[str]] = None
    
    @property
    def model_type(self) -> str:
        return "rf"
    
    @property
    def model_name(self) -> str:
        return "Random Forest"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Note: X_val and y_val are not used by RF (no early stopping),
        but accepted for API consistency.
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"   Samples: {len(X_train)}")
        logger.info(f"   Features: {X_train.shape[1]}")
        logger.info(f"   Hyperparameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}")
        
        # Store class labels
        self._classes = list(np.unique(y_train))
        
        # Create pipeline
        self._pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ))
        ])
        
        # Train
        self._pipeline.fit(X_train, y_train)
        
        # Compute training metrics
        train_pred = self._pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': float(train_acc),
        }
        
        # If validation data provided, compute validation accuracy
        if X_val is not None and y_val is not None:
            val_pred = self._pipeline.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = float(val_acc)
        
        logger.info(f"   Train accuracy: {train_acc:.2%}")
        if 'val_accuracy' in metrics:
            logger.info(f"   Val accuracy: {metrics['val_accuracy']:.2%}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        if self._pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self._pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distributions over classes."""
        if self._pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self._pipeline.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save model to the specified directory.
        
        Creates: {path}/model_rf.pkl
        """
        if self._pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model_rf.pkl')
        
        # Save pipeline and metadata
        save_data = {
            'pipeline': self._pipeline,
            'classes': self._classes,
            'feature_names': self.feature_names,
            'params': self.get_params()
        }
        
        joblib.dump(save_data, model_path, protocol=4)
        logger.info(f"   Saved RF model to {model_path}")
    
    def load(self, path: str) -> None:
        """
        Load model from the specified directory.
        
        Expects: {path}/model_rf.pkl
        """
        model_path = os.path.join(path, 'model_rf.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        save_data = joblib.load(model_path)
        
        self._pipeline = save_data['pipeline']
        self._classes = save_data.get('classes')
        self.feature_names = save_data.get('feature_names')
        
        # Restore hyperparameters
        params = save_data.get('params', {})
        self.n_estimators = params.get('n_estimators', self.n_estimators)
        self.max_depth = params.get('max_depth', self.max_depth)
        self.min_samples_split = params.get('min_samples_split', self.min_samples_split)
        
        logger.info(f"   Loaded RF model from {model_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }
    
    def get_classes(self) -> Optional[List[str]]:
        """Return class labels."""
        return self._classes
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance scores.
        
        Returns dict mapping feature names to importance values.
        """
        if self._pipeline is None:
            return None
        
        classifier = self._pipeline.named_steps['classifier']
        importances = classifier.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances.tolist()))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
    
    @property
    def pipeline(self) -> Optional[Pipeline]:
        """Access the underlying sklearn Pipeline."""
        return self._pipeline
