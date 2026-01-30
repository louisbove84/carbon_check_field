"""
Base Model Interface
====================
Abstract base class that all models must implement.
Enables a pluggable model registry pattern for easy addition of new model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models in the pipeline.
    
    All models must implement these methods to ensure consistent
    behavior across training, inference, and evaluation.
    """
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return the model type identifier (e.g., 'rf', 'dnn').
        Used for TensorBoard logging prefixes and model storage paths.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return a human-readable model name (e.g., 'Random Forest', 'Deep Neural Network').
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (used by DNN for early stopping)
            y_val: Optional validation labels
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training metrics (e.g., {'train_accuracy': 0.95})
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions for the input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability distributions over classes.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probability distributions of shape (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model artifacts to the specified path.
        
        Args:
            path: Directory path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model artifacts from the specified path.
        
        Args:
            path: Directory path containing the model artifacts
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Return the model's hyperparameters as a dictionary.
        Used for logging to TensorBoard and metrics storage.
        
        Returns:
            Dict of hyperparameter names to values
        """
        pass
    
    def get_classes(self) -> Optional[List[str]]:
        """
        Return the class labels if available.
        
        Returns:
            List of class labels or None if not available
        """
        return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance scores if available.
        Only applicable to models that support feature importance (e.g., RF).
        
        Returns:
            Dict mapping feature names to importance scores, or None
        """
        return None
