"""
Model Registry
==============
Pluggable model registry for easy addition of new model types.

Usage:
    from models import MODEL_REGISTRY, get_model
    
    # Get all available models
    for model_type in MODEL_REGISTRY:
        model = get_model(model_type, **config)
        
    # Get specific model
    rf_model = get_model('rf', n_estimators=100)
    dnn_model = get_model('dnn', hidden_layers=[64, 32, 16])
"""

from typing import Dict, Type, Any

from .base_model import BaseModel
from .random_forest import RandomForestModel
from .dnn_model import DNNModel

# Registry mapping model_type to model class
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'rf': RandomForestModel,
    'dnn': DNNModel,
}


def get_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: One of the registered model types ('rf', 'dnn')
        **kwargs: Model-specific configuration parameters
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_type is not registered
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(**kwargs)


def get_all_model_types() -> list:
    """Return list of all registered model types."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BaseModel',
    'RandomForestModel',
    'DNNModel',
    'MODEL_REGISTRY',
    'get_model',
    'get_all_model_types'
]
