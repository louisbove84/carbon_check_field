"""
Deep Neural Network Model
=========================
TensorFlow/Keras DNN implementation with Bayesian hyperparameter tuning.
Uses Keras Tuner for automated hyperparameter optimization.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json

# TensorFlow imports - will be available after requirements update
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import keras_tuner as kt
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    kt = None
    # Import StandardScaler and LabelEncoder for type hints even without TF
    from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DNNModel(BaseModel):
    """
    Deep Neural Network classifier using TensorFlow/Keras.
    
    Architecture: Input -> Dense layers with ReLU -> Dropout -> Softmax output
    Supports Bayesian hyperparameter tuning via Keras Tuner.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        tuner_trials: int = 15,
        use_tuner: bool = True,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize DNN model.
        
        Args:
            hidden_layers: List of hidden layer sizes (e.g., [64, 32, 16])
            dropout_rate: Dropout rate for regularization
            learning_rate: Initial learning rate for Adam optimizer
            epochs: Maximum training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            tuner_trials: Number of Bayesian optimization trials
            use_tuner: Whether to use Keras Tuner for hyperparameter optimization
            random_state: Random seed for reproducibility
            feature_names: Optional list of feature names
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow and keras-tuner are required for DNNModel. "
                "Install with: pip install tensorflow keras-tuner"
            )
        
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.tuner_trials = tuner_trials
        self.use_tuner = use_tuner
        self.random_state = random_state
        self.feature_names = feature_names
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self._model = None  # keras.Model when TF is available
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._classes: Optional[List[str]] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._training_history: Optional[Dict] = None
    
    @property
    def model_type(self) -> str:
        return "dnn"
    
    @property
    def model_name(self) -> str:
        return "Deep Neural Network"
    
    def _build_model(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int],
        dropout_rate: float,
        learning_rate: float
    ):
        """Build and compile a Keras model."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_tuner_model(self, hp, input_dim: int, num_classes: int):
        """Build model for Keras Tuner with tunable hyperparameters."""
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(input_dim,)))
        
        # Tunable number of layers and units
        num_layers = hp.Int('num_layers', min_value=2, max_value=4, default=3)
        
        for i in range(num_layers):
            units = hp.Int(f'units_{i}', min_value=16, max_value=128, step=16, default=64-i*16)
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))
            
            dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
            model.add(layers.Dropout(dropout, name=f'dropout_{i}'))
        
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        
        # Tunable learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, 
                                  sampling='log', default=1e-3)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _run_tuner(
        self,
        X_train: np.ndarray,
        y_train_encoded: np.ndarray,
        X_val: np.ndarray,
        y_val_encoded: np.ndarray,
        input_dim: int,
        num_classes: int
    ) -> Tuple[Any, Dict[str, Any]]:  # Returns (keras.Model, hyperparameters)
        """Run Bayesian hyperparameter optimization."""
        logger.info(f"   Running Bayesian hyperparameter tuning ({self.tuner_trials} trials)...")
        
        # Create tuner
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: self._build_tuner_model(hp, input_dim, num_classes),
            objective='val_accuracy',
            max_trials=self.tuner_trials,
            directory='/tmp/keras_tuner',
            project_name='dnn_crop_classifier',
            overwrite=True
        )
        
        # Early stopping callback
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        )
        
        # Run search
        tuner.search(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        
        # Access hyperparameters using values dictionary for compatibility
        # with newer Keras Tuner versions
        hp_values = best_hp.values
        best_params = {
            'num_layers': hp_values.get('num_layers', 3),
            'dropout': hp_values.get('dropout', 0.3),
            'learning_rate': hp_values.get('learning_rate', 0.001),
        }
        
        # Add per-layer units
        num_layers = hp_values.get('num_layers', 3)
        for i in range(num_layers):
            best_params[f'units_{i}'] = hp_values.get(f'units_{i}', 64 - i * 16)
        
        logger.info(f"   Best hyperparameters: {best_params}")
        
        # Get best model
        best_model = tuner.get_best_models()[0]
        
        return best_model, best_params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the DNN model.
        
        If use_tuner=True, performs Bayesian hyperparameter optimization.
        Otherwise, trains with the provided hyperparameters.
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"   Samples: {len(X_train)}")
        logger.info(f"   Features: {X_train.shape[1]}")
        
        # Initialize preprocessing
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        
        # Fit and transform
        X_train_scaled = self._scaler.fit_transform(X_train)
        y_train_encoded = self._label_encoder.fit_transform(y_train)
        
        self._classes = list(self._label_encoder.classes_)
        input_dim = X_train_scaled.shape[1]
        num_classes = len(self._classes)
        
        logger.info(f"   Classes: {self._classes}")
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler.transform(X_val)
            y_val_encoded = self._label_encoder.transform(y_val)
        else:
            # Use 20% of training data for validation
            from sklearn.model_selection import train_test_split
            X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded = train_test_split(
                X_train_scaled, y_train_encoded, 
                test_size=0.2, random_state=self.random_state, stratify=y_train_encoded
            )
        
        # Train model
        if self.use_tuner:
            self._model, self._best_params = self._run_tuner(
                X_train_scaled, y_train_encoded,
                X_val_scaled, y_val_encoded,
                input_dim, num_classes
            )
        else:
            logger.info(f"   Hyperparameters: layers={self.hidden_layers}, dropout={self.dropout_rate}, lr={self.learning_rate}")
            
            self._model = self._build_model(
                input_dim, num_classes,
                self.hidden_layers, self.dropout_rate, self.learning_rate
            )
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
            
            history = self._model.fit(
                X_train_scaled, y_train_encoded,
                validation_data=(X_val_scaled, y_val_encoded),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            self._training_history = history.history
            self._best_params = self.get_params()
        
        # Compute metrics
        train_pred = self.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        val_pred_encoded = np.argmax(self._model.predict(X_val_scaled, verbose=0), axis=1)
        val_pred = self._label_encoder.inverse_transform(val_pred_encoded)
        val_acc = np.mean(val_pred == (y_val if y_val is not None else self._label_encoder.inverse_transform(y_val_encoded)))
        
        metrics = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
        }
        
        if self._best_params:
            metrics['best_params'] = self._best_params
        
        logger.info(f"   Train accuracy: {train_acc:.2%}")
        logger.info(f"   Val accuracy: {val_acc:.2%}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict(X_scaled, verbose=0)
        pred_encoded = np.argmax(probs, axis=1)
        return self._label_encoder.inverse_transform(pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distributions over classes."""
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled, verbose=0)
    
    def save(self, path: str) -> None:
        """
        Save model to the specified directory.
        
        Creates:
            {path}/model_dnn.keras - Native Keras format model
            {path}/model_dnn_meta.pkl - Scaler, encoder, and metadata
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model in native Keras format (.keras extension required)
        model_path = os.path.join(path, 'model_dnn.keras')
        self._model.save(model_path)
        logger.info(f"   Saved DNN model to {model_path}")
        
        # Save preprocessing and metadata
        meta_path = os.path.join(path, 'model_dnn_meta.pkl')
        meta_data = {
            'scaler': self._scaler,
            'label_encoder': self._label_encoder,
            'classes': self._classes,
            'feature_names': self.feature_names,
            'params': self.get_params(),
            'best_params': self._best_params,
            'training_history': self._training_history
        }
        joblib.dump(meta_data, meta_path)
        logger.info(f"   Saved DNN metadata to {meta_path}")
    
    def load(self, path: str) -> None:
        """
        Load model from the specified directory.
        
        Expects:
            {path}/model_dnn.keras - Native Keras format model (preferred)
            {path}/model_dnn/ - Legacy TensorFlow SavedModel (fallback)
            {path}/model_dnn_meta.pkl - Scaler, encoder, and metadata
        """
        model_path = os.path.join(path, 'model_dnn.keras')
        model_dir_legacy = os.path.join(path, 'model_dnn')
        meta_path = os.path.join(path, 'model_dnn_meta.pkl')
        
        # Try native Keras format first, fall back to legacy SavedModel
        if os.path.exists(model_path):
            self._model = keras.models.load_model(model_path)
            logger.info(f"   Loaded DNN model from {model_path}")
        elif os.path.exists(model_dir_legacy):
            self._model = keras.models.load_model(model_dir_legacy)
            logger.info(f"   Loaded DNN model from {model_dir_legacy} (legacy format)")
        else:
            raise FileNotFoundError(f"DNN model not found at {model_path} or {model_dir_legacy}")
        
        # Load metadata
        if os.path.exists(meta_path):
            meta_data = joblib.load(meta_path)
            self._scaler = meta_data['scaler']
            self._label_encoder = meta_data['label_encoder']
            self._classes = meta_data.get('classes')
            self.feature_names = meta_data.get('feature_names')
            self._best_params = meta_data.get('best_params')
            self._training_history = meta_data.get('training_history')
            
            # Restore hyperparameters
            params = meta_data.get('params', {})
            self.hidden_layers = params.get('hidden_layers', self.hidden_layers)
            self.dropout_rate = params.get('dropout_rate', self.dropout_rate)
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            
            logger.info(f"   Loaded DNN metadata from {meta_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'early_stopping_patience': self.early_stopping_patience,
            'tuner_trials': self.tuner_trials,
            'use_tuner': self.use_tuner
        }
    
    def get_classes(self) -> Optional[List[str]]:
        """Return class labels."""
        return self._classes
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        DNN does not provide built-in feature importance.
        Returns None (could implement gradient-based importance in future).
        """
        return None
    
    def get_training_history(self) -> Optional[Dict]:
        """Return training history (loss, accuracy curves)."""
        return self._training_history
    
    @property
    def keras_model(self):
        """Access the underlying Keras model."""
        return self._model
