""
Prediction module for SmartRetail Analytics.
Handles making predictions using trained models.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

class Predictor:
    """Handles making predictions using trained models."""
    
    def __init__(self, model_dir: str = 'models'):
        """Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.metadata = {}
        self.feature_columns = []
        self.scaler = None
        
    def load_latest_model(self, model_name: Optional[str] = None):
        """Load the most recent model from the model directory.
        
        Args:
            model_name: Optional specific model name to load
            
        Returns:
            Loaded model and metadata
        """
        if model_name is None:
            # Find all model directories and get the most recent one
            model_dirs = list(self.model_dir.glob('*'))
            if not model_dirs:
                raise FileNotFoundError(f"No models found in {self.model_dir}")
                
            # Sort by modification time (newest first)
            model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = model_dirs[0]
        else:
            model_path = self.model_dir / model_name
            
        # Load the model and metadata
        self.model, self.metadata, self.feature_columns = self._load_model(model_path)
        
        # Load the preprocessor if it exists
        preprocessor_path = model_path / 'preprocessor.joblib'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            
        return self.model, self.metadata
    
    def _load_model(self, model_path: Path):
        """Load a model and its metadata from disk."""
        # Load the model
        model_file = model_path / 'model.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = joblib.load(model_file)
        
        # Load metadata
        metadata = {}
        metadata_file = model_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
        # Load feature columns
        feature_columns = []
        features_file = model_path / 'feature_columns.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_columns = json.load(f)
                
        return model, metadata, feature_columns
    
    def preprocess_input(self, input_data: Union[dict, pd.DataFrame]) -> np.ndarray:
        """Preprocess input data for prediction.
        
        Args:
            input_data: Input data as a dictionary or DataFrame
            
        Returns:
            Preprocessed features as a numpy array
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Reorder columns to match training data
        df = df[self.feature_columns]
        
        # Apply preprocessing if available
        if hasattr(self, 'preprocessor'):
            return self.preprocessor.transform(df)
        return df.values
    
    def predict(self, input_data: Union[dict, pd.DataFrame], 
               return_proba: bool = False) -> Union[float, np.ndarray, dict]:
        """Make a prediction using the loaded model.
        
        Args:
            input_data: Input data for prediction
            return_proba: Whether to return class probabilities (for classification)
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_latest_model() first.")
            
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make prediction
        if hasattr(self.model, 'predict_proba') and return_proba:
            predictions = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)
            
        # For single input, return a scalar if possible
        if isinstance(input_data, dict) and len(predictions) == 1:
            return predictions[0]
            
        return predictions
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from the model if available.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame of feature importances
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_latest_model() first.")
            
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")
            
        # Get feature names (use column names if available, otherwise use indices)
        if hasattr(self, 'feature_columns') and self.feature_columns:
            feature_names = self.feature_columns
        else:
            n_features = len(self.model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
        # Create DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        
        # Sort by importance and return top N
        return importance_df.sort_values('importance', ascending=False).head(top_n)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        if not self.metadata:
            return {"status": "No model loaded"}
            
        return {
            "model_type": type(self.model).__name__,
            "training_date": self.metadata.get('training_date', 'Unknown'),
            "metrics": self.metadata.get('metrics', {}),
            "target_column": self.metadata.get('target_column', 'Unknown'),
            "num_features": len(self.feature_columns) if self.feature_columns else 0
        }
    
    def batch_predict(self, 
                     input_data: Union[List[dict], pd.DataFrame],
                     batch_size: int = 1000) -> list:
        """Make predictions on a batch of input data.
        
        Args:
            input_data: List of input dictionaries or a DataFrame
            batch_size: Number of samples per batch
            
        Returns:
            List of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_latest_model() first.")
            
        # Convert to DataFrame if needed
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()
            
        # Process in batches to avoid memory issues
        predictions = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_preds = self.predict(batch)
            predictions.extend(batch_preds)
            
        return predictions
    
    def evaluate(self, 
                X_test: Union[pd.DataFrame, np.ndarray], 
                y_test: Union[pd.Series, np.ndarray],
                metrics: Optional[dict] = None) -> dict:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            metrics: Dictionary of metric names and functions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_latest_model() first.")
            
        if metrics is None:
            from sklearn.metrics import (
                mean_absolute_error, 
                mean_squared_error, 
                r2_score,
                explained_variance_score
            )
            
            metrics = {
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score,
                'explained_variance': explained_variance_score
            }
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        results = {}
        for name, metric_func in metrics.items():
            try:
                results[name] = metric_func(y_test, y_pred)
            except Exception as e:
                print(f"Error calculating {name}: {str(e)}")
                results[name] = None
                
        return results
