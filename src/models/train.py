"""Model training module for SmartRetail Analytics.

This module provides functionality for training, evaluating, and persisting machine learning models
for sales forecasting and retail analytics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class ModelTrainer:
    """Handles training and evaluation of sales forecasting models."""

    def __init__(self, model_dir: str = 'models', random_state: int = 42):
        """Initialize the model trainer.

        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str,
                    test_size: float = 0.2,
                    time_based_split: bool = True,
                    date_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for model training.
        
        Args:
            df: Input DataFrame with features and target
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            time_based_split: Whether to use time-based train-test split
            date_column: Name of the date column (required if time_based_split is True)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]
        
        if time_based_split and date_column is not None:
            # Sort by date and split based on time
            df = df.sort_values(by=date_column)
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # Random split
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state
            )
        
        # Separate features and target
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[self.feature_columns]
        y_test = test_df[target_column]
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_features(self, 
                          X_train: pd.DataFrame, 
                          X_test: Optional[pd.DataFrame] = None,
                          numeric_features: Optional[List[str]] = None,
                          categorical_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features for model training.
        
        Args:
            X_train: Training features
            X_test: Optional test features
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        # If feature lists not provided, infer from data
        if numeric_features is None:
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Define preprocessing for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit and transform training data
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform test data if provided
        X_test_processed = preprocessor.transform(X_test) if X_test is not None else None
        
        # Save the preprocessor for later use
        self.preprocessor = preprocessor
        
        return X_train_processed, X_test_processed
    
    def train_models(self, 
                    X_train: Union[pd.DataFrame, np.ndarray], 
                    y_train: Union[pd.Series, np.ndarray],
                    models: Optional[Dict] = None) -> Dict:
        """Train multiple models and store results.
        
        Args:
            X_train: Training features
            y_train: Training target
            models: Dictionary of model names and initialized models
            
        Returns:
            Dictionary of trained models
        """
        if models is None:
            # Default models to train
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state)
            }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):  # For linear models
                self.feature_importance[name] = model.coef_
        
        return self.models
    
    def evaluate_models(self, 
                       X_test: Union[pd.DataFrame, np.ndarray], 
                       y_test: Union[pd.Series, np.ndarray],
                       metrics: Optional[Dict] = None) -> Dict:
        """Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            metrics: Dictionary of metric names and functions
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        if metrics is None:
            metrics = {
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score
            }
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = {}
            
            for metric_name, metric_func in metrics.items():
                try:
                    score = metric_func(y_test, y_pred)
                    results[name][metric_name] = score
                except Exception as e:
                    print(f"Error calculating {metric_name} for {name}: {str(e)}")
                    results[name][metric_name] = None
        
        self.metrics = results
        return results
    
    def cross_validate(self, 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      cv: int = 5,
                      scoring: str = 'neg_mean_squared_error') -> Dict:
        """Perform cross-validation for all trained models.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary of cross-validation results
        """
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                cv_results[name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'all_scores': scores.tolist()
                }
            except Exception as e:
                print(f"Error in cross-validation for {name}: {str(e)}")
                cv_results[name] = None
        
        return cv_results
    
    def save_model(self, 
                  model_name: str, 
                  model,
                  metadata: Optional[Dict] = None) -> str:
        """Save a trained model to disk.
        
        Args:
            model_name: Name to save the model as
            model: Trained model object
            metadata: Additional metadata to save with the model
            
        Returns:
            Path to the saved model
        """
        # Create model directory if it doesn't exist
        model_path = self.model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_file = model_path / 'model.joblib'
        joblib.dump(model, model_file)
        
        # Save metadata if provided
        if metadata is not None:
            metadata_file = model_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save feature columns if available
        if hasattr(self, 'feature_columns'):
            features_file = model_path / 'feature_columns.json'
            with open(features_file, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
        
        return str(model_path)
    
    def load_model(self, model_name: str):
        """Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model and metadata
        """
        model_path = self.model_dir / model_name
        
        # Load the model
        model_file = model_path / 'model.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        model = joblib.load(model_file)
        
        # Load metadata if it exists
        metadata_file = model_path / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Load feature columns if they exist
        feature_columns = None
        features_file = model_path / 'feature_columns.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_columns = json.load(f)
        
        return model, metadata, feature_columns
    
    def train_pipeline(self, 
                      df: pd.DataFrame, 
                      target_column: str,
                      date_column: Optional[str] = None,
                      test_size: float = 0.2,
                      time_based_split: bool = True) -> Dict:
        """Complete training pipeline from data preparation to model evaluation.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            date_column: Name of the date column (required for time-based split)
            test_size: Fraction of data to use for testing
            time_based_split: Whether to use time-based train-test split
            
        Returns:
            Dictionary containing training results and model information
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            df=df,
            target_column=target_column,
            test_size=test_size,
            time_based_split=time_based_split,
            date_column=date_column
        )
        
        # Preprocess features
        X_train_processed, X_test_processed = self.preprocess_features(X_train, X_test)
        
        # Train models
        models = self.train_models(X_train_processed, y_train)
        
        # Evaluate models
        metrics = self.evaluate_models(X_test_processed, y_test)
        
        # Perform cross-validation
        cv_results = self.cross_validate(X_train_processed, y_train)
        
        # Save the best model
        best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
        best_model = models[best_model_name]
        
        # Create metadata
        metadata = {
            'model_name': best_model_name,
            'target_column': target_column,
            'feature_columns': self.feature_columns,
            'metrics': metrics[best_model_name],
            'cv_results': cv_results.get(best_model_name, {}),
            'training_date': datetime.now().isoformat(),
            'model_type': type(best_model).__name__
        }
        
        # Save the best model
        model_path = self.save_model(
            model_name=f"{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model=best_model,
            metadata=metadata
        )
        
        return {
            'best_model': best_model_name,
            'metrics': metrics,
            'cv_results': cv_results,
            'model_path': model_path,
            'feature_importance': self.feature_importance.get(best_model_name, [])
        }
