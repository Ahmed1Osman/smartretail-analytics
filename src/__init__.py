"""
SmartRetail Analytics - A comprehensive retail analytics solution for sales forecasting and inventory optimization.
"""

__version__ = '0.1.0'

# Import main components to make them available at the package level
from .data.collection import DataCollector
from .data.preprocessing import DataPreprocessor
from .features.feature_engineering import FeatureEngineer
from .models.train import ModelTrainer
from .models.predict import Predictor
from .visualization.plots import Plotter

__all__ = [
    'DataCollector',
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'Predictor',
    'Plotter',
]
