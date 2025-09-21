""
Model training and prediction module for SmartRetail Analytics.

This module provides tools for training, evaluating, and making predictions with machine learning models.
"""

from .train import ModelTrainer
from .predict import Predictor

__all__ = ['ModelTrainer', 'Predictor']
