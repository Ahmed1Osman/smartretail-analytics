""
Data collection module for SmartRetail Analytics.
Handles data loading, validation, and initial processing.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

class DataCollector:
    """Handles data collection from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize with data directory.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a dataset from CSV file.
        
        Args:
            filename: Name of the CSV file (without extension)
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / f"{filename}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
        """Validate the loaded data.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, message)
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"        
        if df.empty:
            return False, "DataFrame is empty"
            
        return True, "Data validation passed"
    
    def get_available_datasets(self) -> Dict[str, str]:
        """Get information about available datasets."""
        return {
            "primary": {
                "name": "Brazilian E-Commerce Public Dataset by Olist",
                "files": [
                    "olist_customers_dataset",
                    "olist_geolocation_dataset",
                    "olist_order_items_dataset",
                    "olist_order_payments_dataset",
                    "olist_order_reviews_dataset",
                    "olist_orders_dataset",
                    "olist_products_dataset",
                    "olist_sellers_dataset",
                ]
            },
            "external": {
                "name": "External Data Sources",
                "files": [
                    "economic_indicators",
                    "weather_data",
                    "holiday_calendar"
                ]
            }
        }
