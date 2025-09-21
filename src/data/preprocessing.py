""
Data preprocessing module for SmartRetail Analytics.
Handles data cleaning, transformation, and feature engineering.
"""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize the preprocessor with default settings."""
        self.date_columns = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
            'shipping_limit_date',
            'review_creation_date',
            'review_answer_timestamp'
        ]
        
    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string dates to datetime objects.
        
        Args:
            df: Input DataFrame with date columns
            
        Returns:
            DataFrame with converted date columns
        """
        df = df.copy()
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop', **kwargs) -> pd.DataFrame:
        """Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Strategy to handle missing values ('drop', 'fill', 'interpolate')
            **kwargs: Additional arguments for the strategy
                - fill_value: Value to use for 'fill' strategy
                - method: Method to use for 'interpolate' strategy
                
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy == 'drop':
            return df.dropna(**kwargs)
        elif strategy == 'fill':
            fill_value = kwargs.get('fill_value', 0)
            return df.fillna(fill_value)
        elif strategy == 'interpolate':
            method = kwargs.get('method', 'linear')
            return df.interpolate(method=method, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def add_date_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract date-based features from a datetime column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the datetime column
            
        Returns:
            DataFrame with added date features
        """
        df = df.copy()
        
        if date_column not in df.columns:
            return df
            
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
        df[f'{date_column}_weekofyear'] = df[date_column].dt.isocalendar().week
        df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df
    
    def preprocess_orders(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the orders dataset.
        
        Args:
            orders_df: Raw orders DataFrame
            
        Returns:
            Preprocessed orders DataFrame
        """
        # Convert dates
        orders_df = self.convert_dates(orders_df)
        
        # Calculate delivery time metrics
        if all(col in orders_df.columns for col in ['order_delivered_customer_date', 'order_purchase_timestamp']):
            orders_df['delivery_time_days'] = (
                orders_df['order_delivered_customer_date'] - 
                orders_df['order_purchase_timestamp']
            ).dt.total_seconds() / (24 * 3600)
        
        # Add date features
        orders_df = self.add_date_features(orders_df, 'order_purchase_timestamp')
        
        return orders_df
    
    def preprocess_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the products dataset.
        
        Args:
            products_df: Raw products DataFrame
            
        Returns:
            Preprocessed products DataFrame
        """
        products_df = products_df.copy()
        
        # Handle missing values in product dimensions
        for col in ['product_length_cm', 'product_height_cm', 'product_width_cm', 'product_weight_g']:
            if col in products_df.columns:
                products_df[col] = products_df[col].fillna(products_df[col].median())
        
        # Calculate product volume
        if all(dim in products_df.columns for dim in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
            products_df['product_volume_cm3'] = (
                products_df['product_length_cm'] * 
                products_df['product_height_cm'] * 
                products_df['product_width_cm']
            )
        
        return products_df
    
    def preprocess_order_items(self, order_items_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the order items dataset.
        
        Args:
            order_items_df: Raw order items DataFrame
            
        Returns:
            Preprocessed order items DataFrame
        """
        order_items_df = order_items_df.copy()
        
        # Convert shipping limit date to datetime
        if 'shipping_limit_date' in order_items_df.columns:
            order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'])
        
        # Calculate total price if not present
        if all(col in order_items_df.columns for col in ['price', 'freight_value']):
            order_items_df['total_price'] = order_items_df['price'] + order_items_df['freight_value']
        
        return order_items_df
