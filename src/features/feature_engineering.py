""
Feature engineering module for SmartRetail Analytics.
Handles creation of derived features and transformations.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    """Handles feature engineering for the retail analytics pipeline."""
    
    def __init__(self):
        """Initialize the feature engineer with default settings."""
        self.customer_features = [
            'total_orders',
            'total_spent',
            'avg_order_value',
            'days_since_last_order',
            'order_frequency',
            'preferred_category',
            'preferred_payment_method'
        ]
        
        self.product_features = [
            'product_volume_cm3',
            'product_weight_g',
            'price_per_volume',
            'avg_rating',
            'review_count',
            'days_since_last_sale',
            'sales_velocity_7d',
            'sales_velocity_30d'
        ]
    
    def create_customer_features(self, 
                              orders_df: pd.DataFrame, 
                              order_items_df: pd.DataFrame,
                              payments_df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level features.
        
        Args:
            orders_df: Processed orders DataFrame
            order_items_df: Processed order items DataFrame
            payments_df: Processed payments DataFrame
            
        Returns:
            DataFrame with customer features
        """
        # Calculate basic customer metrics
        customer_metrics = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'order_purchase_timestamp': ['max', 'min']
        }).reset_index()
        
        # Flatten multi-index columns
        customer_metrics.columns = ['customer_id', 'total_orders', 'last_order_date', 'first_order_date']
        
        # Calculate days since last order
        max_date = orders_df['order_purchase_timestamp'].max()
        customer_metrics['days_since_last_order'] = (
            max_date - customer_metrics['last_order_date']
        ).dt.days
        
        # Calculate customer lifetime in days
        customer_metrics['customer_lifetime_days'] = (
            customer_metrics['last_order_date'] - 
            customer_metrics['first_order_date']
        ).dt.days
        
        # Calculate order frequency (orders per day)
        customer_metrics['order_frequency'] = (
            customer_metrics['total_orders'] / 
            (customer_metrics['customer_lifetime_days'] + 1)  # +1 to avoid division by zero
        )
        
        # Calculate total spent per customer
        if 'order_id' in order_items_df.columns and 'total_price' in order_items_df.columns:
            customer_spend = order_items_df.groupby('order_id')['total_price'].sum().reset_index()
            customer_spend = customer_spend.merge(
                orders_df[['order_id', 'customer_id']], 
                on='order_id', 
                how='left'
            )
            
            customer_spend = customer_spend.groupby('customer_id')['total_price'].agg(
                ['sum', 'mean']
            ).reset_index()
            customer_spend.columns = ['customer_id', 'total_spent', 'avg_order_value']
            
            # Merge with customer metrics
            customer_metrics = customer_metrics.merge(customer_spend, on='customer_id', how='left')
        
        # Add preferred payment method
        if 'payment_type' in payments_df.columns:
            payment_pref = payments_df.groupby('customer_id')['payment_type'].agg(
                lambda x: x.value_counts().index[0]
            ).reset_index()
            payment_pref.columns = ['customer_id', 'preferred_payment_method']
            
            customer_metrics = customer_metrics.merge(payment_pref, on='customer_id', how='left')
        
        return customer_metrics
    
    def create_product_features(self, 
                              products_df: pd.DataFrame, 
                              order_items_df: pd.DataFrame,
                              reviews_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create product-level features.
        
        Args:
            products_df: Processed products DataFrame
            order_items_df: Processed order items DataFrame
            reviews_df: Optional reviews DataFrame
            
        Returns:
            DataFrame with product features
        """
        product_features = products_df.copy()
        
        # Calculate sales metrics
        sales_metrics = order_items_df.groupby('product_id').agg({
            'price': ['count', 'mean', 'sum'],
            'freight_value': 'mean',
            'order_item_id': 'count'
        }).reset_index()
        
        # Flatten multi-index columns
        sales_metrics.columns = [
            'product_id', 
            'sales_count', 
            'avg_price', 
            'total_revenue',
            'avg_freight',
            'total_items_sold'
        ]
        
        # Merge with product features
        product_features = product_features.merge(sales_metrics, on='product_id', how='left')
        
        # Add review metrics if available
        if reviews_df is not None and 'review_score' in reviews_df.columns:
            review_metrics = reviews_df.groupby('product_id')['review_score'].agg(
                ['mean', 'count']
            ).reset_index()
            review_metrics.columns = ['product_id', 'avg_rating', 'review_count']
            
            product_features = product_features.merge(review_metrics, on='product_id', how='left')
        
        # Calculate price per volume if dimensions are available
        if all(dim in product_features.columns for dim in ['product_volume_cm3', 'avg_price']):
            product_features['price_per_volume'] = (
                product_features['avg_price'] / 
                (product_features['product_volume_cm3'] + 1e-6)  # Avoid division by zero
            )
        
        # Add sales velocity (7-day and 30-day)
        if 'order_purchase_timestamp' in order_items_df.columns:
            # This is a simplified example - in practice, you'd need the actual dates
            # and would calculate rolling windows based on the date
            product_features['sales_velocity_7d'] = product_features['sales_count'] / 30  # Placeholder
            product_features['sales_velocity_30d'] = product_features['sales_count'] / 90  # Placeholder
        
        return product_features
    
    def create_time_series_features(self, 
                                  df: pd.DataFrame, 
                                  date_column: str, 
                                  value_column: str,
                                  freq: str = 'D') -> pd.DataFrame:
        """Create time series features from a datetime column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the datetime column
            value_column: Name of the value column to aggregate
            freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            DataFrame with time series features
        """
        if date_column not in df.columns or value_column not in df.columns:
            return df
        
        # Make a copy and set the date as index
        ts_df = df[[date_column, value_column]].copy()
        ts_df = ts_df.set_index(date_column)
        
        # Resample to the specified frequency
        resampled = ts_df.resample(freq)[value_column].sum().reset_index()
        
        # Add time-based features
        resampled = self._add_time_features(resampled, date_column)
        
        # Add lag features
        for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
            resampled[f'lag_{lag}'] = resampled[value_column].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14, 30]:  # 1 week, 2 weeks, 1 month windows
            resampled[f'rolling_mean_{window}'] = (
                resampled[value_column]
                .rolling(window=window, min_periods=1)
                .mean()
            )
            
            resampled[f'rolling_std_{window}'] = (
                resampled[value_column]
                .rolling(window=window, min_periods=1)
                .std()
            )
        
        return resampled
    
    def _add_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add time-based features to a DataFrame."""
        df = df.copy()
        
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
        df[f'{date_column}_weekofyear'] = df[date_column].dt.isocalendar().week
        df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        
        # Add holiday indicators (simplified - in practice, use a holidays library)
        # This is just a placeholder - you'd want to use a proper holiday calendar
        df[f'{date_column}_is_holiday'] = 0
        
        return df
