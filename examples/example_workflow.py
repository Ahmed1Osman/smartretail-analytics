""
Example workflow for SmartRetail Analytics package.

This script demonstrates how to use the SmartRetail Analytics package to:
1. Load and preprocess data
2. Perform feature engineering
3. Train and evaluate machine learning models
4. Make predictions
5. Visualize results
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import SmartRetail Analytics components
from smartretail_analytics.data import DataCollector, DataPreprocessor
from smartretail_analytics.features import FeatureEngineer
from smartretail_analytics.models import ModelTrainer, Predictor
from smartretail_analytics.visualization import Plotter

def main():
    """Main function to demonstrate the SmartRetail Analytics workflow."""
    # Set up paths
    data_dir = Path("data/raw")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Starting SmartRetail Analytics Workflow")
    print("=" * 50)
    
    # 1. Data Collection
    print("\n📊 Step 1: Data Collection")
    print("-" * 30)
    
    # Initialize data collector
    collector = DataCollector(data_dir=data_dir)
    
    # List available datasets
    datasets = collector.get_available_datasets()
    print("Available datasets:")
    for dataset_type, info in datasets.items():
        print(f"- {info['name']}:")
        for file in info['files']:
            print(f"  • {file}")
    
    # Load sample data (replace with actual data loading)
    print("\n📂 Loading sample data...")
    try:
        # Example: Load orders data
        orders_df = pd.DataFrame({
            'order_id': [f'order_{i}' for i in range(1, 101)],
            'customer_id': [f'customer_{i%20+1}' for i in range(100)],
            'order_purchase_timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'order_status': ['delivered'] * 90 + ['shipped'] * 5 + ['canceled'] * 5,
            'order_amount': np.random.normal(100, 30, 100).clip(10, 500)
        })
        
        print(f"✅ Loaded sample orders data: {len(orders_df)} rows")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # 2. Data Preprocessing
    print("\n🧹 Step 2: Data Preprocessing")
    print("-" * 30)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Convert dates
    print("📅 Converting dates...")
    orders_processed = preprocessor.convert_dates(orders_df)
    
    # Add date features
    print("📊 Adding date features...")
    orders_processed = preprocessor.add_date_features(orders_processed, 'order_purchase_timestamp')
    
    print(f"✅ Processed data shape: {orders_processed.shape}")
    
    # 3. Feature Engineering
    print("\n⚙️ Step 3: Feature Engineering")
    print("-" * 30)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create time series features
    print("⏱️ Creating time series features...")
    time_series_df = feature_engineer.create_time_series_features(
        df=orders_processed,
        date_column='order_purchase_timestamp',
        value_column='order_amount',
        freq='D'
    )
    
    print(f"✅ Created {len(time_series_df.columns) - 2} time series features")
    
    # 4. Model Training
    print("\n🤖 Step 4: Model Training")
    print("-" * 30)
    
    # Initialize model trainer
    model_trainer = ModelTrainer(model_dir='models')
    
    # Prepare data for training
    print("📝 Preparing data for training...")
    X = time_series_df.drop(columns=['order_purchase_timestamp', 'order_amount'])
    y = time_series_df['order_amount']
    
    # Train models
    print("🏋️ Training models...")
    results = model_trainer.train_models(X, y)
    
    # Evaluate models
    print("📊 Evaluating models...")
    evaluation = model_trainer.evaluate_models(X, y)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    for model_name, metrics in evaluation.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 5. Make Predictions
    print("\n🔮 Step 5: Making Predictions")
    print("-" * 30)
    
    # Initialize predictor
    predictor = Predictor(model_dir='models')
    
    # Load the latest model
    print("🔄 Loading the best model...")
    model, metadata, feature_columns = predictor.load_latest_model()
    print(f"✅ Loaded model: {metadata.get('model_type', 'Unknown')}")
    print(f"   Trained on: {metadata.get('training_date', 'Unknown')}")
    
    # Make predictions
    print("📈 Making predictions...")
    sample_data = X.iloc[:5].copy()  # Predict on first 5 samples
    predictions = predictor.predict(sample_data)
    
    print("\nSample Predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: Predicted = ${pred:.2f}, Actual = ${y.iloc[i]:.2f}")
    
    # 6. Visualization
    print("\n📊 Step 6: Visualization")
    print("-" * 30)
    
    # Initialize plotter
    plotter = Plotter()
    
    # Create time series plot
    print("📅 Creating time series plot...")
    ts_plot = plotter.time_series_plot(
        df=time_series_df,
        x='order_purchase_timestamp',
        y='order_amount',
        title='Daily Order Amounts Over Time',
        ylabel='Order Amount ($)'
    )
    
    # Save the plot
    plot_path = output_dir / 'time_series_plot.png'
    plotter.save_plot(ts_plot, plot_path)
    print(f"✅ Saved time series plot to: {plot_path}")
    
    # Create feature importance plot (if available)
    if hasattr(predictor.model, 'feature_importances_'):
        print("📊 Creating feature importance plot...")
        importance_plot = plotter.feature_importance_plot(
            feature_importance=dict(zip(X.columns, predictor.model.feature_importances_)),
            title='Feature Importance',
            top_n=10
        )
        
        # Save the plot
        importance_path = output_dir / 'feature_importance.png'
        plotter.save_plot(importance_plot, importance_path)
        print(f"✅ Saved feature importance plot to: {importance_path}")
    
    print("\n🎉 SmartRetail Analytics workflow completed successfully!")

if __name__ == "__main__":
    main()
