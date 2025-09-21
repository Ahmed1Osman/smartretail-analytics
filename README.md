# SmartRetail Analytics

A comprehensive retail analytics solution for sales forecasting and inventory optimization using machine learning.

## 🏗️ Package Structure

The SmartRetail Analytics package is organized into the following modules:

```
smartretail_analytics/
├── __init__.py           # Package initialization
├── data/                 # Data handling
│   ├── __init__.py
│   ├── collection.py     # Data collection and loading
│   └── preprocessing.py  # Data cleaning and transformation
├── features/             # Feature engineering
│   ├── __init__.py
│   └── feature_engineering.py  # Feature creation and transformation
├── models/               # Model training and prediction
│   ├── __init__.py
│   ├── train.py          # Model training utilities
│   └── predict.py        # Model prediction utilities
└── visualization/        # Data visualization
    ├── __init__.py
    └── plots.py          # Plotting utilities
```

## 📋 Project Structure

```
smartretail-analytics/
├── README.md               # Project documentation
├── data/                   # Sample data and collection scripts
├── models/                 # Training notebooks and model artifacts
│   └── code.ipynb          # Jupyter notebook for model training
├── deployment/             # Deployment configurations
│   ├── Dockerfile          # Docker configuration
│   ├── requirements.txt    # Python dependencies
│   ├── main.py             # FastAPI application
│   ├── test_api.py         # API test script
│   ├── sales_forecasting_model.joblib  # Trained model
│   ├── feature_columns.json            # Feature columns
│   ├── model_metadata.json             # Model metadata
│   ├── docker-compose.yml  # Docker Compose configuration
│   ├── cloudbuild.yaml     # Cloud Build configuration
│   └── render.yaml         # Render deployment configuration
└── docs/                   # Additional documentation
    ├── api-screenshot-1.jpeg
    └── api-screenshot-2.jpeg
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker
- Azure CLI (for deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd smartretail-analytics
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r deployment/requirements.txt
   ```

## 🏃‍♂️ Running Locally

### Start the API
```bash
cd deployment
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Health Check
```
GET /health
```
Check if the API is running.

#### 2. Get Model Information
```
GET /model/info
```
Get information about the deployed model.

#### 3. Make Predictions
```
POST /predict
```
Make sales predictions using the trained model.

### API Screenshots

#### Local Development
![Local API Documentation](docs/Screenshot_21-9-2025_7814_localhost.jpeg)
*Local API documentation and testing interface*

#### Production Deployment
![Production API](docs/Screenshot_21-9-2025_7745_smartretailapi-demo.switzerlandnorth.azurecontainer.io.jpeg)
*Production API deployment on Azure Container Instances*

## 🐳 Docker Deployment

### Build the Docker image
```bash
docker build -t smartretail-api -f deployment/Dockerfile .
```

### Run the container
```bash
docker run -d -p 8000:8000 smartretail-api
```

## ☁️ Cloud Deployment

### Azure Container Instances (ACI)
```bash
az container create \
  --resource-group your-resource-group \
  --name smartretail-api \
  --image yourcontainerregistry.azurecr.io/smartretail-api:1.0 \
  --cpu 1 --memory 1.5 \
  --registry-login-server yourcontainerregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label smartretailapi-demo \
  --ports 8000 \
  --location switzerlandnorth \
  --os-type Linux
```

## 📊 Model Information

The sales forecasting model is built using:
- **Algorithm**: Linear Regression
- **Features**: Historical sales data, product categories, promotions
- **Performance**: 98.4% revenue accuracy

## 🚀 Example Workflow

The `examples/` directory contains a complete example workflow demonstrating how to use the SmartRetail Analytics package:

1. **Data Collection**: Load and preprocess retail data
2. **Feature Engineering**: Create meaningful features for modeling
3. **Model Training**: Train and evaluate machine learning models
4. **Prediction**: Make predictions using the trained model
5. **Visualization**: Create insightful visualizations

To run the example:

```bash
# Navigate to the examples directory
cd examples

# Install example dependencies
pip install -r requirements.txt

# Run the example workflow
python example_workflow.py
```

This will execute the workflow and save visualizations to the `output` directory.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
