# SmartRetail Analytics - Production Deployment

## Quick Start

1. **Local Testing:**
   ```bash
   cd deployment
   pip install -r requirements.txt
   python main.py
   ```
   Visit: http://localhost:8000/docs

2. **Docker Deployment:**
   ```bash
   docker build -t smartretail-api .
   docker run -p 8000:8000 smartretail-api
   ```

3. **Cloud Deployment Options:**

   ### Render.com (Free)
   - Connect your GitHub repo
   - Use the render.yaml configuration
   - Automatic deployments on git push

   ### Google Cloud Run
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

   ### Hugging Face Spaces
   - Create new Space with Docker
   - Upload files to Space repository

## API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check and model status
- `POST /predict` - Generate sales forecasts
- `GET /model/info` - Model performance metrics
- `GET /docs` - Interactive API documentation

## Model Performance

- WMAPE: 5.81% (Excellent forecasting accuracy)
- R²: 0.967 (Explains 96.7% of variance)
- Revenue Accuracy: 98.4%
- Business Impact: $1.6M annual optimization potential

## Features

- ✅ Production-ready REST API
- ✅ Model performance monitoring
- ✅ Health checks and error handling
- ✅ Docker containerization
- ✅ Cloud deployment configurations
- ✅ Interactive API documentation
