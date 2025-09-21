
import requests
import json
from datetime import date, timedelta

def test_api(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    
    print(f"Testing API at: {base_url}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        print(f"Health: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test prediction
    try:
        prediction_data = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-07"
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_data
        )
        
        print(f"Prediction endpoint: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Predictions for {len(results)} days")
            print(f"First prediction: {results[0] if results else 'None'}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Prediction test failed: {e}")

if __name__ == "__main__":
    test_api()
