import pytest
from fastapi.testclient import TestClient
from app.main import app
import numpy as np

client = TestClient(app)

def test_api_running():
    """Test que l'API démarre"""
    response = client.get("/health")
    assert response.status_code == 200
    print("✅ API is running")

def test_health_response_structure():
    """Test la structure de la réponse /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" in data
    print(f"✅ Health check - Model loaded: {data['model_loaded']}")

def test_openapi_docs():
    """Test documentation accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    print("✅ OpenAPI docs accessible")

def test_openapi_schema():
    """Test schéma OpenAPI valide"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/predict" in data["paths"]
    assert "/health" in data["paths"]
    assert "/model-info" in data["paths"]
    print("✅ OpenAPI schema valid")

def test_predict_validation_empty():
    """Test validation - signal vide"""
    response = client.post("/predict", json={"signal": []})
    assert response.status_code == 422
    print("✅ Empty signal validation works")

def test_predict_validation_short():
    """Test validation - signal trop court"""
    response = client.post("/predict", json={"signal": [1, 2, 3]})
    assert response.status_code == 422
    print("✅ Short signal validation works")

def test_predict_validation_missing():
    """Test validation - champ manquant"""
    response = client.post("/predict", json={})
    assert response.status_code == 422
    print("✅ Missing field validation works")

# Tests conditionnels - seulement si le modèle est chargé
def test_model_info_if_loaded():
    """Test /model-info si le modèle est chargé"""
    health = client.get("/health").json()
    
    if not health.get("model_loaded", False):
        pytest.skip("Model not loaded - skipping model test")
    
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "model_type" in data
    print("✅ Model info retrieved")

def test_predict_if_loaded():
    """Test prédiction si le modèle est chargé"""
    health = client.get("/health").json()
    
    if not health.get("model_loaded", False):
        pytest.skip("Model not loaded - skipping prediction test")
    
    np.random.seed(42)
    signal = np.random.randn(3000).tolist()
    
    response = client.post("/predict", json={"signal": signal})
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Wake", "N1", "N2", "N3", "REM"]
    assert 0 <= data["confidence"] <= 1
    print(f"✅ Prediction - Predicted: {data['predicted_class']}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])