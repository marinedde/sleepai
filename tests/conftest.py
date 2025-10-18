import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import numpy as np

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

@pytest.fixture(scope="session", autouse=True)
def mock_model():
    """Mock le modèle ML pour les tests"""
    
    # Créer un mock du pipeline
    mock_pipeline = Mock()
    
    # Configurer le comportement du mock
    def mock_predict(X):
        # Retourner une prédiction aléatoire mais cohérente
        return np.array([2])  # Toujours prédire "N2" pour simplicité
    
    def mock_predict_proba(X):
        # Retourner des probabilités fictives
        return np.array([[0.1, 0.15, 0.5, 0.2, 0.05]])  # Wake, N1, N2, N3, REM
    
    mock_pipeline.predict = mock_predict
    mock_pipeline.predict_proba = mock_predict_proba
    
    # Patcher le chargement du modèle
    with patch('joblib.load', return_value=mock_pipeline):
        # Forcer le rechargement de app.main avec le mock
        if 'app.main' in sys.modules:
            del sys.modules['app.main']
        
        yield mock_pipeline