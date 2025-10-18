"""
API FastAPI pour la classification de stades de sommeil.

Cette API expose le modèle SleepAI via des endpoints REST.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import logging

from app.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.ml_model import SleepStageClassifier

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable globale pour le modèle (sera initialisée au startup)
model: SleepStageClassifier = None

# Chemin absolu du modèle
PROJECT_ROOT = Path(__file__).parent.parent  # Remonte de app/ vers sleepai/
MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "rf_v2_pipeline_fixed.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.
    
    - startup: Charge le modèle au démarrage
    - shutdown: Nettoyage (si nécessaire)
    """
    # Startup: Charger le modèle
    global model
    logger.info("🚀 Démarrage de l'API SleepAI...")
    logger.info(f"📂 Chemin du modèle : {MODEL_PATH}")
    
    try:
        model = SleepStageClassifier(model_path=str(MODEL_PATH))
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur au chargement du modèle: {e}")
        raise
    
    yield  # L'API tourne ici
    
    # Shutdown: Nettoyage
    logger.info("🛑 Arrêt de l'API SleepAI...")


# Créer l'application FastAPI
app = FastAPI(
    title="SleepAI API",
    description="""
    API de classification automatique de stades de sommeil à partir de signaux EEG.
    
    ## Fonctionnalités
    
    * **Prédiction** : Classifie un signal EEG en 5 stades (Wake, N1, N2, N3, REM)
    * **Monitoring** : Endpoints de santé et d'information sur le modèle
    
    ## Utilisation
    
    Envoyez un signal EEG de 30 secondes (3000 points à 100Hz) au endpoint `/predict`.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS (pour autoriser les requêtes depuis un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint racine.
    
    Retourne un message de bienvenue et les endpoints disponibles.
    """
    return {
        "message": "Bienvenue sur l'API SleepAI 🌙",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "documentation": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Vérifie l'état de santé de l'API.
    
    Retourne:
    - status: "healthy" ou "unhealthy"
    - model_loaded: True si le modèle est chargé
    - model_path: Chemin du modèle
    """
    is_healthy = model is not None and model.is_loaded()
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_path=str(model.model_path) if model else "N/A"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def get_model_info():
    """
    Retourne les informations sur le modèle chargé.
    
    Inclut:
    - Type de modèle
    - Métriques de performance (accuracy, F1-score, etc.)
    - Classes prédites
    - Nombre de features
    """
    if model is None or not model.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé"
        )
    
    return ModelInfoResponse(**model.get_model_info())


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sleep_stage(request: PredictionRequest):
    """
    Prédit le stade de sommeil à partir d'un signal EEG.
    
    ## Input
    
    - **signal**: Liste de 3000 valeurs numériques (30s à 100Hz)
    
    ## Output
    
    - **predicted_class**: Stade de sommeil prédit (Wake, N1, N2, N3, REM)
    - **predicted_index**: Index de la classe (0-4)
    - **confidence**: Confiance de la prédiction (0-1)
    - **probabilities**: Probabilités pour chaque classe
    
    ## Exemple
```python
    import requests
    
    signal = [0.5, -0.2, 1.3, ...] # 3000 valeurs
    response = requests.post("http://localhost:8000/predict", json={"signal": signal})
    print(response.json())
```
    """
    # Vérifier que le modèle est chargé
    if model is None or not model.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Veuillez redémarrer le serveur."
        )
    
    try:
        # Convertir la liste en array numpy
        signal_array = np.array(request.signal).reshape(1, -1)
        
        # Faire la prédiction
        predicted_class, predicted_index, confidence, probabilities = model.predict(signal_array)
        
        # Retourner la réponse
        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except ValueError as e:
        # Erreur de validation du signal
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signal invalide: {str(e)}"
        )
    except Exception as e:
        # Erreur interne
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )


# Pour exécuter en mode développement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)