"""
Modèles Pydantic pour la validation des données de l'API.

Pydantic permet de :
- Valider automatiquement les types de données
- Générer une documentation automatique
- Gérer les erreurs de manière claire
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict
import numpy as np


class PredictionRequest(BaseModel):
    """
    Requête pour prédire un stade de sommeil.
    
    Le signal EEG doit contenir exactement 3000 points (30s à 100Hz).
    """
    signal: List[float] = Field(
        ...,  # ... signifie "obligatoire"
        description="Signal EEG de 30 secondes (3000 points à 100Hz)",
        min_length=3000,
        max_length=3000,
        example=[0.5, -0.2, 1.3] + [0.0] * 2997  # Exemple tronqué pour la doc
    )
    
    @validator('signal')
    def validate_signal(cls, v):
        """Vérifie que le signal contient des valeurs numériques valides."""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Le signal doit contenir uniquement des nombres")
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Le signal contient des valeurs NaN ou infinies")
        return v


class PredictionResponse(BaseModel):
    """
    Réponse après prédiction d'un stade de sommeil.
    """
    predicted_class: str = Field(
        ...,
        description="Classe de sommeil prédite (Wake, N1, N2, N3, REM)"
    )
    predicted_index: int = Field(
        ...,
        description="Index de la classe (0-4)",
        ge=0,
        le=4
    )
    confidence: float = Field(
        ...,
        description="Confiance de la prédiction (0-1)",
        ge=0.0,
        le=1.0
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilités pour chaque classe"
    )


class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle chargé."""
    model_type: str
    accuracy: float
    f1_score: float
    cohens_kappa: float
    classes: List[str]
    n_features: int
    training_date: str