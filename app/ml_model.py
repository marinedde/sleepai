"""
Classe pour charger et utiliser le modèle de classification de sommeil.

Cette classe encapsule toute la logique ML pour :
- Charger le pipeline depuis le disque
- Faire des prédictions
- Gérer les erreurs
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging
from app.feature_extractor import FeatureExtractor  # Import nécessaire pour joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepStageClassifier:
    """
    Classificateur de stades de sommeil utilisant un pipeline sklearn.
    
    Le pipeline contient :
    1. FeatureExtractor : Extrait 16 features du signal EEG brut
    2. StandardScaler : Normalise les features
    3. RandomForestClassifier : Prédit le stade de sommeil
    """
    
    # Mapping des indices vers les noms de classes
    CLASS_NAMES = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM'
    }
    
    # Métadonnées du modèle (à remplir avec tes vraies valeurs)
    MODEL_METADATA = {
        'model_type': 'Random Forest with Feature Engineering',
        'accuracy': 0.6462,
        'f1_score': 0.6377,
        'cohens_kappa': 0.5247,
        'n_features': 16,
        'training_date': '2025-10-16'
    }
    
    def __init__(self, model_path: str):
        """
        Initialise le classificateur en chargeant le pipeline.
        
        Args:
            model_path: Chemin vers le fichier .joblib du pipeline
        """
        self.model_path = Path(model_path)
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Charge le pipeline depuis le disque."""
        try:
            logger.info(f"Chargement du modèle depuis {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            
            self.pipeline = joblib.load(self.model_path)
            logger.info("✅ Modèle chargé avec succès")
            logger.info(f"   Étapes du pipeline: {list(self.pipeline.named_steps.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def predict(self, signal: np.ndarray) -> Tuple[str, int, float, Dict[str, float]]:
        """
        Prédit le stade de sommeil à partir d'un signal EEG.
        
        Args:
            signal: Signal EEG de shape (1, 3000) ou (3000,)
        
        Returns:
            Tuple contenant:
            - predicted_class (str): Nom de la classe ('Wake', 'N1', etc.)
            - predicted_index (int): Index de la classe (0-4)
            - confidence (float): Confiance de la prédiction (0-1)
            - probabilities (dict): Probabilités pour chaque classe
        
        Raises:
            ValueError: Si le signal n'a pas la bonne shape
        """
        # Validation de la shape
        signal = np.array(signal)
        
        if signal.ndim == 1:
            # Si shape (3000,), reshaper en (1, 3000)
            signal = signal.reshape(1, -1)
        
        if signal.shape != (1, 3000):
            raise ValueError(
                f"Signal doit avoir shape (1, 3000) ou (3000,), reçu {signal.shape}"
            )
        
        # Prédiction
        try:
            # Prédire la classe
            prediction = self.pipeline.predict(signal)
            predicted_index = int(prediction[0])
            predicted_class = self.CLASS_NAMES[predicted_index]
            
            # Prédire les probabilités
            probabilities_array = self.pipeline.predict_proba(signal)[0]
            confidence = float(np.max(probabilities_array))
            
            # Créer le dictionnaire de probabilités
            probabilities = {
                self.CLASS_NAMES[i]: float(prob)
                for i, prob in enumerate(probabilities_array)
            }
            
            logger.info(f"Prédiction: {predicted_class} (confiance: {confidence:.2%})")
            
            return predicted_class, predicted_index, confidence, probabilities
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle."""
        return {
            **self.MODEL_METADATA,
            'classes': list(self.CLASS_NAMES.values()),
            'model_loaded': self.pipeline is not None
        }
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self.pipeline is not None