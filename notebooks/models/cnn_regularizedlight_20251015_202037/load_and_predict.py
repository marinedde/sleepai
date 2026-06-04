
"""
Script pour charger et utiliser le modèle CNN_RegularizedLight
Créé le: 20251015_202037
Accuracy: 67.51%
"""

import numpy as np
from tensorflow.keras.models import load_model
import json

# Chemin du modèle
MODEL_PATH = "cnn_model.h5"  # ou "cnn_model_savedmodel"
METADATA_PATH = "metadata.json"
NORM_PATH = "normalization_params.json"

# Charger le modèle
model = load_model(MODEL_PATH)

# Charger les métadonnées
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

# Charger les paramètres de normalisation
with open(NORM_PATH, 'r') as f:
    norm_params = json.load(f)

# Classes de sortie
CLASS_NAMES = metadata["architecture"]["class_names"]

def predict_sleep_stage(signal_3d):
    """
    Prédire le stade de sommeil à partir d'un signal EEG 3D
    
    Args:
        signal_3d: array de shape (1, 500, 6) - 500 timesteps × 6 canaux
        
    Returns:
        Dictionnaire avec:
        - predicted_class: classe prédite (Wake, N1, N2, N3, REM)
        - probabilities: probabilités pour chaque classe
        - confidence: confiance de la prédiction
    """
    
    # S'assurer que le signal a la bonne shape
    if signal_3d.shape != (1, 500, 6):
        raise ValueError(f"Signal doit avoir shape (1, 500, 6), reçu {signal_3d.shape}")
    
    # Normaliser le signal si nécessaire
    signal_normalized = (signal_3d - norm_params['mean_train']) / norm_params['std_train']
    
    # Prédiction
    predictions = model.predict(signal_normalized, verbose=0)
    
    # Résultats
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    return {
        "predicted_class": predicted_class,
        "predicted_idx": int(predicted_idx),
        "probabilities": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))},
        "confidence": confidence
    }

# Exemple d'utilisation:
# signal = np.random.randn(1, 500, 6)  # Signal EEG aléatoire
# result = predict_sleep_stage(signal)
# print(result)
