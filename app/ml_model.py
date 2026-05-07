"""
Classificateur de stades de sommeil (EEG).

Charge et utilise le pipeline sklearn :
  FeatureExtractor → StandardScaler → RandomForestClassifier

Note : le pipeline sauvegardé contient StandardScaler + RF uniquement.
L'extraction des features est faite ici avant d'appeler le pipeline.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

from app.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Extracteur utilisé pour transformer le signal brut en 16 features
_extractor = FeatureExtractor(fs=100, expected_len=3000)


class SleepStageClassifier:
    """
    Classificateur EEG — 5 stades de sommeil.

    Flux :
        signal brut (3000,) → FeatureExtractor → 16 features → pipeline sklearn
    """

    CLASS_NAMES = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM',
    }

    CLASS_INTERPRETATIONS = {
        'Wake': "Éveil — Patient réveillé ou en micro-éveil",
        'N1'  : "Sommeil léger N1 — Endormissement, stade de transition",
        'N2'  : "Sommeil léger N2 — Stade le plus fréquent, fuseaux de sommeil",
        'N3'  : "Sommeil profond N3 — Ondes lentes, récupération physique",
        'REM' : "Sommeil paradoxal REM — Rêves, récupération cognitive",
    }

    MODEL_METADATA = {
        'model_type'   : 'Random Forest + Feature Engineering (16 features EEG)',
        'accuracy'     : 0.8344,
        'f1_weighted'  : 0.8302,
        'n_features'   : 16,
        'training_date': '2026-05-07',
        'dataset'      : 'Sleep-EDF Expanded (PhysioNet) — 28 sujets',
    }

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.pipeline   = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle EEG non trouvé : {self.model_path}")
        logger.info(f"Chargement modèle EEG : {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info("Modèle EEG chargé")

    def predict(
        self, signal: np.ndarray
    ) -> Tuple[str, int, float, Dict[str, float], str]:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        if signal.shape[1] != 3000:
            raise ValueError(
                f"Signal EEG doit contenir 3000 points, reçu {signal.shape[1]}"
            )

        # Extraction des 16 features depuis le signal brut
        features = _extractor.transform(signal)  # (1, 16)

        # Prédiction via pipeline (StandardScaler + RF)
        pred_idx      = int(self.pipeline.predict(features)[0])
        pred_class    = self.CLASS_NAMES[pred_idx]
        proba_arr     = self.pipeline.predict_proba(features)[0]
        confidence    = float(np.max(proba_arr))
        probabilities = {
            self.CLASS_NAMES[i]: float(p)
            for i, p in enumerate(proba_arr)
        }
        interpretation = self.CLASS_INTERPRETATIONS[pred_class]

        logger.info(f"EEG → {pred_class} (confiance : {confidence:.2%})")
        return pred_class, pred_idx, confidence, probabilities, interpretation

    def get_info(self) -> dict:
        return {
            **self.MODEL_METADATA,
            'classes'     : list(self.CLASS_NAMES.values()),
            'model_loaded': self.is_loaded(),
        }

    def is_loaded(self) -> bool:
        return self.pipeline is not None
