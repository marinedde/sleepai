"""
Référence statistique d'entraînement pour la détection de drift des features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "models" / "baseline_stats.json"


def load_baseline(task: str) -> Optional[dict]:
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tasks", {}).get(task)


def compare_features(
    features: np.ndarray,
    task: str,
    threshold: float = 2.0,
) -> dict:
    """
    Compare la moyenne du batch courant à la baseline d'entraînement.
    Drift si distance normalisée (z-score moyen) > threshold.
    """
    ref = load_baseline(task)
    if ref is None:
        return {
            "drift_detected": False,
            "message": "Baseline absente — lancez python_scripts/retrain.py",
            "threshold": threshold,
        }

    ref_mean = np.array(ref["mean"], dtype=np.float64)
    ref_std = np.array(ref["std"], dtype=np.float64)
    ref_std = np.where(ref_std < 1e-8, 1e-8, ref_std)

    batch_mean = np.mean(features, axis=0)
    z = np.abs((batch_mean - ref_mean) / ref_std)
    score = float(np.mean(z))
    drift = score > threshold

    return {
        "drift_detected": drift,
        "feature_drift_score": round(score, 4),
        "threshold": threshold,
        "n_features_compared": int(features.shape[1]),
        "message": (
            f"Drift features détecté (score={score:.3f} > {threshold})"
            if drift
            else f"Distribution features stable (score={score:.3f})"
        ),
    }
