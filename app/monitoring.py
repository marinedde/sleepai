import json
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleMonitor:
    """Système de monitoring simple pour logger et analyser les prédictions"""
    
    def __init__(self, log_file: str = "logs/predictions.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 Monitoring initialisé : {self.log_file}")
    
    def log_prediction(self, 
                      signal: List[float],
                      prediction: str,
                      confidence: float,
                      probabilities: Dict[str, float],
                      processing_time: float = None):
        """Logger une prédiction avec ses métadonnées"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "signal_stats": {
                "mean": float(np.mean(signal)),
                "std": float(np.std(signal)),
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "length": len(signal)
            },
            "processing_time_ms": processing_time
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Erreur lors du logging : {e}")
    
    def get_recent_logs(self, n: int = 100) -> List[Dict]:
        """Récupérer les N derniers logs"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                return [json.loads(line) for line in lines[-n:]]
        except Exception as e:
            logger.error(f"Erreur lecture logs : {e}")
            return []
    
    def get_statistics(self, last_n: int = 100) -> Dict:
        """Calculer des statistiques sur les dernières prédictions"""
        logs = self.get_recent_logs(last_n)
        
        if not logs:
            return {
                "message": "No predictions logged yet",
                "total_predictions": 0
            }
        
        predictions = [log["prediction"] for log in logs]
        confidences = [log["confidence"] for log in logs]
        
        # Distribution des classes
        unique, counts = np.unique(predictions, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # Statistiques de confiance
        confidence_stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences))
        }
        
        # Confiance moyenne par classe
        confidence_by_class = {}
        for cls in set(predictions):
            cls_confidences = [log["confidence"] for log in logs if log["prediction"] == cls]
            confidence_by_class[cls] = float(np.mean(cls_confidences))
        
        # Temps de traitement moyen
        processing_times = [log.get("processing_time_ms", 0) for log in logs if log.get("processing_time_ms")]
        avg_processing_time = float(np.mean(processing_times)) if processing_times else 0
        
        return {
            "total_predictions": len(logs),
            "time_range": {
                "first": logs[0]["timestamp"],
                "last": logs[-1]["timestamp"]
            },
            "class_distribution": class_distribution,
            "confidence_stats": confidence_stats,
            "confidence_by_class": confidence_by_class,
            "avg_processing_time_ms": avg_processing_time,
            "last_prediction": logs[-1] if logs else None
        }
    
    def detect_drift(self, threshold: float = 0.1, window_size: int = 50) -> Dict:
        """Détecter une potentielle dérive du modèle"""
        logs = self.get_recent_logs(window_size * 2)
        
        if len(logs) < window_size:
            return {
                "drift_detected": False, 
                "message": "Not enough data for drift detection",
                "required_samples": window_size,
                "current_samples": len(logs)
            }
        
        # Comparer les confidences récentes vs anciennes
        recent_confidences = [log["confidence"] for log in logs[-window_size:]]
        older_confidences = [log["confidence"] for log in logs[-2*window_size:-window_size]]
        
        recent_avg = np.mean(recent_confidences)
        older_avg = np.mean(older_confidences)
        
        difference = abs(recent_avg - older_avg)
        drift = difference > threshold
        
        # Distribution des classes
        recent_preds = [log["prediction"] for log in logs[-window_size:]]
        older_preds = [log["prediction"] for log in logs[-2*window_size:-window_size]]
        
        recent_dist = dict(zip(*np.unique(recent_preds, return_counts=True)))
        older_dist = dict(zip(*np.unique(older_preds, return_counts=True)))
        
        return {
            "drift_detected": drift,
            "confidence_drift": {
                "recent_avg": float(recent_avg),
                "older_avg": float(older_avg),
                "difference": float(difference),
                "threshold": threshold
            },
            "class_distribution_shift": {
                "recent": recent_dist,
                "older": older_dist
            },
            "recommendation": "Retrain model" if drift else "Model performing normally"
        }