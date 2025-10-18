"""
Feature extractor pour les signaux EEG.

Cette classe extrait 16 features d'un signal EEG brut (3000 points).
Elle est utilisée dans le pipeline sklearn.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.signal import welch


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformeur custom qui extrait les features EEG.
    
    Hérite de BaseEstimator (pour get_params/set_params)
    et TransformerMixin (pour fit_transform)
    → Entièrement sérialisable par joblib ✅
    
    Features extraites (16 au total):
    - 8 statistiques temporelles : mean, std, min, max, Q1, Q3, skewness, kurtosis
    - 5 puissances spectrales : Delta, Theta, Alpha, Beta, Gamma
    - 3 ratios de puissance : delta/total, theta/total, alpha/total
    """
    
    def __init__(self, fs=100, expected_len=3000):
        """
        Parameters
        ----------
        fs : int
            Fréquence d'échantillonnage (Hz)
        expected_len : int
            Longueur attendue du signal brut (30s × 100Hz = 3000)
        """
        self.fs = fs
        self.expected_len = expected_len
    
    def fit(self, X, y=None):
        """Fit ne fait rien, juste pour sklearn compatibility"""
        return self
    
    def transform(self, X):
        """
        Transforme les signaux bruts en features.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, expected_len)
            Signaux EEG bruts (3000 points = 30s × 100Hz)
        
        Returns
        -------
        features : array, shape (n_samples, 16)
            Features extraites
        """
        X = np.asarray(X)
        
        # Valider shape
        if X.ndim != 2 or X.shape[1] != self.expected_len:
            raise ValueError(
                f"X doit être (N, {self.expected_len}), reçu {X.shape}"
            )
        
        # Extraire features pour chaque sample
        features_list = []
        for i, signal in enumerate(X):
            try:
                feats = self._extract_features_from_signal(signal)
                features_list.append(feats)
            except Exception as e:
                print(f"⚠️ Erreur extraction features sample {i}: {e}")
                # Retourner features par défaut (tous 0)
                feats = np.zeros(16)
                features_list.append(feats)
        
        # Stack en array 2D
        features_array = np.vstack(features_list)
        
        return features_array
    
    def _extract_features_from_signal(self, epoch):
        """
        Extrait 16 features d'une époque EEG.
        
        Parameters
        ----------
        epoch : array, shape (3000,)
            Signal EEG de 30 secondes
        
        Returns
        -------
        features : array, shape (16,)
            Features extraites
        """
        features = []
        
        # 1. Statistiques temporelles (8 features)
        features.append(np.mean(epoch))              # Amplitude moyenne
        features.append(np.std(epoch))               # Écart-type
        features.append(np.min(epoch))               # Min
        features.append(np.max(epoch))               # Max
        features.append(np.percentile(epoch, 25))    # Q1
        features.append(np.percentile(epoch, 75))    # Q3
        features.append(stats.skew(epoch))           # Asymétrie
        features.append(stats.kurtosis(epoch))       # Aplatissement
        
        # 2. Analyse spectrale (puissance par bande)
        freqs, psd = welch(epoch, fs=self.fs, nperseg=256)
        
        # Delta (0.5-4 Hz) - Sommeil profond
        delta_idx = (freqs >= 0.5) & (freqs < 4)
        delta_power = np.mean(psd[delta_idx]) if delta_idx.any() else 0
        features.append(delta_power)
        
        # Theta (4-8 Hz) - Somnolence
        theta_idx = (freqs >= 4) & (freqs < 8)
        theta_power = np.mean(psd[theta_idx]) if theta_idx.any() else 0
        features.append(theta_power)
        
        # Alpha (8-13 Hz) - Relaxation
        alpha_idx = (freqs >= 8) & (freqs < 13)
        alpha_power = np.mean(psd[alpha_idx]) if alpha_idx.any() else 0
        features.append(alpha_power)
        
        # Beta (13-30 Hz) - Éveil actif
        beta_idx = (freqs >= 13) & (freqs < 30)
        beta_power = np.mean(psd[beta_idx]) if beta_idx.any() else 0
        features.append(beta_power)
        
        # Gamma (30-35 Hz) - Cognition
        gamma_idx = (freqs >= 30) & (freqs <= 35)
        gamma_power = np.mean(psd[gamma_idx]) if gamma_idx.any() else 0
        features.append(gamma_power)
        
        # 3. Ratios de puissance (3 features)
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
        if total_power > 0:
            features.append(delta_power / total_power)  # Ratio delta
            features.append(theta_power / total_power)  # Ratio theta
            features.append(alpha_power / total_power)  # Ratio alpha
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)