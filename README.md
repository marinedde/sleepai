# 🌙 SleepAI - Classification Automatique des Stades de Sommeil

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Pipeline MLOps complet pour la classification automatique des stades de sommeil à partir de signaux EEG polysomnographiques.

**🌐 API Déployée :** https://sleepai-api.onrender.com  
**📚 Documentation :** https://sleepai-api.onrender.com/docs

---

## 📋 Table des Matières

- [Problématique & Objectif](#-problématique--objectif)
- [Architecture](#%EF%B8%8F-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Endpoints](#-api-endpoints)
- [Performance](#-performance-du-modèle)
- [Impact Business](#-impact-business-potentiel)
- [Technologies](#%EF%B8%8F-technologies)
- [Monitoring](#-monitoring--logs)
- [Limitations & Roadmap](#%EF%B8%8F-limitations--améliorations-futures)
- [Documentation](#-documentation-technique)
- [Auteur](#-auteur)

---

## 🎯 Problématique & Objectif

### 🏥 Contexte Médical

L'analyse manuelle d'une polysomnographie (enregistrement du sommeil) prend **4 heures par patient** pour un médecin spécialisé. Cette tâche chronophage consiste à classifier manuellement chaque segment de 30 secondes d'enregistrement en 5 stades de sommeil.

**Problèmes identifiés :**
- ⏱️ **Temps d'analyse** : 4 heures par patient
- 💰 **Coût des logiciels** : 10 000€ à 20 000€ pour les outils actuels
- 🚫 **Pas d'IA moderne** : Les logiciels actuels n'intègrent pas de Machine Learning récent
- 👨‍⚕️ **Charge de travail** : Les médecins du sommeil sont surchargés
- 📈 **Demande croissante** : Les troubles du sommeil augmentent (apnée, insomnie)

### 🎯 Objectif du Projet

Développer un **système de classification automatique** des stades de sommeil qui pourrait réduire le temps d'analyse de **4 heures à 30 minutes**, tout en maintenant une qualité diagnostique acceptable.

**Ce projet démontre :**
- ✅ Pipeline MLOps complet (end-to-end)
- ✅ Déploiement cloud automatisé
- ✅ CI/CD et monitoring
- ✅ Interface utilisateur intuitive

> ⚠️ **Note importante** : Ce projet est une **preuve de concept technique** (POC) dans le cadre du Jedha Bootcamp. Il n'est **pas certifié pour un usage médical** et nécessiterait une validation clinique approfondie avant toute utilisation réelle.

**Contexte :** Projet final Lead Data Science - Jedha Bootcamp 2025

---

## 🏗️ Architecture

### Schéma Global
```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│  Sleep-EDF DB   │────────▶│  Model Training  │────────▶│  Random Forest  │
│  (Physionet)    │         │   (Notebook)     │         │   Pipeline      │
│   2584 samples  │         │  Feature Eng.    │         │   64.6% acc.    │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └────────┬────────┘
                                                                   │
                                                                   │ Joblib
                                                                   ▼
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│  Streamlit UI   │◀───────│   FastAPI REST   │◀────────│  Model (69 MB)  │
│  (Dashboard)    │  HTTP   │      API         │ Load    │   + Features    │
│                 │         │  Port 8000       │         │                 │
└─────────────────┘         └────────┬─────────┘         └─────────────────┘
                                     │
                                     │ Logs
                                     ▼
                            ┌──────────────────┐
                            │                  │
                            │   Monitoring     │
                            │  - Stats         │
                            │  - Drift Check   │
                            │  - Predictions   │
                            │                  │
                            └──────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline                              │
│                                                                      │
│  GitHub Push ──▶ Tests (pytest) ──▶ Build Docker ──▶ Deploy Render │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

### Flux de Données
```
Signal EEG (3000 points) 
    │
    ├─▶ Feature Extraction (16 features)
    │      ├─ Temporelles (8) : mean, std, min, max, Q1, Q3, skew, kurtosis
    │      ├─ Fréquentielles (5) : Delta, Theta, Alpha, Beta, Gamma
    │      └─ Ratios (3) : Normalized power ratios
    │
    ├─▶ Scaling (StandardScaler)
    │
    └─▶ Classification (Random Forest)
           └─▶ Output: {Wake, N1, N2, N3, REM} + Probabilities
```

---

## ✨ Fonctionnalités

### ✅ Machine Learning
- **Modèle** : Random Forest (500 estimateurs)
- **Performance** : 64.6% accuracy, F1-score: 0.638, Kappa: 0.525
- **Feature Engineering** : 16 features (temporelles + fréquentielles)
- **Classes** : 5 stades de sommeil (Wake, N1, N2, N3, REM)
- **Pipeline complet** : Feature extraction + Scaling + Classification

### ✅ API REST (FastAPI)
- ✅ Endpoint `/predict` : Classification de signaux EEG
- ✅ Endpoint `/health` : Health check
- ✅ Endpoint `/model-info` : Informations du modèle
- ✅ Endpoints `/monitoring/*` : Stats, drift detection, logs
- ✅ Documentation auto : `/docs` (Swagger UI)
- ✅ Validation Pydantic : Données vérifiées

### ✅ Monitoring
- 📊 Logging des prédictions (JSONL)
- 📈 Statistiques en temps réel
- 🔍 Détection de drift du modèle
- ⏱️ Métriques de performance (temps de traitement)
- 📋 Historique des prédictions

### ✅ CI/CD
- 🧪 Tests automatisés (pytest) : 7 tests
- 🐳 Build Docker automatique
- 🚀 Déploiement continu sur Render
- 🔄 GitHub Actions workflow
- ✅ Tout au vert dans le pipeline

### ✅ Interface Utilisateur
- 🎨 Dashboard Streamlit interactif
- 📊 Visualisation des signaux EEG
- 🎲 Génération de signaux synthétiques par stade
- 🔮 Résultats de prédiction en temps réel
- 📈 Graphiques de probabilités

---

## 📦 Dataset

**Source :** [Sleep-EDF Database Expanded (Physionet)](https://physionet.org/content/sleep-edfx/1.0.0/)

- **Échantillons** : 2584 segments de 30 secondes
- **Fréquence d'échantillonnage** : 100 Hz
- **Longueur du signal** : 3000 points par segment
- **Distribution des classes** :
  - Wake: 12% (éveil)
  - N1: 22% (sommeil léger)
  - N2: 31% (sommeil intermédiaire)
  - N3: 30% (sommeil profond)
  - REM: 5% (sommeil paradoxal)

**Prétraitement :**
- Balancing des classes avec RandomUnderSampler
- Normalisation des signaux
- Extraction de 16 features par segment

---

## 🧠 Les Stades de Sommeil Expliqués

### Classification AASM (American Academy of Sleep Medicine)

Le sommeil est divisé en **5 stades distincts** selon les critères de l'AASM. Chaque stade a des **caractéristiques EEG uniques** qui permettent de les identifier.

### 😴 Wake (Éveil)

**Caractéristiques physiologiques :**
- Yeux ouverts ou fermés
- Conscience de l'environnement
- Mouvements oculaires volontaires

**Signature EEG :**
- **Activité Alpha** (8-13 Hz) : Ondes régulières, prédominantes yeux fermés
- **Activité Beta** (13-30 Hz) : Ondes rapides, activité mentale
- Amplitude : 20-60 μV
- Pattern : Désynchronisé, irrégulier

**Proportion du sommeil :** <5% (éveil bref normal)

**Exemple de tracé :**
```
Éveil (Alpha dominant)
  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲  ← Ondes régulières ~10 Hz
```

---

### 💤 N1 (Sommeil Léger - Stade 1)

**Caractéristiques physiologiques :**
- Transition éveil → sommeil
- Sensation de "tomber"
- Réveil facile
- Perte progressive du tonus musculaire

**Signature EEG :**
- **Disparition de l'Alpha** (8-13 Hz)
- **Apparition du Theta** (4-8 Hz) : Ondes lentes
- Amplitude : 50-75 μV
- Pattern : Mixte, début de ralentissement

**Proportion du sommeil :** 5-10%

**Exemple de tracé :**
```
N1 (Theta émergent)
    ╱‾‾╲___╱‾‾╲___╱‾‾╲  ← Ondes plus lentes ~6 Hz
```

---

### 😌 N2 (Sommeil Léger - Stade 2)

**Caractéristiques physiologiques :**
- Sommeil confirmé
- Réveil encore possible mais plus difficile
- Température corporelle baisse
- Rythme cardiaque ralentit

**Signature EEG :**
- **Fuseaux de sommeil** (Sleep Spindles) : Bouffées de 12-14 Hz
- **Complexes K** : Ondes biphasiques de grande amplitude
- **Ondes Theta** continues (4-8 Hz)
- Amplitude : 75-150 μV

**Proportion du sommeil :** 45-55% (le plus représenté)

**Exemple de tracé :**
```
N2 (avec fuseau)
    ╱‾╲╱‾╲╱‾╲___        ← Fuseau de sommeil
  ╱            ╲__╱     ← Complexe K
```

**Caractéristiques uniques :**
- **Fuseaux** : Bursts de 0.5-2 secondes
- **Complexes K** : Réponse aux stimuli externes

---

### 🌙 N3 (Sommeil Profond - Sommeil Lent)

**Caractéristiques physiologiques :**
- Sommeil le plus réparateur
- Réveil très difficile
- Sécrétion d'hormone de croissance
- Régénération tissulaire
- Consolidation mémoire

**Signature EEG :**
- **Ondes Delta dominantes** (0.5-4 Hz) : >20% du tracé
- Très grande amplitude : 75-200+ μV
- Pattern : Lent, synchronisé
- Ondes lentes de grande amplitude

**Proportion du sommeil :** 15-25%

**Exemple de tracé :**
```
N3 (Delta dominant)
      ╱‾‾‾‾‾╲
    ╱         ╲___      ← Ondes très lentes ~2 Hz
  ╱               ╲___  ← Amplitude élevée
```

**Pourquoi "profond" ?**
- Seuil d'éveil très élevé
- Activité cérébrale très ralentie
- Phase de récupération physique

---

### 🔮 REM (Rapid Eye Movement - Sommeil Paradoxal)

**Caractéristiques physiologiques :**
- **Mouvements oculaires rapides** (d'où le nom REM)
- **Atonie musculaire** : Paralysie temporaire des muscles
- **Rêves intenses** et mémorables
- Activité cérébrale proche de l'éveil
- Régulation émotionnelle

**Signature EEG :**
- **Activité mixte** : Ressemble à l'éveil !
- **Ondes Theta** (4-8 Hz)
- **Ondes Beta/Gamma** (>15 Hz)
- Amplitude : Faible, 20-50 μV
- Pattern : Désynchronisé, rapide

**Proportion du sommeil :** 20-25%

**Exemple de tracé :**
```
REM (mixte rapide)
  ╱╲╱╲‾╲╱╲╱‾╲╱╲╱╲╱‾    ← Ondes rapides mixtes
                         ← Ressemble à l'éveil !
```

**Pourquoi "paradoxal" ?**
- Cerveau actif (comme éveil) MAIS corps paralysé
- Activité cérébrale intense mais sommeil profond
- Paradoxe entre activité mentale et immobilité physique

---

### 📊 Cycle du Sommeil (Hypnogramme)

Une nuit typique de 8 heures :
```
Éveil   ████░░░░░░░░░░░░░░░░░░░░░░░░
REM     ░░░░░░░░░░██░░░░░███░░░████░
N1      ░██░░░░░░░░░░░░░░░░░░░░░░░░░
N2      ░░░███░░██░░░████░░░░██░░░░░
N3      ░░░░░░██░░██░░░░░░░░░░░░░░░░
        ├───────┼───────┼───────┼────
        0h     3h      6h      8h

Cycle 1    Cycle 2    Cycle 3    Cycle 4
(90-110 min par cycle)
```

**Progression typique :**
1. **Début de nuit** : Plus de N3 (sommeil profond)
2. **Milieu de nuit** : Alternance N2/N3
3. **Fin de nuit** : Plus de REM (rêves)

---

### 🎯 Pourquoi la Classification est Difficile ?

#### 1. **Similarités entre stades**

| Confusion Fréquente | Raison |
|---------------------|--------|
| **Wake ↔ REM** | Tracés EEG très similaires (activité rapide) |
| **N1 ↔ N2** | Transition progressive, pas de frontière nette |
| **N2 ↔ N3** | Dépend du % de Delta (seuil : 20%) |

#### 2. **Variabilité inter-individuelle**
- Âge (enfants vs adultes vs seniors)
- Médicaments (somnifères, antidépresseurs)
- Pathologies (apnée, insomnie)
- Qualité du signal (artéfacts, bruit)

#### 3. **Artefacts fréquents**
- Mouvements du patient
- Contractions musculaires
- Interférences électriques
- Électrodes mal positionnées

#### 4. **Expertise requise**
- **4 heures d'analyse manuelle** par un expert
- Nécessite formation spécialisée (polysomnographe)
- Fatigue de l'analyste → erreurs

---

### 🤖 Apport du Machine Learning

**Ce que le modèle apprend à reconnaître :**

| Stade | Features Clés Apprises |
|-------|------------------------|
| **Wake** | Alpha fort (8-13 Hz), Beta présent, faible Delta |
| **N1** | Theta émergent, Alpha disparaît, transition |
| **N2** | Fuseaux (12-14 Hz), Complexes K, Theta stable |
| **N3** | Delta massif (0.5-4 Hz), haute amplitude |
| **REM** | Mixte rapide, faible amplitude, ressemble Wake |

**Défis pour le ML :**
- ❌ **Wake vs REM** : Très similaires en fréquence
- ❌ **Transitions** : Frontières floues entre stades
- ❌ **REM sous-représenté** : Seulement 5% des données
- ✅ **N3 facile** : Delta très distinctif
- ✅ **N2 bon** : Fuseaux caractéristiques

**Résultat actuel du modèle :**
- ✅ N3 : 73% F1 (excellent)
- ✅ N2 : 68% F1 (bon)
- ⚠️ Wake : 59% F1 (moyen)
- ⚠️ N1 : 50% F1 (moyen)
- ❌ REM : 43% F1 (difficile)

---

### 📚 Références Médicales

- **AASM Scoring Manual** : Standard international de classification
- **Rechtschaffen & Kales (1968)** : Première classification standardisée
- **Berry et al. (2012)** : Règles actualisées de scoring

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- pip
- Git
- Docker (optionnel)

### Installation Locale
```bash
# 1. Cloner le repository
git clone https://github.com/marinedde/sleepai.git
cd sleepai

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import sklearn, fastapi, numpy; print('✅ Installation OK')"
```

### Installation Docker
```bash
# Build l'image
docker build -t sleepai:latest .

# Run le conteneur
docker run -d -p 8000:8000 --name sleepai sleepai:latest

# Vérifier
curl http://localhost:8000/health
```

---

## 🎮 Utilisation

### 1. Lancer l'API
```bash
# Démarrer l'API FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**L'API est accessible sur :** `http://localhost:8000`  
**Documentation interactive :** `http://localhost:8000/docs`

### 2. Lancer le Dashboard Streamlit
```bash
# Dans un nouveau terminal
streamlit run dashboard/streamlit_app.py
```

**Le dashboard est accessible sur :** `http://localhost:8501`

### 3. Faire une Prédiction (Python)
```python
import requests
import numpy as np

# Générer un signal de test (30 secondes à 100 Hz)
signal = np.random.randn(3000).tolist()

# Appeler l'API
response = requests.post(
    "http://localhost:8000/predict",
    json={"signal": signal}
)

# Afficher le résultat
result = response.json()
print(f"Stade prédit: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")
print(f"Probabilités: {result['probabilities']}")
```

### 4. Utiliser l'API en Production
```bash
# Health check
curl https://sleepai-api.onrender.com/health

# Informations du modèle
curl https://sleepai-api.onrender.com/model-info

# Monitoring
curl https://sleepai-api.onrender.com/monitoring/stats
```

> ⚠️ **Note Render** : L'instance gratuite se met en veille après 15 min d'inactivité. La première requête peut prendre 30-60 secondes.

---

## 📡 API Endpoints

### Principaux Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil avec liste des endpoints |
| `/health` | GET | Health check de l'API |
| `/model-info` | GET | Informations du modèle ML |
| `/predict` | POST | Prédiction de stade de sommeil |
| `/docs` | GET | Documentation Swagger interactive |

### Endpoints de Monitoring

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/monitoring/stats` | GET | Statistiques des prédictions |
| `/monitoring/drift` | GET | Détection de drift du modèle |
| `/monitoring/recent` | GET | Dernières prédictions loggées |

### Détails des Endpoints

#### `GET /health`

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/rf_v2_final_pipeline.joblib"
}
```

#### `GET /model-info`

**Réponse :**
```json
{
  "model_type": "Random Forest with Feature Engineering",
  "accuracy": 0.6462,
  "f1_score": 0.6377,
  "cohens_kappa": 0.5247,
  "classes": ["Wake", "N1", "N2", "N3", "REM"],
  "n_features": 16,
  "training_date": "2025-10-16"
}
```

#### `POST /predict`

**Requête :**
```json
{
  "signal": [0.1, 0.2, 0.3, ..., 0.5]  // 3000 valeurs
}
```

**Réponse :**
```json
{
  "predicted_class": "N2",
  "predicted_index": 2,
  "confidence": 0.71,
  "probabilities": {
    "Wake": 0.05,
    "N1": 0.10,
    "N2": 0.71,
    "N3": 0.12,
    "REM": 0.02
  }
}
```

#### `GET /monitoring/stats`

**Réponse :**
```json
{
  "total_predictions": 150,
  "time_range": {
    "first": "2025-10-20T09:00:00",
    "last": "2025-10-20T12:30:00"
  },
  "class_distribution": {
    "Wake": 15,
    "N1": 30,
    "N2": 65,
    "N3": 35,
    "REM": 5
  },
  "confidence_stats": {
    "mean": 0.68,
    "std": 0.15,
    "min": 0.42,
    "max": 0.95
  },
  "avg_processing_time_ms": 45.2
}
```

---

## 📊 Performance du Modèle

### Métriques Globales

- **Accuracy** : 64.6%
- **F1-Score (weighted)** : 0.638
- **Cohen's Kappa** : 0.525
- **Temps d'inférence** : ~30-50 ms

### Performance par Classe

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Wake   | 0.61      | 0.58   | 0.59     | 57      |
| N1     | 0.52      | 0.48   | 0.50     | 113     |
| N2     | 0.66      | 0.71   | 0.68     | 173     |
| N3     | 0.72      | 0.75   | 0.73     | 165     |
| REM    | 0.58      | 0.35   | 0.43     | 46      |

**Analyse :**
- ✅ **Meilleure performance** : N3 (sommeil profond) - F1: 0.73
- ✅ **Bonne performance** : N2 (sommeil intermédiaire) - F1: 0.68
- ⚠️ **Performance moyenne** : Wake et N1 - F1: ~0.55
- ❌ **Difficulté** : REM (sous-représenté) - F1: 0.43

### Matrice de Confusion
```
Prédictions →    Wake   N1    N2    N3   REM
Réel ↓
Wake        [     33    12     8     3     1  ]
N1          [     10    54    35    11     3  ]
N2          [      5    19   123    23     3  ]
N3          [      1     7    32   124     1  ]
REM         [      4    10    16     0    16  ]
```

### Comparaison des Modèles

| Modèle | Accuracy | F1-Score | Taille | Temps |
|--------|----------|----------|--------|-------|
| Baseline (signaux bruts) | 34% | 0.32 | 20 MB | ~50 ms |
| **RF Pipeline (prod)** | **64.6%** | **0.64** | **69 MB** | **~30 ms** |
| CNN 1D (expérimental) | 67% | 0.65 | 15 MB | ~80 ms |

> 💡 Le **Random Forest** est en production car plus stable et rapide que le CNN.

---

## 💼 Impact Business Potentiel

### Réduction du Temps d'Analyse

| Méthode | Temps par Patient | Coût Estimé* |
|---------|-------------------|--------------|
| **Analyse manuelle** | 4 heures | 400€ |
| **Avec SleepAI (POC)** | 30 minutes | 50€ |
| **Gain potentiel** | **-87.5%** | **-87.5%** |

*Basé sur un taux horaire médical de 100€/h

### Bénéfices Potentiels

**Pour les médecins :**
- ⏰ **Gain de temps** : 3h30 par patient libérées
- 📊 **Pré-analyse automatique** : Résultats suggérés pour validation
- 🎯 **Focus sur les cas complexes** : Plus de temps pour diagnostics difficiles

**Pour les patients :**
- 📅 **Délais réduits** : Résultats disponibles plus rapidement
- 💰 **Coûts potentiellement réduits** : Optimisation du temps médical
- 🏥 **Meilleur accès aux soins** : Plus de disponibilités

**Pour le système de santé :**
- 📈 **Capacité augmentée** : Traiter 8x plus de patients par jour
- 💵 **Économies** : Réduction des coûts de diagnostic
- 🌍 **Accessibilité** : Solution open-source vs logiciels propriétaires (10-20k€)

### Retour sur Investissement (ROI)

Avec **100 patients/mois** :
- **Temps gagné** : 350 heures/mois
- **Économies** : ~35 000€/mois
- **ROI** : Rentabilité dès le 1er mois (solution open-source)

> ⚠️ Ces chiffres sont des **estimations théoriques** basées sur le POC actuel. Un déploiement réel nécessiterait une validation clinique rigoureuse avec des médecins du sommeil.

---

## 🛠️ Technologies

### Machine Learning
- **Scikit-learn** 1.3.2 : Pipeline ML, Random Forest
- **NumPy** 1.24.3 : Calculs numériques
- **SciPy** 1.11.4 : Traitement du signal (FFT, filtrage)
- **Imbalanced-learn** 0.11.0 : Gestion des classes déséquilibrées

### API & Backend
- **FastAPI** 0.109.0 : Framework API REST
- **Uvicorn** 0.27.0 : Serveur ASGI
- **Pydantic** 2.5.3 : Validation des données

### Frontend
- **Streamlit** : Dashboard interactif
- **Matplotlib** : Visualisations EEG

### DevOps & MLOps
- **Docker** : Containerisation
- **Render** : Hébergement cloud
- **GitHub Actions** : CI/CD
- **pytest** : Tests automatisés (7 tests)

### Data Science
- **Jupyter** : Notebooks d'exploration
- **Pandas** : Manipulation de données

---

## 🔍 Feature Engineering

### Features Temporelles (8)
1. **Mean** : Moyenne du signal
2. **Std** : Écart-type
3. **Min** : Valeur minimale
4. **Max** : Valeur maximale
5. **Q1** : 1er quartile (25%)
6. **Q3** : 3ème quartile (75%)
7. **Skewness** : Asymétrie de la distribution
8. **Kurtosis** : Aplatissement de la distribution

### Features Fréquentielles (5)

Obtenues par transformée de Fourier (FFT) :

9. **Delta Power** (0.5-4 Hz) : Sommeil profond
10. **Theta Power** (4-8 Hz) : Somnolence
11. **Alpha Power** (8-13 Hz) : Relaxation
12. **Beta Power** (13-30 Hz) : Éveil actif
13. **Gamma Power** (30-50 Hz) : Cognition

### Ratios de Puissance (3)

14. **Delta/Total** : Ratio normalisé
15. **Theta/Total** : Ratio normalisé
16. **Alpha/Total** : Ratio normalisé

---

## 📈 Monitoring & Logs

### Système de Monitoring

Le système log automatiquement chaque prédiction dans `logs/predictions.jsonl` :
```json
{
  "timestamp": "2025-10-20T10:30:15.123456",
  "prediction": "N2",
  "confidence": 0.71,
  "probabilities": {
    "Wake": 0.05,
    "N1": 0.10,
    "N2": 0.71,
    "N3": 0.12,
    "REM": 0.02
  },
  "signal_stats": {
    "mean": 0.02,
    "std": 0.85,
    "min": -2.1,
    "max": 2.3,
    "length": 3000
  },
  "processing_time_ms": 45.2
}
```

### Détection de Drift

Le système peut détecter une dérive du modèle en comparant :
- Les confidences récentes vs anciennes
- La distribution des classes prédites
- Les temps de traitement

**Exemple de drift détecté :**
```json
{
  "drift_detected": true,
  "confidence_drift": {
    "recent_avg": 0.55,
    "older_avg": 0.72,
    "difference": 0.17,
    "threshold": 0.10
  },
  "recommendation": "Retrain model"
}
```

---

## 🧪 Tests

### Lancer les Tests
```bash
# Tous les tests
pytest tests/ -v

# Avec coverage
pytest tests/ --cov=app --cov-report=html

# Voir le rapport
open htmlcov/index.html
```

### Résultats des Tests
```
✅ test_api_running - PASSED
✅ test_health_response_structure - PASSED
✅ test_openapi_docs - PASSED
✅ test_openapi_schema - PASSED
✅ test_predict_validation_empty - PASSED
✅ test_predict_validation_short - PASSED
✅ test_predict_validation_missing - PASSED
⏭️  test_model_info_if_loaded - SKIPPED (si modèle non chargé)
⏭️  test_predict_if_loaded - SKIPPED (si modèle non chargé)

7 passed, 2 skipped - Coverage: 41%
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

À chaque push sur `main`, le pipeline exécute :

1. **Tests** ✅
   - Installation des dépendances
   - Exécution de pytest
   - Génération du rapport de coverage

2. **Lint** ✅
   - Vérification avec flake8
   - Détection des erreurs de syntaxe

3. **Build Docker** ✅
   - Construction de l'image
   - Test du conteneur
   - Vérification du health check

4. **Deploy** ✅
   - Notification à Render
   - Déploiement automatique

### Status du Pipeline

![CI/CD Status](https://github.com/marinedde/sleepai/actions/workflows/deploy.yml/badge.svg)

**Voir les runs :** [GitHub Actions](https://github.com/marinedde/sleepai/actions)

---

## ⚠️ Limitations & Améliorations Futures

### Limitations Actuelles

1. **Performance** : 64.6% accuracy 
   - ✅ Acceptable pour un POC technique
   - ❌ Insuffisant pour usage médical (objectif : >85%)
   
2. **Dataset limité** : Sleep-EDF uniquement (2584 échantillons)
   - Manque de diversité (âge, pathologies, ethnicités)
   - Données publiques, pas de cas cliniques variés

3. **Classes déséquilibrées** : REM sous-représenté (5%)
   - Impact sur les performances de détection du REM
   - F1-score REM : 0.43 seulement

4. **Architecture simple** : Random Forest
   - Pas de capture des dépendances temporelles
   - Features manuelles vs apprentissage automatique des patterns

5. **Pas de certification médicale**
   - Aucune validation clinique
   - Non conforme aux normes médicales (CE/FDA)
   - Ne peut pas être utilisé pour du diagnostic réel

### Expérimentations Réalisées

**CNN 1D testé :**
- Architecture : 3 couches convolutionnelles + pooling
- Performance : **67% accuracy** (+2.4% vs RF)
- F1-Score : 0.65
- Conclusion : Prometteur mais nécessite plus de données et optimisation
- Code disponible : `notebooks/cnn_model.ipynb`

### Roadmap V2.0

#### Court Terme (1-2 mois)
- [ ] **CNN 1D optimisé** : Pousser à 70%+ avec data augmentation
- [ ] **LSTM/Bi-LSTM** : Capturer les dépendances temporelles longues
- [ ] **Ensemble methods** : Combiner RF + CNN + LSTM (voting/stacking)
- [ ] **Dataset augmenté** : Collaborer avec hôpitaux (objectif : 10k+ échantillons)
- [ ] **Support EDF natif** : Lire directement les fichiers .edf
- [ ] **Export PDF** : Rapports automatiques pour médecins

#### Moyen Terme (3-6 mois)
- [ ] **Validation clinique** : Tests avec médecins du sommeil
- [ ] **Détection apnées** : Nouvelle fonctionnalité (SpO2, débit respiratoire)
- [ ] **Multi-canal** : Intégrer EOG, EMG, ECG
- [ ] **Interface professionnelle** : Dashboard pour médecins
- [ ] **Authentification** : Gestion sécurisée des patients (OAuth2)
- [ ] **Monitoring avancé** : Détection de drift, alertes automatiques

#### Long Terme (1 an+)
- [ ] **Certification CE/FDA** : Conformité dispositif médical
- [ ] **Hébergement HDS** : Données de santé sécurisées
- [ ] **RGPD/HIPAA** : Conformité internationale
- [ ] **Application mobile** : Pour patients et médecins
- [ ] **Intégration FHIR** : Interopérabilité avec dossiers médicaux électroniques
- [ ] **Objectif final** : **>90% accuracy** avec validation clinique

### Vision Long Terme

Créer une **solution open-source et accessible** qui aide les médecins à :
- ⏰ Réduire le temps d'analyse de **4h à 30 minutes**
- 💰 Économiser **~35 000€/mois** pour un cabinet traitant 100 patients
- 🌍 Démocratiser l'accès aux diagnostics du sommeil (vs 10-20k€ logiciels propriétaires)
- 🤝 Maintenir le **contrôle médical** (IA en support, pas en remplacement)

---

## 📚 Documentation Technique

### Structure du Projet
```
sleepai/
├── app/                          # Code source de l'API
│   ├── __init__.py
│   ├── main.py                   # FastAPI app + endpoints
│   ├── models.py                 # Modèles Pydantic (validation)
│   ├── ml_model.py               # Wrapper du modèle ML
│   ├── feature_extractor.py      # Extraction de features
│   └── monitoring.py             # Système de monitoring
│
├── dashboard/                    # Interface Streamlit
│   └── streamlit_app.py          # Dashboard interactif
│
├── models/                       # Modèles ML sauvegardés
│   └── rf_v2_final_pipeline.joblib  # Pipeline production (69 MB)
│
├── notebooks/                    # Notebooks d'analyse
│   ├── data_exploration.ipynb    # EDA
│   ├── model_training.ipynb      # Entraînement RF
│   └── cnn_model.ipynb           # Expérimentations CNN
│
├── tests/                        # Tests unitaires
│   ├── __init__.py
│   ├── conftest.py               # Configuration pytest
│   └── test_api.py               # Tests des endpoints
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD pipeline
│
├── logs/                         # Logs de monitoring (git ignored)
│   └── predictions.jsonl         # Prédictions loggées
│
├── Dockerfile                    # Configuration Docker
├── requirements.txt              # Dépendances Python
├── pytest.ini                    # Configuration pytest
├── .gitignore
└── README.md
```

### Dépendances Principales
```txt
# API
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3

# ML
scikit-learn==1.3.2
numpy==1.24.3
scipy==1.11.4
joblib==1.3.2
imbalanced-learn==0.11.0

# Tests
httpx
pytest
pytest-cov
```

---

## 🌐 Déploiement Production

**API en production :** https://sleepai-api.onrender.com

### Configuration Render

Le projet est configuré pour un déploiement automatique sur Render :

- **Type** : Web Service
- **Runtime** : Docker
- **Plan** : Free (avec limitations)
- **Auto-deploy** : Activé (sur push main)
- **Health Check** : `/health`

### Limitations du Plan Gratuit

⚠️ **Important à savoir :**
- **Mise en veille** : Après 15 min d'inactivité
- **Réveil** : Première requête peut prendre 30-60 secondes
- **RAM** : 512 MB (suffisant pour le modèle)
- **Bande passante** : Limitée

---

## 👥 Auteur

**Marine Deldicque**  
Lead Data Science - Jedha Bootcamp 2025

- 🐙 GitHub : [@marinedde](https://github.com/marinedde)
- 📧 Email : [marine.deldicque@gmail.com]
- 💼 LinkedIn : [Marine Deldicque]

**Projet** : Formation Lead Data Science MLOps  
**Date** : Octobre 2025  
**Encadrant** : Raphaël Rialland (Jedha)

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

**En résumé :**
- ✅ Utilisation libre (personnel et commercial)
- ✅ Modification et distribution autorisées
- ✅ Attribution requise
- ⚠️ Aucune garantie fournie

---

## 🙏 Remerciements

- **Jedha Bootcamp** : Formation Lead Data Science et accompagnement
- **Raphaël RIALLAND** : Teacher Assistant
- **Physionet** : Sleep-EDF Database (données d'entraînement)
- **Open Source Community** : FastAPI, Scikit-learn, Streamlit
- **Médecins du sommeil** : Insights sur la problématique métier

---

## 📖 Références

### Académiques
1. Iber, C., et al. (2007). *The AASM Manual for the Scoring of Sleep and Associated Events*
2. Kemp, B., et al. (2000). *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG*
3. Goldberger, A., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals*
4. Berry, R. B., et al. (2012). *Rules for Scoring Respiratory Events in Sleep: Update of the 2007 AASM Manual*

### Techniques
- **FastAPI Documentation** : https://fastapi.tiangolo.com/
- **Scikit-learn User Guide** : https://scikit-learn.org/stable/user_guide.html
- **Streamlit Documentation** : https://docs.streamlit.io/
- **Sleep-EDF Database** : https://physionet.org/content/sleep-edfx/1.0.0/

---

## 🔗 Liens Utiles

- 🌐 **API Déployée** : https://sleepai-api.onrender.com
- 📚 **Documentation API** : https://sleepai-api.onrender.com/docs
- 🐙 **GitHub Repository** : https://github.com/marinedde/sleepai
- 🔄 **GitHub Actions** : https://github.com/marinedde/sleepai/actions
- 📊 **Dataset Sleep-EDF** : https://physionet.org/content/sleep-edfx/1.0.0/

---

<div align="center">

### ⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐

**Made with ❤️ and ☕ by Marine Deldicque**

*"Démocratiser l'accès aux diagnostics du sommeil grâce à l'IA"*

---

**[⬆ Retour en haut](#-sleepai---classification-automatique-des-stades-de-sommeil)**

</div>