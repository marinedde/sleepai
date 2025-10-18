# 🧠 SleepAI - Pipeline MLOps pour l'Analyse de Polysomnographie

> Pipeline MLOps complet pour la classification automatique des stades de sommeil à partir de signaux EEG polysomnographiques.

**Technologies :** Python 3.10 | FastAPI | Docker | Scikit-learn | GitHub Actions

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Endpoints](#-api-endpoints)
- [Modèles](#-modèles)
- [Technologies](#-technologies)
- [Roadmap](#-roadmap)

---

## 🎯 À propos

Ce projet implémente un pipeline MLOps de bout en bout pour la classification automatique des stades de sommeil (Wake, N1, N2, N3, REM) à partir de signaux EEG. Il s'inscrit dans le cadre du programme de formation Jedha Bootcamp et démontre la maîtrise des pratiques MLOps modernes.

**Objectif :** Créer une solution déployable en production pour assister les médecins dans l'analyse du sommeil.

---

## 🏗️ Architecture

```
sleepai/
├── app/                          # Application FastAPI
│   ├── main.py                  # API endpoints
│   ├── ml_model.py              # Gestion du modèle ML
│   ├── feature_extractor.py    # Extraction de features
│   └── models.py                # Schémas Pydantic
│
├── models/                       # Modèles ML sérialisés
│   ├── rf_v2_pipeline.joblib    # Pipeline Random Forest (production)
│   ├── rf_v3_best_params.pkl    # Hyperparamètres optimisés
│   └── rf_v3_feature_names.pkl  # Noms des features
│
├── notebooks/                    # Notebooks d'exploration
│   ├── preprocessing.ipynb      # Prétraitement des données
│   ├── model_training.ipynb     # Entraînement des modèles
│   └── cnn_model.ipynb          # Expérimentations CNN
│
├── data/                         # Données (non versionnées)
│   ├── raw/                     # Données brutes
│   └── processed/               # Données prétraitées
│
├── Dockerfile                    # Configuration Docker
├── docker-compose.yml           # Orchestration Docker
└── requirements.txt             # Dépendances Python
```

---

## ✨ Fonctionnalités

### Phase 1 : Pipeline ML ✅
- Prétraitement des signaux EEG
- Extraction de features temporelles et fréquentielles
- Entraînement Random Forest avec validation croisée
- Optimisation des hyperparamètres
- Sérialisation du pipeline complet

### Phase 2 : API REST ✅
- API FastAPI avec documentation automatique (Swagger)
- Endpoint de santé (`/health`)
- Endpoint de prédiction (`/predict`)
- Chargement automatique du modèle au démarrage
- Validation des données avec Pydantic

### Phase 3 : Dockerization ✅
- Container Docker optimisé
- Image Python 3.10-slim
- Orchestration avec docker-compose
- Volumes pour modèles et données
- Health checks automatiques

### Phase 4 : CI/CD (En cours)
- Tests automatisés avec GitHub Actions
- Build et push d'images Docker
- Déploiement automatique

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Docker et Docker Compose
- Git

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/marinedde/sleepai.git
cd sleepai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🐳 Utilisation avec Docker (Recommandé)

### Lancer l'API

```bash
# Build et lancement
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down
```

L'API sera accessible sur `http://localhost:8000`

### Documentation interactive

Une fois l'API lancée, accédez à la documentation Swagger :

```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/rf_v2_pipeline.joblib"
}
```

### Prédiction

```bash
POST /predict
Content-Type: application/json

{
  "signal": [0.1, 0.2, 0.3, ...],
  "sampling_rate": 100
}
```

**Réponse :**
```json
{
  "prediction": "N2",
  "confidence": 0.87,
  "probabilities": {
    "Wake": 0.05,
    "N1": 0.03,
    "N2": 0.87,
    "N3": 0.02,
    "REM": 0.03
  }
}
```

---

## 🤖 Modèles

### Modèle en Production

**Random Forest Pipeline v2** (`rf_v2_pipeline.joblib`)
- **Taille :** 7 MB
- **Features :** Temporelles + Fréquentielles (FFT, Puissance des bandes)
- **Performance :**
  - Accuracy : ~65%
  - F1-Score weighted : ~0.64
  - Cohen's Kappa : ~0.52
- **Architecture :** Pipeline Scikit-learn complet (extraction de features + scaling + classification)

> **Note :** Les performances sont perfectibles mais suffisantes pour démontrer un pipeline MLOps complet. L'accent a été mis sur l'infrastructure de déploiement plutôt que l'optimisation du modèle.

### Features extraites

- **Temporelles :** Mean, Std, Min, Max, Skewness, Kurtosis
- **Fréquentielles :** Puissance des bandes Delta, Theta, Alpha, Beta, Gamma
- **Statistiques :** Zero-crossing rate, Energy

### Modèles expérimentaux (non versionnés)

- `random_forest_baseline.pkl` (20 MB) - Version initiale
- `rf_v3_final.pkl` (69 MB) - Version optimisée avec plus de features
- `cnn_best_model.h5` (4.3 MB) - Expérimentations CNN

> Ces modèles sont disponibles sur demande ou peuvent être réentraînés avec les notebooks fournis.

---

## 🛠️ Technologies

### Machine Learning
- **Scikit-learn** : Pipeline ML, Random Forest
- **NumPy / Pandas** : Manipulation de données
- **SciPy** : Traitement du signal

### API & Backend
- **FastAPI** : Framework API REST
- **Uvicorn** : Serveur ASGI
- **Pydantic** : Validation de données

### DevOps & MLOps
- **Docker** : Containerisation
- **Docker Compose** : Orchestration
- **GitHub Actions** : CI/CD (à venir)

### Data Science
- **Jupyter** : Notebooks d'exploration
- **Matplotlib / Seaborn** : Visualisation

---

## 📈 Roadmap

### ✅ Complété
- [x] Pipeline ML complet
- [x] API REST FastAPI
- [x] Dockerization
- [x] Documentation

### 🔄 En cours
- [ ] CI/CD avec GitHub Actions
- [ ] Tests unitaires et d'intégration
- [ ] Dashboard Streamlit

### 🔮 Futur
- [ ] Déploiement cloud (Render / Railway)
- [ ] Monitoring et logging avancé
- [ ] Détection de drift
- [ ] Support de formats EDF supplémentaires

---

## 📊 Performances

| Modèle | Accuracy | F1-Score | Taille | Temps inférence |
|--------|----------|----------|--------|-----------------|
| RF Baseline | 34% | 0.32 | 20 MB | ~50ms |
| RF v2 Pipeline | 65% | 0.64 | 7 MB | ~30ms |

> **Note :** Le focus du projet est sur le déploiement MLOps plutôt que l'optimisation du modèle. Les performances sont acceptables pour une démonstration de pipeline complet.

---

## 👥 Auteur

**Marine Deldicque**
- GitHub : [@marinedde](https://github.com/marinedde)
- Projet Jedha Bootcamp - MLOps

---

## 📄 License

Ce projet est sous licence MIT.

---

## 🙏 Remerciements

- Jedha Bootcamp pour le programme de formation
- Dataset Sleep-EDF pour les données d'entraînement
- Communauté FastAPI et Scikit-learn

---

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.