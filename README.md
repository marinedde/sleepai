# 🧠 SleepAI - Pipeline MLOps pour l'Analyse de Polysomnographie

> Pipeline MLOps complet pour la classification automatique des stades de sommeil à partir de signaux EEG polysomnographiques.

**Technologies :** Python 3.10 | FastAPI | Docker | Scikit-learn | Streamlit | GitHub Actions

**🌐 API Déployée :** [https://sleepai-api.onrender.com](https://sleepai-api.onrender.com/docs)

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Démo en ligne](#-démo-en-ligne)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Endpoints](#-api-endpoints)
- [Dashboard](#-dashboard)
- [Modèles](#-modèles)
- [Technologies](#-technologies)
- [Performances](#-performances)
- [Perspectives](#-perspectives)

---

## 🎯 À propos

Ce projet implémente un pipeline MLOps de bout en bout pour la classification automatique des stades de sommeil (Wake, N1, N2, N3, REM) à partir de signaux EEG. Il s'inscrit dans le cadre du programme de formation Jedha Bootcamp et démontre la maîtrise des pratiques MLOps modernes.

**Contexte médical :** La polysomnographie prend en moyenne **4h d'analyse manuelle** par patient. Les logiciels actuels coûtent entre 10 000 et 20 000€ et n'intègrent pas de Machine Learning moderne. Ce projet propose une approche automatisée et accessible.

---

## 🚀 Démo en ligne

**API REST déployée :** [https://sleepai-api.onrender.com](https://sleepai-api.onrender.com/docs)

**Endpoints disponibles :**
- 🏥 Health check : `GET /health`
- 📊 Informations modèle : `GET /model-info`
- 🔮 Prédiction : `POST /predict`

**Exemple de requête :**
```bash
# Test de santé
curl https://sleepai-api.onrender.com/health

# Informations du modèle
curl https://sleepai-api.onrender.com/model-info
```

> **Note :** L'instance gratuite Render se met en veille après 15 minutes d'inactivité. La première requête peut prendre 30 secondes pour réveiller le service.

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
├── dashboard/                    # Interface Streamlit
│   ├── streamlit_app.py         # Application dashboard
│   └── requirements.txt         # Dépendances Streamlit
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
├── render.yaml                   # Configuration Render
└── requirements.txt             # Dépendances Python
```

**Architecture applicative :**
```
[Dashboard Streamlit] --HTTP--> [API FastAPI sur Render] --> [Modèle Random Forest]
```

---

## ✨ Fonctionnalités

### Phase 1 : Pipeline ML ✅
- Prétraitement des signaux EEG
- Extraction de 16 features (temporelles + fréquentielles)
- Entraînement Random Forest avec validation croisée
- Optimisation des hyperparamètres
- Sérialisation du pipeline complet

### Phase 2 : API REST ✅
- API FastAPI avec documentation automatique (Swagger)
- Endpoint de santé (`/health`)
- Endpoint d'information modèle (`/model-info`)
- Endpoint de prédiction (`/predict`)
- Chargement automatique du modèle au démarrage
- Validation des données avec Pydantic

### Phase 3 : Dockerization ✅
- Container Docker optimisé (Python 3.10-slim)
- Orchestration avec docker-compose
- Volumes pour modèles et données
- Déploiement sur Render

### Phase 4 : Dashboard Streamlit ✅
- Interface graphique interactive
- Visualisation des signaux EEG
- Prédiction en temps réel
- Affichage des probabilités par stade

### Phase 5 : CI/CD (En cours)
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
  "model_path": "/app/notebooks/models/rf_v2_pipeline_fixed.joblib"
}
```

### Informations du Modèle

```bash
GET /model-info
```

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

### Prédiction

```bash
POST /predict
Content-Type: application/json

{
  "signal": [0.1, 0.2, 0.3, ..., 0.5],
  "sampling_rate": 100
}
```

**Réponse :**
```json
{
  "prediction": "N2",
  "confidence": 0.65,
  "probabilities": {
    "Wake": 0.05,
    "N1": 0.15,
    "N2": 0.65,
    "N3": 0.10,
    "REM": 0.05
  },
  "processing_time_ms": 45
}
```

---

## 🎨 Dashboard

### Lancer le Dashboard localement

```bash
# Installer les dépendances
pip install -r dashboard/requirements.txt

# Lancer l'application
streamlit run dashboard/streamlit_app.py
```

Le dashboard sera accessible sur `http://localhost:8501`

### Fonctionnalités du Dashboard

- 📊 **Visualisation des signaux EEG** en temps réel
- 🔮 **Prédiction interactive** des stades de sommeil
- 📈 **Graphiques de probabilités** par classe
- 🎯 **Modes de génération** : signal synthétique ou aléatoire
- 📡 **Connexion à l'API** déployée sur Render

---

## 🤖 Modèles

### Modèle en Production

**Random Forest Pipeline v2** (`rf_v2_pipeline.joblib`)
- **Taille :** 7 MB
- **Features :** 16 features extraites
  - **Temporelles (8)** : Mean, Std, Min, Max, Q1, Q3, Skewness, Kurtosis
  - **Fréquentielles (5)** : Puissance Delta, Theta, Alpha, Beta, Gamma
  - **Ratios (3)** : Ratios de puissance normalisés
- **Performance :**
  - Accuracy : 65%
  - F1-Score weighted : 0.64
  - Cohen's Kappa : 0.52
- **Architecture :** Pipeline Scikit-learn complet (extraction + scaling + classification)

> **Note :** Les performances sont acceptables pour une preuve de concept. Le focus du projet est sur l'infrastructure MLOps plutôt que l'optimisation du modèle. Une amélioration future nécessitera plus de données et l'expertise de professionnels de santé.

### Features extraites

#### Statistiques temporelles (8 features)
- Amplitude moyenne et écart-type
- Min, Max, Q1, Q3
- Skewness (asymétrie) et Kurtosis (aplatissement)

#### Analyse spectrale (5 features)
- **Delta (0.5-4 Hz)** : Sommeil profond
- **Theta (4-8 Hz)** : Somnolence
- **Alpha (8-13 Hz)** : Relaxation
- **Beta (13-30 Hz)** : Éveil actif
- **Gamma (30-35 Hz)** : Cognition

#### Ratios de puissance (3 features)
- Ratios normalisés entre les bandes de fréquence

---

## 🛠️ Technologies

### Machine Learning
- **Scikit-learn** : Pipeline ML, Random Forest
- **NumPy / Pandas** : Manipulation de données
- **SciPy** : Traitement du signal (FFT, filtrage)

### API & Backend
- **FastAPI** : Framework API REST
- **Uvicorn** : Serveur ASGI
- **Pydantic** : Validation de données

### Frontend
- **Streamlit** : Dashboard interactif
- **Matplotlib** : Visualisations

### DevOps & MLOps
- **Docker** : Containerisation
- **Docker Compose** : Orchestration
- **Render** : Déploiement cloud
- **GitHub Actions** : CI/CD (à venir)

### Data Science
- **Jupyter** : Notebooks d'exploration
- **imbalanced-learn (SMOTE)** : Gestion des classes déséquilibrées

---

## 📊 Performances

| Modèle | Accuracy | F1-Score | Taille | Temps inférence |
|--------|----------|----------|--------|-----------------|
| RF Baseline (signaux bruts) | 34% | 0.32 | 20 MB | ~50ms |
| RF v2 Pipeline (features) | 65% | 0.64 | 7 MB | ~30ms |

### Matrice de confusion (Test set)

```
         Pred: Wake  N1   N2   N3  REM
Actual:
Wake        28      21    4    3    1
N1          17      66   24    1    5
N2           3      11  113   35   11
N3           1       0   24  140    0
REM          0      14   21    0   11
```

> **Note :** Le modèle performe bien sur N2 et N3 (sommeil profond), mais moins bien sur Wake et REM. Cela est cohérent avec la littérature scientifique où ces stades sont plus difficiles à distinguer.

---

## 🔮 Perspectives

### Limitations actuelles
- Accuracy à 65% (preuve de concept, objectif >90% pour usage clinique)
- Dataset limité (Sleep-EDF uniquement)
- Pas d'interface pour professionnels de santé
- Pas de certification médicale

### Évolutions futures

#### Court terme (1-3 mois)
- ✅ Amélioration du modèle (CNN, Transformers)
- ✅ Augmentation du dataset (collaboration médicale)
- ✅ Tests A/B sur plusieurs architectures
- ✅ Export de rapports PDF

#### Moyen terme (3-6 mois)
- 🔬 **Détection d'apnées du sommeil**
  - Nouvelles features (SpO2, débit respiratoire)
  - Collaboration avec pneumologues
  - Validation clinique
- 📊 Interface professionnelle pour médecins
- 🔐 Authentification et gestion des patients
- 📈 Monitoring et détection de drift

#### Long terme (6-12 mois)
- 🏥 **Certification dispositif médical** (marquage CE)
- 🔒 Hébergement de données de santé (HDS)
- 🌍 Conformité RGPD
- 📱 Application mobile pour patients
- 🔗 Intégration FHIR (interopérabilité)

### Vision

Créer une solution **open-source** et **accessible** pour aider les médecins de ville à réduire le temps d'analyse des polysomnographies de **4h à 30 minutes**, tout en maintenant une qualité diagnostique élevée.

---

## 📈 Roadmap MLOps

- [x] Pipeline ML complet et sérialisé
- [x] API REST FastAPI avec documentation
- [x] Dockerization complète
- [x] Déploiement sur Render
- [x] Dashboard Streamlit interactif
- [ ] CI/CD avec GitHub Actions
- [ ] Tests unitaires et d'intégration
- [ ] Monitoring et logging avancé
- [ ] Versioning des modèles (DVC ou MLflow)

---

## 👥 Auteur

**Marine Deldicque**
- GitHub : [@marinedde](https://github.com/marinedde)
- Projet : Jedha Bootcamp - MLOps
- Date : Octobre 2025

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **Jedha Bootcamp** pour le programme de formation MLOps
- **Sleep-EDF Database** pour les données d'entraînement (Physionet)
- Communautés **FastAPI**, **Scikit-learn** et **Streamlit**
- **Raphael** (mentor) pour les conseils stratégiques

---

## 📞 Contact

Pour toute question, suggestion ou collaboration :
- 📧 Ouvrir une **issue** sur GitHub
- 💬 Contacter via le profil GitHub

---

## 🔗 Liens utiles

- [API Documentation (Swagger)](https://sleepai-api.onrender.com/docs)
- [GitHub Repository](https://github.com/marinedde/sleepai)
- [Dataset Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**⭐ Si ce projet vous plaît, n'hésitez pas à le star sur GitHub ! ⭐**

Made with ❤️ and ☕ by Marine Deldicque

</div>