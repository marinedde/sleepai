# 🧠 SleepAI — Pipeline MLOps pour l'Analyse Polysomnographique

> Pipeline MLOps complet pour la classification automatique des stades de sommeil (Wake, N1, N2, N3, REM) à partir de signaux EEG polysomnographiques.

**Jedha Bootcamp — AIA 2026 | Marine Deldicque**

---

## 🚀 Démos en ligne

| Composant | URL |
|-----------|-----|
| 🔬 API FastAPI | [marinedde-sleepai-api.hf.space](https://marinedde-sleepai-api.hf.space/docs) |
| 🧠 Dashboard Streamlit | [marinedde-sleepai-dashboard.hf.space](https://marinedde-sleepai-dashboard.hf.space) |

---

## 🏗️ Architecture

```
sleepai/
├── app/                        # API FastAPI
│   └── main.py                 # Endpoints REST
├── dashboard/
│   └── streamlit_app.py        # Dashboard interactif
├── models/                     # Modèles entraînés (.joblib)
├── notebooks/                  # Exploration et entraînement
├── tests/                      # Tests unitaires
├── mlruns/                     # Tracking MLflow
├── Dockerfile                  # Containerisation
├── requirements.txt            # Dépendances
└── .github/workflows/          # CI/CD GitHub Actions
```

---

## 🤖 Modèle

- **Algorithme** : Random Forest (scikit-learn)
- **Dataset** : Sleep-EDF (PhysioNet) — signaux EEG polysomnographiques
- **Features** : 16 features temporelles et spectrales extraites du signal EEG
- **Classes** : Wake, N1, N2, N3, REM
- **Tracking** : MLflow

---

## ⚙️ Pipeline MLOps

```
Signal EEG brut
      ↓
Extraction de features (16 features)
      ↓
StandardScaler → Random Forest
      ↓
API FastAPI (Docker) ←→ Dashboard Streamlit
      ↓
CI/CD GitHub Actions → HuggingFace Spaces
```

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|-------------|
| Modèle | scikit-learn, Random Forest |
| Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Containerisation | Docker |
| Déploiement | HuggingFace Spaces |
| CI/CD | GitHub Actions |

---

## 📂 Données

Les données ne sont pas incluses dans ce repo.

- Dataset Sleep-EDF : [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

Placer les fichiers `.edf` dans le dossier `data/` avant de lancer les notebooks.

---

## ▶️ Installation locale

```bash
git clone https://github.com/marinedde/sleepai
cd sleepai
pip install -r requirements.txt

# Lancer l'API
uvicorn app.main:app --reload

# Lancer le dashboard
streamlit run dashboard/streamlit_app.py
```

---

## Auteure

**Marine Deldicque** — Jedha Bootcamp CDSD/AIA 2026
