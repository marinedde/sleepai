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
│   └── main.py                 # Endpoints REST + monitoring drift
├── dashboard/
│   └── streamlit_app.py        # Dashboard (EEG + ECG + rapport IA + validation clinique)
├── models/                     # Modèles entraînés (.joblib)
├── notebooks/                  # Exploration et entraînement
├── tests/                      # Tests unitaires
├── mlruns/                     # Tracking MLflow
├── data/demo/                  # Signaux démo (.npy)
├── Dockerfile                  # Containerisation
├── requirements.txt            # Dépendances
└── .github/workflows/          # CI/CD GitHub Actions
```

---

## 🖥️ Fonctionnalités du Dashboard

**Page Analyse EEG — Classification des stades du sommeil**
- Visualisation du signal EEG brut + densité spectrale de puissance (bandes δ, θ, α, β)
- Prédiction du stade (Wake / N1 / N2 / N3 / REM) avec probabilités
- Validation clinique médecin : confirmation, correction, commentaire libre

**Page Analyse ECG — Détection des apnées**
- Visualisation du signal ECG + indicateurs SpO₂, AHI
- Prédiction binaire (Normal / Apnée) avec probabilités
- Données cliniques complémentaires (ronflement, somnolence diurne)
- Validation clinique médecin intégrée

**Page Rapport complet**
- Synthèse EEG + ECG combinée
- Rapport clinique généré par IA (Claude — Anthropic) en < 5 secondes
- Export `.txt` et `.json`
- Mode hors-ligne si clé API non disponible

**Page Monitoring**
- Statistiques d'utilisation en temps réel (total, EEG, ECG, confiance moyenne)
- Détection de drift du modèle
- Historique des 10 dernières prédictions

---

## 🤖 Modèles

### Modèle 1 — Classification des stades du sommeil (EEG)
- **Algorithme** : Random Forest (scikit-learn)
- **Dataset** : [Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet) — signaux EEG polysomnographiques
- **Features** : 16 features temporelles et spectrales extraites du signal EEG
- **Classes** : Wake, N1, N2, N3, REM
- **Endpoint API** : `POST /predict/sleep-stage`

### Modèle 2 — Détection des apnées du sommeil (ECG)
- **Algorithme** : Random Forest (scikit-learn)
- **Dataset** : [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/) (PhysioNet) — signaux ECG avec annotations minute par minute
- **Features** : features spectrales et temporelles extraites du signal ECG
- **Classes** : Normal, Apnée
- **Endpoint API** : `POST /predict/apnea`

**Tracking des expériences** : MLflow

---

## ⚙️ Pipeline MLOps

```
Signal EEG brut                Signal ECG brut
      ↓                               ↓
Extraction de features         Extraction de features
(16 features EEG)              (features ECG)
      ↓                               ↓
StandardScaler → RF            StandardScaler → RF
(stades sommeil)               (détection apnée)
      ↓                               ↓
      └──────────────┬────────────────┘
                     ↓
         API FastAPI (Docker)
                     ↓
         Dashboard Streamlit
                     ↓
    CI/CD GitHub Actions → HuggingFace Spaces
```

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|-------------|
| Modèles | scikit-learn, Random Forest |
| Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| IA générative | Claude (Anthropic API) — rapports cliniques |
| Containerisation | Docker |
| Déploiement | HuggingFace Spaces |
| CI/CD | GitHub Actions |

---

## 📂 Données

Les données ne sont pas incluses dans ce repo.

| Dataset | Source | Usage |
|---------|--------|-------|
| Sleep-EDF Database | [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) | Classification des stades EEG |
| Apnea-ECG Database | [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/) | Détection d'apnées ECG |

Placer les fichiers `.edf` dans le dossier `data/raw/` avant de lancer les notebooks.

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
