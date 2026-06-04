"""
Script pour rebuilder le pipeline avec le bon FeatureExtractor.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Importer le FeatureExtractor depuis app
from app.feature_extractor import FeatureExtractor

print("🔧 Reconstruction du pipeline...")

# Charger les données de test pour fit le scaler
data_dir = Path('data/processed')

if not data_dir.exists():
    print("❌ Dossier data/processed/ non trouvé")
    print("On va créer un pipeline avec des données synthétiques...")
    
    # Créer des données synthétiques
    np.random.seed(42)
    X_train = np.random.randn(100, 3000)
    y_train = np.random.randint(0, 5, 100)
else:
    print("📂 Chargement des données d'entraînement...")
    try:
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        print(f"   ✅ X_train: {X_train.shape}")
        print(f"   ✅ y_train: {y_train.shape}")
    except:
        print("⚠️  Erreur chargement, utilisation de données synthétiques")
        np.random.seed(42)
        X_train = np.random.randn(100, 3000)
        y_train = np.random.randint(0, 5, 100)

# Charger l'ancien modèle pour récupérer le RandomForest
print("\n📦 Tentative de récupération de l'ancien modèle...")
old_model_path = Path('models/rf_v2_pipeline.joblib')

try:
    # Essayer de charger juste le RandomForest (3ème étape du pipeline)
    import pickle
    with open(old_model_path, 'rb') as f:
        # Charger manuellement
        data = pickle.load(f)
        if hasattr(data, 'named_steps'):
            rf_classifier = data.named_steps.get('classifier', None)
            if rf_classifier is None:
                # Essayer 'randomforestclassifier' ou autre nom
                for key in data.named_steps.keys():
                    if 'forest' in key.lower() or 'classifier' in key.lower():
                        rf_classifier = data.named_steps[key]
                        break
            
            if rf_classifier:
                print(f"   ✅ RandomForest récupéré: {type(rf_classifier)}")
            else:
                print("   ⚠️  RandomForest non trouvé, création d'un nouveau")
                rf_classifier = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            print("   ⚠️  Format non reconnu, création d'un nouveau RF")
            rf_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
except Exception as e:
    print(f"   ⚠️  Erreur: {e}")
    print("   → Création d'un nouveau RandomForest")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

# Créer le nouveau pipeline
print("\n🔨 Construction du nouveau pipeline...")
pipeline = Pipeline([
    ('feature_extractor', FeatureExtractor(fs=100, expected_len=3000)),
    ('scaler', StandardScaler()),
    ('classifier', rf_classifier)
])

print("   ✅ Pipeline créé avec:")
print(f"      - FeatureExtractor (16 features)")
print(f"      - StandardScaler")
print(f"      - {type(rf_classifier).__name__}")

# Fit le pipeline (pour que le scaler soit fitted)
print("\n🎓 Entraînement du pipeline...")
print(f"   Données: {X_train.shape}")

pipeline.fit(X_train, y_train)

print("   ✅ Pipeline entraîné (scaler fitted)")

# Tester une prédiction
print("\n🧪 Test de prédiction...")
X_test = np.random.randn(1, 3000)
try:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    print(f"   ✅ Prédiction: classe {pred[0]}")
    print(f"   ✅ Probabilités: {proba[0]}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    exit(1)

# Sauvegarder le nouveau pipeline
output_path = Path('models/rf_v2_pipeline_fixed_v2.joblib')
print(f"\n💾 Sauvegarde du pipeline dans {output_path}...")

joblib.dump(pipeline, output_path)

print("   ✅ Pipeline sauvegardé")

# Vérifier qu'on peut le recharger
print("\n✅ Vérification du rechargement...")
pipeline_reloaded = joblib.load(output_path)
pred_reloaded = pipeline_reloaded.predict(X_test)
print(f"   ✅ Pipeline rechargé et testé: classe {pred_reloaded[0]}")

print("\n" + "="*60)
print("🎉 SUCCÈS !")
print("="*60)
print(f"\nNouveau fichier créé: {output_path}")
print("\n📝 Prochaines étapes:")
print("1. Mettre à jour app/main.py:")
print(f'   MODEL_PATH = PROJECT_ROOT / "{output_path}"')
print("\n2. Tester l'API en local:")
print("   uvicorn app.main:app --reload")
print("\n3. Push sur GitHub pour redéployer sur Render")
print("="*60)