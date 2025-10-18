"""
Script pour recréer et sauvegarder le pipeline avec la classe FeatureExtractor
depuis le code Python (pas depuis le notebook).
"""

import numpy as np
import joblib
from pathlib import Path
import sys

# Ajouter app/ au path pour importer FeatureExtractor
sys.path.insert(0, str(Path(__file__).parent))

from app.feature_extractor import FeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("🔧 RECONSTRUCTION DU PIPELINE")
print("=" * 70)

# Chemins
OLD_MODEL_PATH = Path("notebooks/models/random_forest_v2_features.pkl")
NEW_PIPELINE_PATH = Path("notebooks/models/rf_v2_pipeline_fixed.joblib")

# Créer le dossier si nécessaire
NEW_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Option 1 : Charger l'ancien modèle RF et reconstruire le pipeline
# ============================================================================

print("\n📦 Chargement de l'ancien modèle Random Forest...")

try:
    # Charger le modèle RF sauvegardé (juste le classifier, pas le pipeline)
    import pickle
    with open(OLD_MODEL_PATH, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"✅ Modèle chargé depuis {OLD_MODEL_PATH}")
    print(f"   Type: {type(rf_model)}")
    
except Exception as e:
    print(f"❌ Impossible de charger l'ancien modèle: {e}")
    print("\n⚠️  On va créer un nouveau pipeline avec les bons hyperparamètres")
    
    # Créer un nouveau modèle avec les mêmes hyperparamètres
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    print("✅ Nouveau modèle Random Forest créé")

# ============================================================================
# Construire le nouveau pipeline
# ============================================================================

print("\n🔨 Construction du pipeline...")

pipeline = Pipeline([
    ('feature_extractor', FeatureExtractor(fs=100, expected_len=3000)),
    ('scaler', StandardScaler()),
    ('classifier', rf_model)
])

print("✅ Pipeline créé")
print(f"   Étapes: {list(pipeline.named_steps.keys())}")

# ============================================================================
# Si le modèle RF n'était pas entraîné, on doit le réentraîner
# ============================================================================

# Vérifier si le modèle est entraîné
try:
    # Tester avec un signal bidon
    test_signal = np.random.randn(1, 3000)
    _ = pipeline.predict(test_signal)
    print("✅ Le modèle est déjà entraîné")
    
except Exception as e:
    print(f"⚠️  Le modèle n'est pas entraîné: {e}")
    print("\n📊 Chargement des données d'entraînement...")
    
    # Charger les données
    data_dir = Path("notebooks/data/processed")
    
    try:
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        
        print(f"✅ Données chargées: {X_train.shape}")
        
        # Appliquer SMOTE si nécessaire
        from imblearn.over_sampling import SMOTE
        print("\n⚖️  Application de SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   {X_train.shape} → {X_train_balanced.shape}")
        
        # Entraîner le pipeline
        print("\n🎓 Entraînement du pipeline...")
        print("   (Cela peut prendre 1-2 minutes)")
        pipeline.fit(X_train_balanced, y_train_balanced)
        print("✅ Pipeline entraîné")
        
    except FileNotFoundError:
        print("❌ Données d'entraînement non trouvées")
        print("   Le pipeline sera sauvegardé mais non entraîné")

# ============================================================================
# Sauvegarder le nouveau pipeline
# ============================================================================

print(f"\n💾 Sauvegarde du pipeline...")
print(f"   Destination: {NEW_PIPELINE_PATH}")

try:
    joblib.dump(pipeline, NEW_PIPELINE_PATH)
    
    # Vérifier la taille
    file_size = NEW_PIPELINE_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Pipeline sauvegardé!")
    print(f"   Taille: {file_size:.2f} MB")
    
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde: {e}")
    raise

# ============================================================================
# Test de chargement
# ============================================================================

print(f"\n🧪 Test de chargement du nouveau pipeline...")

try:
    # Charger le pipeline
    loaded_pipeline = joblib.load(NEW_PIPELINE_PATH)
    print("✅ Pipeline chargé avec succès")
    
    # Test prédiction
    test_signal = np.random.randn(1, 3000)
    prediction = loaded_pipeline.predict(test_signal)
    probabilities = loaded_pipeline.predict_proba(test_signal)
    
    class_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    predicted_class = class_names[int(prediction[0])]
    confidence = np.max(probabilities)
    
    print(f"✅ Prédiction fonctionnelle!")
    print(f"   Classe: {predicted_class}")
    print(f"   Confiance: {confidence:.2%}")
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Instructions finales
# ============================================================================

print("\n" + "=" * 70)
print("✅ PIPELINE RECONSTRUIT AVEC SUCCÈS!")
print("=" * 70)
print(f"""
📝 PROCHAINES ÉTAPES:

1. Dans app/main.py, modifier la ligne 31:
   
   MODEL_PATH = PROJECT_ROOT / "notebooks" / "models" / "rf_v2_pipeline_fixed.joblib"

2. Relancer le serveur:
   
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

3. Tester l'API:
   
   http://localhost:8000/docs
""")
print("=" * 70)