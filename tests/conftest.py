import sys
from pathlib import Path

# Répertoire racine du projet (pour imports app.*)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
