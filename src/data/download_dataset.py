"""
Script de téléchargement du dataset Sleep-EDF (VERSION CORRIGÉE)
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Télécharge un fichier avec barre de progression"""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def get_available_subjects():
    """
    Liste des sujets réellement disponibles sur PhysioNet
    Format: (subject_id, a_psg_name, a_hypnogram_name)
    """
    # Cette liste contient les vrais noms de fichiers disponibles
    subjects = [
        ("SC4001E0", "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
        ("SC4002E0", "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
        ("SC4011E0", "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
        ("SC4012E0", "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
        ("SC4021E0", "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
        ("SC4022E0", "SC4022E0-PSG.edf", "SC4022EJ-Hypnogram.edf"),
        ("SC4031E0", "SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf"),
        ("SC4032E0", "SC4032E0-PSG.edf", "SC4032EH-Hypnogram.edf"),
        ("SC4041E0", "SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf"),
        ("SC4042E0", "SC4042E0-PSG.edf", "SC4042EC-Hypnogram.edf"),
        ("SC4051E0", "SC4051E0-PSG.edf", "SC4051EC-Hypnogram.edf"),
        ("SC4052E0", "SC4052E0-PSG.edf", "SC4052EC-Hypnogram.edf"),
        ("SC4061E0", "SC4061E0-PSG.edf", "SC4061EC-Hypnogram.edf"),
        ("SC4062E0", "SC4062E0-PSG.edf", "SC4062EC-Hypnogram.edf"),
        ("SC4071E0", "SC4071E0-PSG.edf", "SC4071EC-Hypnogram.edf"),
        ("SC4072E0", "SC4072E0-PSG.edf", "SC4072EH-Hypnogram.edf"),
        ("SC4081E0", "SC4081E0-PSG.edf", "SC4081EC-Hypnogram.edf"),
        ("SC4082E0", "SC4082E0-PSG.edf", "SC4082EP-Hypnogram.edf"),
        ("SC4091E0", "SC4091E0-PSG.edf", "SC4091EC-Hypnogram.edf"),
        ("SC4092E0", "SC4092E0-PSG.edf", "SC4092EC-Hypnogram.edf"),
        ("SC4101E0", "SC4101E0-PSG.edf", "SC4101EC-Hypnogram.edf"),
        ("SC4102E0", "SC4102E0-PSG.edf", "SC4102EC-Hypnogram.edf"),
        ("SC4111E0", "SC4111E0-PSG.edf", "SC4111EC-Hypnogram.edf"),
        ("SC4112E0", "SC4112E0-PSG.edf", "SC4112EC-Hypnogram.edf"),
        ("SC4121E0", "SC4121E0-PSG.edf", "SC4121EC-Hypnogram.edf"),
        ("SC4122E0", "SC4122E0-PSG.edf", "SC4122EH-Hypnogram.edf"),
        ("SC4131E0", "SC4131E0-PSG.edf", "SC4131EC-Hypnogram.edf"),
        ("SC4141E0", "SC4141E0-PSG.edf", "SC4141EU-Hypnogram.edf"),
        ("SC4142E0", "SC4142E0-PSG.edf", "SC4142EU-Hypnogram.edf"),
        ("SC4151E0", "SC4151E0-PSG.edf", "SC4151EC-Hypnogram.edf"),
    ]
    return subjects

def main():
    # Créer le dossier
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URL de base
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    
    # Liste des sujets disponibles
    subjects = get_available_subjects()
    
    print(f"🌙 Téléchargement de {len(subjects)} sujets Sleep-EDF")
    print(f"📁 Destination: {output_dir}")
    print("-" * 60)
    
    downloaded = 0
    failed = 0
    
    for subject_id, psg_name, hypno_name in subjects:
        print(f"\n📦 Sujet {subject_id}")
        
        # Télécharger PSG
        psg_path = output_dir / psg_name
        if psg_path.exists():
            print(f"  ⏭️  PSG déjà téléchargé")
        else:
            print(f"  ⬇️  PSG...")
            url = base_url + psg_name
            if download_file(url, str(psg_path)):
                downloaded += 1
            else:
                failed += 1
                continue
        
        # Petit délai pour ne pas surcharger le serveur
        time.sleep(0.5)
        
        # Télécharger Hypnogramme
        hypno_path = output_dir / hypno_name
        if hypno_path.exists():
            print(f"  ⏭️  Hypnogramme déjà téléchargé")
        else:
            print(f"  ⬇️  Hypnogramme...")
            url = base_url + hypno_name
            if download_file(url, str(hypno_path)):
                downloaded += 1
            else:
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Téléchargement terminé!")
    print(f"📊 Nouveaux fichiers: {downloaded}")
    print(f"❌ Échecs: {failed}")
    
    # Vérifier ce qu'on a
    psg_files = list(output_dir.glob("*-PSG.edf"))
    hypno_files = list(output_dir.glob("*-Hypnogram.edf"))
    
    print(f"\n📈 Total dans data/raw:")
    print(f"   PSG: {len(psg_files)}")
    print(f"   Hypnogrammes: {len(hypno_files)}")
    
    if len(psg_files) >= 20 and len(hypno_files) >= 20:
        print(f"\n🎉 Excellent! Tu as assez de données pour entraîner le modèle!")
    elif len(psg_files) >= 10 and len(hypno_files) >= 10:
        print(f"\n✅ Bien! Suffisant pour un MVP")
    else:
        print(f"\n⚠️ Peu de données - le modèle risque de ne pas être très bon")

if __name__ == "__main__":
    main()