import wfdb
import numpy as np
from pathlib import Path

DATA_DIR   = Path('data/raw_apnea')
DEMO_DIR   = Path('data/demo')
DEMO_DIR.mkdir(exist_ok=True)

# Signal apnée — sujet a01, minute 5 (apnée certaine)
rec = wfdb.rdrecord(str(DATA_DIR / 'a01'))
ann = wfdb.rdann(str(DATA_DIR / 'a01'), 'apn')

# Trouver une minute apnée
for i, sym in enumerate(ann.symbol):
    if sym == 'A':
        seg = rec.p_signal[i*6000:(i+1)*6000, 0]
        np.save(DEMO_DIR / 'demo_ecg_apnea.npy', seg)
        print(f'ECG apnée extrait — minute {i}')
        break

# Trouver une minute normale (sujet c01)
rec2 = wfdb.rdrecord(str(DATA_DIR / 'c01'))
ann2 = wfdb.rdann(str(DATA_DIR / 'c01'), 'apn')
for i, sym in enumerate(ann2.symbol):
    if sym == 'N':
        seg = rec2.p_signal[i*6000:(i+1)*6000, 0]
        np.save(DEMO_DIR / 'demo_ecg_normal.npy', seg)
        print(f'ECG normal extrait — minute {i}')
        break

# EEG — extraire une époque N3 et une Wake depuis Sleep-EDF
import mne
DATA_EEG = Path('data/raw')
psg = sorted(DATA_EEG.glob('*PSG.edf'))[0]
hyp = sorted(DATA_EEG.glob('*Hypnogram.edf'))[0]

STAGE_MAP = {'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,
             'Sleep stage 3':3,'Sleep stage 4':3,'Sleep stage R':4}

raw   = mne.io.read_raw_edf(psg, include=['EEG Fpz-Cz'], preload=True, verbose=False)
data, _ = raw['EEG Fpz-Cz']
data  = data[0]
annot = mne.read_annotations(hyp)

saved = {s: False for s in [0, 2, 3, 4]}
for a in annot:
    if a['description'] not in STAGE_MAP:
        continue
    stage = STAGE_MAP[a['description']]
    if stage in saved and not saved[stage]:
        start = int(a['onset'] * 100)
        epoch = data[start:start+3000]
        if len(epoch) == 3000:
            name = {0:'wake', 2:'n2', 3:'n3', 4:'rem'}[stage]
            np.save(DEMO_DIR / f'demo_eeg_{name}.npy', epoch)
            print(f'EEG {name} extrait')
            saved[stage] = True
    if all(saved.values()):
        break

print('\\nFichiers créés dans data/demo/ :')
for f in sorted(DEMO_DIR.glob('*.npy')):
    print(f'  {f.name}')