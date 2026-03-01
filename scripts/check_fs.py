import joblib
import sys

fs = joblib.load('models/herg/feature_selection.pkl')
print('Type:', type(fs))
if isinstance(fs, dict):
    print('Keys:', fs.keys())
    for k, v in fs.items():
        if hasattr(v, '__len__'):
            print(f'{k}: len={len(v)}, type={type(v)}')
        else:
            print(f'{k}: {v}')
elif hasattr(fs, '__len__'):
    print('Length:', len(fs))
