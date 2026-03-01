"""File I/O utilities."""
from pathlib import Path
import pandas as pd

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_csv(path):
    return pd.read_csv(path)
