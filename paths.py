import sys
from pathlib import Path

def get_root_path() -> str:
    return Path(__file__).resolve().parent

ROOT_PATH = get_root_path()

DATA_DIR = ROOT_PATH / "data"
NOTEBOOKS_DIR = ROOT_PATH / "notebooks"
SRC_DIR = ROOT_PATH / "src"

RAW_DATA = DATA_DIR / "raw"

def set_paths():
    for folder in (ROOT_PATH, SRC_DIR, DATA_DIR):
        folder = str(folder)
        if folder not in sys.path:
            sys.path.insert(0, folder)

set_paths()