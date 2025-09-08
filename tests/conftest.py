import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path (…/PeopleCounter)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
