"""MNIST readout and retrieval protection experiments using the Experiment 5 baseline."""
from pathlib import Path
import sys

# Earlier experiments live in script directories rather than installed packages.
ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src", ROOT / "experiment-id 5"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
