"""Where the dataset and outputs live.

Looked up at call time, never bound at import, so a notebook or test can point elsewhere by setting
the environment variable before calling.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """The CirCor DigiScope folder: `training_data.csv` and `training_data/` (README, "Dataset")."""
    return Path(os.environ.get("MURMUR_DATA", ROOT / "data"))


def outputs_dir() -> Path:
    """Trained models, predictions and figures — not committed."""
    return Path(os.environ.get("MURMUR_OUTPUTS", ROOT / "outputs"))


def results_dir() -> Path:
    """Stored results the notebooks read — small tables and the trained CNN weights, committed so a
    notebook shows its numbers without retraining. A notebook run with RECOMPUTE = True rewrites them."""
    return Path(os.environ.get("MURMUR_RESULTS", ROOT / "results"))
