"""The challenge metric: weighted accuracy over patients (Present 5, Unknown 3, Absent 1)."""
from __future__ import annotations

import numpy as np

from murmur.core.constants import CLASS_WEIGHTS, CLASSES


def confusion(y_true, y_pred) -> np.ndarray:
    """Patient-level confusion matrix, rows = true class, columns = predicted, in CLASSES order."""
    index = {c: k for k, c in enumerate(CLASSES)}
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[index[t], index[p]] += 1
    return m


def weighted_accuracy(y_true, y_pred) -> float:
    """Sum of weight × correct over sum of weight × patients, per true class (PhysioNet 2022)."""
    m = confusion(y_true, y_pred)
    w = np.array([CLASS_WEIGHTS[c] for c in CLASSES])
    return float((w * np.diag(m)).sum() / (w * m.sum(axis=1)).sum())
