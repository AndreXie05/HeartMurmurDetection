"""From recording probabilities to one decision per patient.

A murmur is heard only at some auscultation positions (19 % of Present patients' recordings are away
from it), so Present takes the patient's strongest recording; Unknown and Absent are averaged.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from murmur.core.constants import CLASSES
from murmur.core.metric import weighted_accuracy


def aggregate(recording_probs: pd.DataFrame, patient_of: pd.Series) -> pd.DataFrame:
    """recording_probs: one row per recording, columns CLASSES. Returns one row per patient:
    [max P(Present), mean P(Unknown), mean P(Absent)], renormalised to sum to 1."""
    grouped = recording_probs.groupby(patient_of.loc[recording_probs.index].to_numpy())
    out = pd.DataFrame({"Present": grouped["Present"].max(), "Unknown": grouped["Unknown"].mean(),
                        "Absent": grouped["Absent"].mean()})[list(CLASSES)]
    return out.div(out.sum(axis=1), axis=0)


def decide(patient_probs: pd.DataFrame, multipliers=(1.0, 1.0, 1.0)) -> pd.Series:
    """The class with the highest probability × multiplier (multipliers in CLASSES order)."""
    scores = patient_probs[list(CLASSES)].to_numpy() * np.asarray(multipliers)
    return pd.Series(np.array(CLASSES)[scores.argmax(axis=1)], index=patient_probs.index)


def tune_multipliers(patient_probs: pd.DataFrame, y_true: pd.Series,
                     grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)) -> tuple[float, float, float]:
    """Multipliers for Present and Unknown (Absent fixed at 1) that maximise weighted accuracy on the
    given patients. Tune on validation predictions only, never on the patients being scored."""
    best, best_wa = (1.0, 1.0, 1.0), -1.0
    for mp, mu in itertools.product(grid, grid):
        wa = weighted_accuracy(y_true.loc[patient_probs.index], decide(patient_probs, (mp, mu, 1.0)))
        if wa > best_wa:
            best, best_wa = (mp, mu, 1.0), wa
    return best
