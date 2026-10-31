"""Scoring cross-validated CNN runs: from out-of-fold recording probabilities to patient decisions.

Every number here comes from patients the network never trained on. Anything fitted on top of the CNN
(a recalibration, multipliers) is refit for each held-out fold on the other four folds' patients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from murmur.core.constants import CLASS_WEIGHTS, CLASSES
from murmur.core.metric import confusion, weighted_accuracy
from murmur.models import patient


def patient_probs(run_dir: Path, patient_of: pd.Series) -> pd.DataFrame:
    """A run's out-of-fold recording probabilities, aggregated to patients (models.patient.aggregate)."""
    oof = pd.read_csv(run_dir / "oof-recordings.tsv", sep="\t", index_col="recording_id")
    return patient.aggregate(oof[list(CLASSES)], patient_of)


def fused(probs: pd.DataFrame, extra: pd.DataFrame | None, y: pd.Series, folds: pd.Series) -> pd.DataFrame:
    """Logistic regression on log CNN probabilities (+ extra patient features), refit per held-out fold on
    the other folds, sample weights = the metric's class weights. Returns out-of-fold probabilities."""
    X = np.log(probs[list(CLASSES)].clip(1e-6))
    if extra is not None:
        X = X.join(extra.loc[probs.index])
    out = pd.DataFrame(index=probs.index, columns=list(CLASSES), dtype=float)
    for k in folds.unique():
        tr, te = folds != k, folds == k
        scaler = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=5000)
        lr.fit(scaler.transform(X[tr]), y[tr], sample_weight=y[tr].map(CLASS_WEIGHTS))
        out.loc[te] = pd.DataFrame(lr.predict_proba(scaler.transform(X[te])), index=X.index[te],
                                   columns=lr.classes_)[list(CLASSES)].to_numpy()
    return out


def nested_decide(probs: pd.DataFrame, y: pd.Series, folds: pd.Series) -> pd.Series:
    """Present/Unknown multipliers tuned on the other folds' patients, applied to each held-out fold."""
    pred = pd.Series(index=probs.index, dtype=object)
    for k in folds.unique():
        tr, te = folds != k, folds == k
        pred[te] = patient.decide(probs[te], patient.tune_multipliers(probs[tr], y[tr]))
    return pred


def score(name: str, y: pd.Series, pred: pd.Series, folds: pd.Series) -> dict:
    """WA overall and per fold, and the patient-level confusion matrix (rows true, columns predicted)."""
    y = y.loc[pred.index]
    per_fold = [weighted_accuracy(y[folds == k], pred[folds == k]) for k in sorted(folds.unique())]
    m = confusion(y, pred)
    return {"decision": name, "WA": weighted_accuracy(y, pred), "fold mean": float(np.mean(per_fold)),
            "fold sd": float(np.std(per_fold)), "per fold": np.round(per_fold, 3).tolist(),
            **{f"{c} caught": int(m[k, k]) for k, c in enumerate(CLASSES)}, "confusion": m}


def patient_extra(lane_table: pd.DataFrame, patient_of: pd.Series, patients: pd.DataFrame, index) -> pd.DataFrame:
    """Patient means of the tagger heart-likeness, the not-beat-locked baseline and the voice share, and
    whether the patient is an infant or neonate. Missing patients get the median."""
    cols = ["tagger_heart_mean", "nbl_median", "voice_share"]
    per = lane_table[cols].groupby(patient_of.loc[lane_table.index].to_numpy()).mean().reindex(index)
    per = per.fillna(per.median())
    per["infant"] = patients.loc[index, "age"].isin(["Infant", "Neonate"]).astype(float)
    return per


def seeds(run_dirs: list[Path], patient_of: pd.Series) -> list[pd.DataFrame]:
    """Patient probabilities of several runs of the same model (one per training seed), on common patients."""
    probs = [patient_probs(d, patient_of) for d in run_dirs]
    return [p.loc[probs[0].index] for p in probs]


def ensemble(probs: list[pd.DataFrame]) -> pd.DataFrame:
    """The seed ensemble: patient probabilities averaged over runs."""
    return sum(probs) / len(probs)


def fold_wa(probs: pd.DataFrame, y: pd.Series, folds: pd.Series) -> np.ndarray:
    """Argmax-decision WA in each held-out fold."""
    pred = patient.decide(probs)
    return np.array([weighted_accuracy(y[pred.index][folds == k], pred[folds == k]) for k in sorted(folds.unique())])
