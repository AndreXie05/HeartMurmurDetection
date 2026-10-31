"""A linear probe on frozen per-recording features: how much do they know about the murmur?

Patient vector = mean of its recordings' vectors; StandardScaler → PCA 128 (unsupervised, fitted inside
each fold; 4096 raw dims on ~600 patients made the inner search take over 30 min) → multinomial logistic
regression with C chosen by inner 5-fold log-loss. Out-of-fold over the frozen folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from murmur.core.constants import CLASS_WEIGHTS, CLASSES
from murmur.core.metric import confusion, weighted_accuracy


def oof_probs(X_rec: np.ndarray, rec_patient: np.ndarray, patient_ids, folds: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Out-of-fold class probabilities per patient (rows = patient_ids, columns = CLASSES)."""
    X = np.stack([X_rec[rec_patient == p].mean(0) for p in patient_ids])
    target = y.loc[patient_ids].map(CLASSES.index).to_numpy()
    fold = folds.loc[patient_ids].to_numpy()
    proba = np.zeros((len(patient_ids), len(CLASSES)))
    for k in np.unique(fold):
        probe = make_pipeline(StandardScaler(), PCA(min(128, X.shape[1]), random_state=0), LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 6), cv=5, scoring="neg_log_loss", max_iter=2000))
        probe.fit(X[fold != k], target[fold != k])
        proba[fold == k] = probe.predict_proba(X[fold == k])
    return pd.DataFrame(proba, index=list(patient_ids), columns=list(CLASSES))


def summary(name: str, probs: pd.DataFrame, y: pd.Series) -> tuple[dict, np.ndarray]:
    """WA with the decision argmax(weight × probability) — which maximises expected weighted accuracy —
    the AUCs for Present and Unknown, and the confusion matrix."""
    w = np.array([CLASS_WEIGHTS[c] for c in CLASSES])
    pred = pd.Series(np.array(CLASSES)[(probs.to_numpy() * w).argmax(axis=1)], index=probs.index)
    yy = y.loc[probs.index]
    row = {"features": name, "WA": weighted_accuracy(yy, pred),
           "AUC Present": roc_auc_score(yy == "Present", probs["Present"]),
           "AUC Unknown": roc_auc_score(yy == "Unknown", probs["Unknown"])}
    return row, confusion(yy, pred)
