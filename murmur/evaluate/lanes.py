"""Do the label-free noise lanes (murmur.preprocess.noise) separate Unknown — and leave murmurs alone?

Each lane is summarised per recording (baseline = median, events = share of frames above a level);
patients take the mean over their recordings.
"""
from __future__ import annotations

from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from murmur.data import circor, segments
from murmur.preprocess import noise

SUMMARIES = ["nbl_median", "nbl_p90", "voice_share", "voice_mean", "above_median", "above_p90",
             "above_left_median", "above_left_p90", "sudden_share"]
@lru_cache(maxsize=1)
def _recordings() -> pd.DataFrame:
    """The recordings table, read once per worker process."""
    return circor.recordings()


def _nan_stat(curve: np.ndarray, fn) -> float:
    return float(fn(curve)) if np.isfinite(curve).any() else np.nan


def summarise(rid: str) -> dict:
    """The lane summaries of one recording."""
    rec = _recordings().loc[rid]
    x, seg = circor.audio(rec), circor.segmentation(rec)
    s1 = [s for s, _ in segments.intervals(seg, 1)]
    nbl = noise.not_beat_locked_sync(x, s1)
    voice = noise.voice(x)
    above = noise.above_heart_band(x)
    left = noise.above_heart_band_leftover(x, s1)
    sudden = noise.sudden_off_beat(x, segments.intervals(seg, 1) + segments.intervals(seg, 3))
    return {"recording_id": rid,
            "nbl_median": _nan_stat(nbl, np.nanmedian), "nbl_p90": _nan_stat(nbl, lambda c: np.nanpercentile(c, 90)),
            "voice_share": float((voice > 0.3).mean()), "voice_mean": float(voice.mean()),
            "above_median": float(np.median(above)), "above_p90": float(np.percentile(above, 90)),
            "above_left_median": _nan_stat(left, np.nanmedian),
            "above_left_p90": _nan_stat(left, lambda c: np.nanpercentile(c, 90)),
            "sudden_share": float((sudden > 0.5).mean())}


def table(recording_ids, workers: int = 4) -> pd.DataFrame:
    """`summarise` for many recordings, in `workers` processes (~0.5 s per recording each). Run with
    OMP_NUM_THREADS=1 so the processes do not oversubscribe the CPU."""
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(summarise, list(recording_ids), chunksize=8))
    return pd.DataFrame(rows).set_index("recording_id").loc[list(recording_ids)]


def per_patient(recording_table: pd.DataFrame, patient_of: pd.Series) -> pd.DataFrame:
    return recording_table.groupby(patient_of.loc[recording_table.index].to_numpy()).mean()


def auc_table(per: pd.DataFrame, murmur: pd.Series, infant: pd.Series) -> pd.DataFrame:
    """Per feature: AUC for Unknown (all, infants, older) and Present vs Absent (a noise lane should be near
    0.5 there: higher means murmurs leak into it). Below 0.5 = lower values go with the class."""
    rows = []
    for f in per.columns:
        v = per[f].fillna(per[f].median())
        pa = murmur.isin(["Present", "Absent"])
        rows.append({"feature": f, "AUC Unknown": roc_auc_score(murmur == "Unknown", v),
                     "infants": roc_auc_score(murmur[infant] == "Unknown", v[infant]),
                     "older": roc_auc_score(murmur[~infant] == "Unknown", v[~infant]),
                     "AUC Present vs Absent": roc_auc_score(murmur[pa] == "Present", v[pa])})
    return pd.DataFrame(rows).set_index("feature")


def oof_auc_unknown(per: pd.DataFrame, murmur: pd.Series, folds: pd.Series, columns: list[str]) -> float:
    """Out-of-fold AUC for Unknown of a logistic regression on `columns`, refit per held-out fold."""
    X = per[columns].fillna(per[columns].median()).to_numpy()
    y = (murmur == "Unknown").to_numpy()
    oof = np.zeros(len(y))
    for k in folds.unique():
        tr, te = (folds != k).to_numpy(), (folds == k).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


GRADES = {"I/VI": 1, "II/VI": 2, "III/VI": 3}
PITCHES = {"Low": 1, "Medium": 2, "High": 3}


def leak_check(recording_table: pd.DataFrame, patients: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Do the lanes rise with how loud (grade) and how high (pitch) a murmur is? Present patients only,
    their recordings at a murmur location, Spearman ρ per lane. A noise lane should show ~0: a positive ρ
    means murmurs are read as noise, which would push Present towards Unknown."""
    from scipy.stats import spearmanr
    t = recording_table.copy()
    t["patient_id"] = recordings.loc[t.index, "patient_id"].to_numpy()
    present = patients[patients["murmur"] == "Present"]
    t = t[t["patient_id"].isin(present.index)]
    at = [recordings.loc[r, "location"] in present.loc[p, "murmur_locations"] for r, p in zip(t.index, t["patient_id"])]
    per = t[at].groupby("patient_id").mean(numeric_only=True)
    grade = present.loc[per.index, "systolic_grading"].map(GRADES)
    pitch = present.loc[per.index, "systolic_pitch"].map(PITCHES)
    rows = [{"lane": c, "ρ with grade": spearmanr(grade, per[c], nan_policy="omit")[0],
             "ρ with pitch": spearmanr(pitch, per[c], nan_policy="omit")[0]} for c in per.columns]
    return pd.DataFrame(rows).set_index("lane")


def heart_rate_check(recording_table: pd.DataFrame, patients: pd.DataFrame, recordings: pd.DataFrame,
                     lanes=("nbl_median", "above_left_median", "above_median")) -> tuple[pd.DataFrame, pd.Series]:
    """Spearman ρ of heart rate with each lane within age group × class (so age does not fake it), and the
    AUC for Unknown of heart rate alone (all, infants, children)."""
    from scipy.stats import spearmanr
    t = recording_table.copy()
    t["bpm"] = [segments.heart_rate(circor.segmentation(recordings.loc[r])) for r in t.index]
    t["patient_id"] = recordings.loc[t.index, "patient_id"].to_numpy()
    t["murmur"] = patients.loc[t["patient_id"], "murmur"].to_numpy()
    t["age"] = patients.loc[t["patient_id"], "age"].to_numpy()
    rho = pd.DataFrame({f"{age}/{cls}": {lane: spearmanr(t.loc[(t.age == age) & (t.murmur == cls), "bpm"],
                                                         t.loc[(t.age == age) & (t.murmur == cls), lane],
                                                         nan_policy="omit")[0] for lane in lanes}
                        for age in ("Infant", "Child") for cls in ("Absent", "Present")})
    per = t.groupby("patient_id")["bpm"].median()
    per = per.fillna(per.median())
    m, age = patients.loc[per.index, "murmur"], patients.loc[per.index, "age"]
    auc = pd.Series({group: roc_auc_score(m[mask] == "Unknown", per[mask]) for group, mask in
                     [("all", age.notna() | age.isna()), ("infants", age == "Infant"), ("children", age == "Child")]},
                    name="AUC Unknown from heart rate")
    return rho, auc
