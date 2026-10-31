"""Cross-validation of a window CNN over the frozen folds of the training patients.

In each fold: train on the other four folds minus a stratified 15 % inner validation (which does the
early stopping), then predict the held-out fold. The test split is never touched. Writes, per run:
oof-recordings.tsv (out-of-fold recording probabilities), history-fold{k}.tsv, cnn-fold{k}.pt (weights)
and summary.json (best epoch, epochs run, time per fold).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from murmur.data import circor, split
from murmur.models.cnn import SmallCNN
from murmur.preprocess import windows
from murmur.train.loop import predict_recordings, train_fold

SEED = 2026
INPUTS = {"spectrogram": (windows.all_spectrograms, 1),         # input name -> (builder, channels)
          "repet": (windows.all_repet_channels, 3)}             # spectrogram + repeating part + leftover


def training_set():
    """(train split rows, patients, training recordings, recording -> patient, recording -> label).
    Every recording carries its patient's Murmur label."""
    s, patients, recs = split.load(), circor.patients(), circor.recordings()
    train = s[s.split == "train"]
    recs = recs[recs["patient_id"].isin(train.index)]
    patient_of = recs["patient_id"]
    rec_label = patients.loc[patient_of, "murmur"].set_axis(recs.index)
    return train, patients, recs, patient_of, rec_label


def run(out: Path, inputs: str = "spectrogram", repeat: int = 0, device: str | None = None) -> pd.DataFrame:
    """Train and predict all folds; seeds are SEED + 100 × repeat + fold. Returns the per-fold summary."""
    train, _, recs, patient_of, rec_label = training_set()
    build, channels = INPUTS[inputs]
    specs = build(recs)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = SEED + 100 * repeat
    out.mkdir(parents=True, exist_ok=True)
    ids = lambda ps: list(recs.index[recs["patient_id"].isin(ps)])
    oof, summary = [], []
    for fold in sorted(train["fold"].unique()):
        rest = train[train["fold"] != fold]
        fit_p, val_p = train_test_split(rest.index, test_size=0.15, stratify=rest["stratum"], random_state=seed + fold)
        result = train_fold(SmallCNN(in_channels=channels), specs, rec_label, ids(fit_p), ids(val_p), patient_of,
                            seed + fold, device)
        model = SmallCNN(in_channels=channels).to(device)
        model.load_state_dict(result["state"])
        oof.append(predict_recordings(model, specs, ids(train.index[train["fold"] == fold]), device).assign(fold=fold))
        h = result["history"]
        h.assign(fold=fold).to_csv(out / f"history-fold{fold}.tsv", sep="\t", index=False)
        torch.save(result["state"], out / f"cnn-fold{fold}.pt")
        summary.append({"fold": int(fold), "best_epoch": int(result["best_epoch"]), "epochs_run": int(len(h)),
                        "inner_val_wa": float(h["val_wa"].max()), "seconds": float(h["seconds"].sum())})
        print(summary[-1], flush=True)
    pd.concat(oof).to_csv(out / "oof-recordings.tsv", sep="\t", index_label="recording_id")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return pd.DataFrame(summary)
