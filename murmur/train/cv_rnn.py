"""Cross-validation of the GRU (models/rnn.py), with exactly the CNN's protocol (train/cv.py).

Same frozen folds, same 15 % inner validation for early stopping, same seeds, same loss (weighted
5/3/1), same training loop and patient aggregation; only the network changes. The test split is never
touched. Writes the same files as the CNN runs (oof-recordings.tsv, history-fold{k}.tsv,
gru-fold{k}.pt, summary.json), so murmur.evaluate.decisions scores both the same way.

From the command line (a GPU helps):  python -m murmur.train.cv_rnn --inputs repet --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from murmur.core.paths import results_dir
from murmur.models.rnn import SmallGRU
from murmur.train.cv import INPUTS, SEED, training_set
from murmur.train.loop import predict_recordings, train_fold


def run(out: Path, inputs: str = "repet", repeat: int = 0, device: str | None = None,
        max_epochs: int = 40, patience: int = 8) -> pd.DataFrame:
    """Train and predict all folds; seeds are SEED + 100 × repeat + fold (as train/cv.py)."""
    train, _, recs, patient_of, rec_label = training_set()
    build, channels = INPUTS[inputs]
    specs = build(recs)
    n_bins = next(iter(specs.values())).shape[-2]
    make = lambda: SmallGRU(n_bins, in_channels=channels)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = SEED + 100 * repeat
    out.mkdir(parents=True, exist_ok=True)
    ids = lambda ps: list(recs.index[recs["patient_id"].isin(ps)])
    oof, summary = [], []
    for fold in sorted(train["fold"].unique()):
        rest = train[train["fold"] != fold]
        fit_p, val_p = train_test_split(rest.index, test_size=0.15, stratify=rest["stratum"], random_state=seed + fold)
        result = train_fold(make(), specs, rec_label, ids(fit_p), ids(val_p), patient_of, seed + fold, device,
                            max_epochs=max_epochs, patience=patience)
        model = make().to(device)
        model.load_state_dict(result["state"])
        oof.append(predict_recordings(model, specs, ids(train.index[train["fold"] == fold]), device).assign(fold=fold))
        h = result["history"]
        h.assign(fold=fold).to_csv(out / f"history-fold{fold}.tsv", sep="\t", index=False)
        torch.save(result["state"], out / f"gru-fold{fold}.pt")
        summary.append({"fold": int(fold), "best_epoch": int(result["best_epoch"]), "epochs_run": int(len(h)),
                        "inner_val_wa": float(h["val_wa"].max()), "seconds": float(h["seconds"].sum())})
        print(summary[-1], flush=True)
    pd.concat(oof).to_csv(out / "oof-recordings.tsv", sep="\t", index_label="recording_id")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return pd.DataFrame(summary)


def run_dir(version: str, seed: int) -> Path:
    """results/rnn/<version>/seed<k>, next to results/cnn/."""
    return results_dir() / "rnn" / version / f"seed{seed}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", choices=list(INPUTS), default="repet")
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    version = {"spectrogram": "plain", "repet": "3-channel"}[args.inputs]
    for s in args.seed:
        print(f"== {version}, seed {s}", flush=True)
        run(run_dir(version, s), args.inputs, repeat=s, device=args.device)
