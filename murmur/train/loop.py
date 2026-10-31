"""Training a window classifier with early stopping on patient-level weighted accuracy.

Every window of a recording carries its patient's label (the spec allows only the Murmur label).
The loss weighs classes by the metric's weights (Present 5, Unknown 3, Absent 1). Each epoch cuts the
windows at a new random offset, so the network does not see the same 3 s twice.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from murmur.core.constants import CLASS_WEIGHTS, CLASSES
from murmur.core.metric import weighted_accuracy
from murmur.models import patient
from murmur.preprocess import windows

BATCH = 64


def _batches(specs: dict, items: list[tuple[str, int]], labels: dict | None, device, shuffle_rng=None):
    order = np.arange(len(items)) if shuffle_rng is None else shuffle_rng.permutation(len(items))
    for k in range(0, len(order), BATCH):
        chunk = [items[j] for j in order[k:k + BATCH]]
        x = torch.from_numpy(np.stack([windows.cut(specs[r], s) for r, s in chunk]).astype(np.float32))
        y = None if labels is None else torch.tensor([CLASSES.index(labels[r]) for r, _ in chunk])
        x = x[:, None] if x.ndim == 3 else x          # one channel: add the axis; several: already there
        yield chunk, x.to(device), (None if y is None else y.to(device))


@torch.no_grad()
def predict_recordings(model: nn.Module, specs: dict, ids, device) -> pd.DataFrame:
    """Mean softmax over each recording's windows (fixed offset 0): one row per recording."""
    model.eval()
    items = [(r, s) for r in ids for s in windows.starts(specs[r].shape[-1])]
    sums = {r: np.zeros(len(CLASSES)) for r in ids}
    counts = {r: 0 for r in ids}
    for chunk, x, _ in _batches(specs, items, None, device):
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        for (r, _), p in zip(chunk, probs):
            sums[r] += p
            counts[r] += 1
    return pd.DataFrame({r: sums[r] / counts[r] for r in ids}, index=list(CLASSES)).T


def train_fold(model: nn.Module, specs: dict, rec_label: pd.Series, train_ids, val_ids, patient_of: pd.Series,
               seed: int, device, max_epochs: int = 40, patience: int = 8, lr: float = 1e-3) -> dict:
    """Train on `train_ids` recordings, early-stop on patient WA of `val_ids` recordings.
    Returns {"state": best weights, "best_epoch", "history": per-epoch rows}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([float(CLASS_WEIGHTS[c]) for c in CLASSES], device=device))
    labels = rec_label.to_dict()
    val = rec_label.loc[list(val_ids)]
    y_val = val.groupby(patient_of.loc[val.index].to_numpy()).first()
    best = {"wa": -1.0, "state": None, "best_epoch": 0}
    history = []
    for epoch in range(1, max_epochs + 1):
        started, model_loss, n = time.time(), 0.0, 0
        model.train()
        items = [(r, s) for r in train_ids for s in windows.starts(specs[r].shape[-1], int(rng.integers(windows.HOP)))]
        for _, x, y in _batches(specs, items, labels, device, shuffle_rng=rng):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            model_loss, n = model_loss + loss.item() * len(y), n + len(y)
        probs = patient.aggregate(predict_recordings(model, specs, val_ids, device), patient_of)
        wa = weighted_accuracy(y_val.loc[probs.index], patient.decide(probs))
        history.append({"epoch": epoch, "train_loss": model_loss / n, "val_wa": wa, "seconds": time.time() - started})
        if wa > best["wa"]:
            best = {"wa": wa, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "best_epoch": epoch}
        elif epoch - best["best_epoch"] >= patience:
            break
    return {"state": best["state"], "best_epoch": best["best_epoch"], "history": pd.DataFrame(history)}
