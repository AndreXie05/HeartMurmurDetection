"""Per-recording measurements over the whole dataset, for exploring it before modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from murmur.data import circor, segments
from murmur.preprocess import spectral


def segmentation_table(recordings: pd.DataFrame) -> pd.DataFrame:
    """segments.summary for every recording."""
    return pd.DataFrame([segments.summary(circor.segmentation(r)) for _, r in recordings.iterrows()],
                        index=recordings.index)


def spectra(recordings: pd.DataFrame, n_fft: int = 1024) -> dict[str, np.ndarray]:
    """Per recording: the Welch PSD of the whole raw signal, and the mean spectrum of its systoles
    and of its diastoles (NaN rows where a recording has none). Keys: freqs, whole, systole, diastole."""
    whole, systole, diastole = [], [], []
    empty = np.full(n_fft // 2 + 1, np.nan)
    for _, rec in recordings.iterrows():
        x, seg = circor.audio(rec), circor.segmentation(rec)
        whole.append(spectral.psd(x, n_fft)[1])
        for store, state in ((systole, 2), (diastole, 4)):
            mean = spectral.interval_psd(x, segments.intervals(seg, state), n_fft)
            store.append(empty if mean is None else mean)
    return {"freqs": spectral.psd_frequencies(n_fft), "whole": np.array(whole),
            "systole": np.array(systole), "diastole": np.array(diastole)}


def quality_table(recordings: pd.DataFrame, window_s: float = 2.0) -> pd.DataFrame:
    """Per 2 s window: recording id, window start, out-of-band power fraction, in-band flatness, and
    the fraction of the window the annotators left unmarked (the reference the scores are tested on)."""
    rows = []
    for rid, rec in recordings.iterrows():
        x, seg = circor.audio(rec), circor.segmentation(rec)
        for i, (outside, flat) in enumerate(spectral.window_quality(x, window_s)):
            start, end = i * window_s, (i + 1) * window_s
            overlap = (seg["end"].clip(upper=end) - seg["start"].clip(lower=start)).clip(lower=0)
            rows.append({"recording_id": rid, "start": start, "out_of_band": outside, "flatness": flat,
                         "unmarked": overlap[seg["state"] == 0].sum() / window_s})
    return pd.DataFrame(rows)
