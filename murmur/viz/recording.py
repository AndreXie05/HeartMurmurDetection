"""One recording drawn as a spectrogram with its segmentation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from murmur.data import circor
from murmur.preprocess import spectral


def spectrogram(recording: pd.Series, title: str = "", state: int = 2, ax=None):
    """Band-passed log spectrogram (25–800 Hz), with every interval in `state` (default: systole)
    ticked white along the top."""
    x = spectral.bandpass(circor.audio(recording))
    freqs, times, power = spectral.log_spectrogram(x)
    if ax is None:
        ax = plt.subplots(figsize=(12, 2.4))[1]
    ax.pcolormesh(times, freqs, power, shading="auto", vmin=np.percentile(power, 5), vmax=power.max())
    seg = circor.segmentation(recording)
    for _, row in seg[seg["state"] == state].iterrows():
        ax.axvspan(row["start"], row["end"], ymin=0.97, ymax=1, color="white")
    ax.set(title=title or recording.name, ylabel="Hz", xlabel="s")
    return ax
