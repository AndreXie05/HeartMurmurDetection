"""One recording with what each detector heard: spectrogram, heart cycle, the tagger's and our lanes."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from murmur.data import circor, segments
from murmur.preprocess import noise, spectral

OURS = {"voice": "tab:orange", "not beat-locked": "deeppink"}
TAGGER = {"heart": "tab:red", "speech": "goldenrod"}
SEG_COLORS = {1: "0.5", 2: "tab:orange", 3: "0.5", 4: "tab:blue"}


def lanes(recording: pd.Series, tagger: dict[str, np.ndarray] | None = None, title: str = ""):
    """Top: 0–2000 Hz spectrogram. Then the experts' heart cycle (grey S1/S2, orange systole, blue
    diastole), the tagger's heart and speech curves (if given, 100 per second), and our voice and
    not-beat-locked lanes (computed here from the audio)."""
    x, seg = circor.audio(recording), circor.segmentation(recording)
    s1 = [s for s, _ in segments.intervals(seg, 1)]
    ours = {"voice": noise.voice(x), "not beat-locked": noise.not_beat_locked_sync(x, s1)}
    f, t, db = spectral.log_spectrogram(x, n_fft=256, hop=40, fmax=2000.0)
    rows = 3 + (tagger is not None)
    fig, ax = plt.subplots(rows, 1, figsize=(12, 1.6 + 1.1 * rows), sharex=True,
                           gridspec_kw={"height_ratios": [3, 0.3] + [1] * (rows - 2)})
    ax[0].pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=np.percentile(db, 2), vmax=np.percentile(db, 99.7))
    ax[0].set(ylabel="Hz", title=title or recording.name)
    for state, color in SEG_COLORS.items():
        for a, b in segments.intervals(seg, state):
            ax[1].axvspan(a, b, color=color)
    ax[1].set(yticks=[], ylabel="cycle")
    k = 2
    if tagger is not None:
        for name, color in TAGGER.items():
            ax[k].plot(np.arange(len(tagger[name])) / 100, tagger[name], color=color, label=f"tagger: {name}")
        ax[k].set(ylim=(0, 1), ylabel="tagger")
        ax[k].legend(loc="upper right", fontsize=8)
        k += 1
    for name, color in OURS.items():
        ax[k].plot(np.arange(len(ours[name])) / noise.FPS, ours[name], color=color, label=f"ours: {name}")
    ax[k].set(ylim=(0, 1), ylabel="ours", xlabel="s")
    ax[k].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
