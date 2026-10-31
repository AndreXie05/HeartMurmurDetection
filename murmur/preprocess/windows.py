"""The CNN's input: a log-power spectrogram of the heart band, cut into fixed-length windows.

25–800 Hz holds the heart sounds and murmurs (the data exploration). 100 frames per second; each
recording is standardised on its own (mean 0, sd 1 over all its bins), so the stethoscope's gain and
the chest's damping do not set the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from murmur.core.paths import outputs_dir
from murmur.data import circor, segments
from murmur.preprocess import noise, spectral

FPS = 100
FMIN, FMAX = 25.0, 800.0
WINDOW_S, HOP_S = 3.0, 1.5
WINDOW, HOP = int(WINDOW_S * FPS), int(HOP_S * FPS)


def spectrogram(x: np.ndarray) -> np.ndarray:
    """(frequency bins, frames) float32, standardised per recording."""
    f, _, db = spectral.log_spectrogram(x, n_fft=256, hop=4000 // FPS, fmax=FMAX)
    db = db[f >= FMIN]
    return ((db - db.mean()) / (db.std() + 1e-6)).astype(np.float32)


def all_spectrograms(recordings: pd.DataFrame) -> dict[str, np.ndarray]:
    """Spectrograms of every recording, cached as float16 in outputs/ (recomputed if the cache is
    missing a recording)."""
    path = outputs_dir() / "spectrograms-25-800.npz"
    cached = dict(np.load(path)) if path.is_file() else {}
    missing = [r for r in recordings.index if r not in cached]
    for rid in missing:
        cached[rid] = spectrogram(circor.audio(recordings.loc[rid])).astype(np.float16)
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cached)
    return {rid: cached[rid] for rid in recordings.index}


def repet_channels(x: np.ndarray, s1_starts: list[float]) -> np.ndarray:
    """(3, bins, frames) float32: the spectrogram, its beat-locked repeating part (heart and murmur) and
    the leftover (noise), on the same 25–800 Hz grid (noise.beat_sync_split, upsampled from 50 to 100
    frames/s). All three are in dB, standardised with the full spectrogram's mean and sd so the network
    sees them on one scale. With under 3 heart cycles the repeating part is the whole spectrogram and
    the leftover is empty (the floor)."""
    full = spectrogram(x)
    split = noise.beat_sync_split(x, s1_starts)
    if split is None:
        return np.stack([full, full, np.full_like(full, full.min())])
    freqs, s, repeating, _ = split
    keep = freqs <= FMAX
    db = lambda m: 20 * np.log10(np.repeat(m[keep], 2, axis=1)[:, :full.shape[1]] + 1e-6)
    whole, rep, left = db(s), db(repeating), db(np.maximum(s - repeating, 0))
    # floor at the recording's 1st-percentile level: where nothing is left over the leftover would sit
    # at -120 dB (the epsilon), a value with no meaning that would dominate the channel's scale
    floor = np.percentile(whole, 1)
    rep, left = np.maximum(rep, floor), np.maximum(left, floor)
    mean, sd = whole.mean(), whole.std() + 1e-6
    pad = lambda a: np.pad(a, ((0, 0), (0, full.shape[1] - a.shape[1])), mode="edge")
    return np.stack([full, pad((rep - mean) / sd), pad((left - mean) / sd)]).astype(np.float32)


def all_repet_channels(recordings: pd.DataFrame) -> dict[str, np.ndarray]:
    """`repet_channels` of every recording (beat grid from the experts' S1 onsets), cached as float16."""
    path = outputs_dir() / "channels-repet-25-800.npz"
    cached = dict(np.load(path)) if path.is_file() else {}
    missing = [r for r in recordings.index if r not in cached]
    for rid in missing:
        rec = recordings.loc[rid]
        s1 = [s for s, _ in segments.intervals(circor.segmentation(rec), 1)]
        cached[rid] = repet_channels(circor.audio(rec), s1).astype(np.float16)
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cached)
    return {rid: cached[rid] for rid in recordings.index}


def starts(n_frames: int, offset: int = 0) -> list[int]:
    """Window start frames every HOP, beginning at `offset`; a recording shorter than a window gets one
    window at 0 (padded by `cut`)."""
    if n_frames <= WINDOW:
        return [0]
    return list(range(offset % HOP, n_frames - WINDOW + 1, HOP))


def cut(spec: np.ndarray, start: int) -> np.ndarray:
    """One window ([channels,] bins, WINDOW frames), zero-padded at the end if the recording is short."""
    w = spec[..., start:start + WINDOW]
    if w.shape[-1] < WINDOW:
        w = np.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, WINDOW - w.shape[-1])])
    return w
