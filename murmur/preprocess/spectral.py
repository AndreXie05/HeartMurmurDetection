"""Frequency-domain views of a recording: filtering, spectra, spectrograms, window quality."""
from __future__ import annotations

import librosa
import numpy as np
from scipy import signal as sps

from murmur.core.constants import SAMPLE_RATE


def bandpass(x: np.ndarray, low: float = 25.0, high: float = 800.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass: removes baseline drift below `low` and hiss above `high`."""
    sos = sps.butter(order, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def psd(x: np.ndarray, n_fft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of the whole signal: (frequencies, power)."""
    return sps.welch(x, fs=SAMPLE_RATE, nperseg=n_fft)


def interval_psd(x: np.ndarray, spans: list[tuple[float, float]], n_fft: int = 1024) -> np.ndarray | None:
    """Mean power spectrum over short intervals (e.g. every systole), each Hann-windowed and
    zero-padded to `n_fft`, so intervals of different lengths share one frequency axis. None if no
    interval is long enough (>= 32 samples)."""
    spectra = []
    for start, end in spans:
        piece = x[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if len(piece) < 32:
            continue
        piece = piece[:n_fft] * np.hanning(min(len(piece), n_fft))
        spectra.append(np.abs(np.fft.rfft(piece, n_fft)) ** 2 / len(piece))
    return np.mean(spectra, axis=0) if spectra else None


def psd_frequencies(n_fft: int = 1024) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)


def log_spectrogram(x: np.ndarray, n_fft: int = 256, hop: int = 64, fmax: float = 800.0):
    """Linear-frequency log-power spectrogram up to `fmax`: (frequencies, times, dB)."""
    f, t, s = sps.stft(x, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop)
    keep = f <= fmax
    return f[keep], t, 10 * np.log10(np.abs(s[keep]) ** 2 + 1e-10)


def mel_spectrogram(x: np.ndarray, n_mels: int = 64, n_fft: int = 256, hop: int = 64,
                    fmax: float = 800.0) -> np.ndarray:
    """Log-mel spectrogram (dB), mels x frames."""
    m = librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=25.0, fmax=fmax)
    return librosa.power_to_db(m, ref=np.max)


def window_quality(x: np.ndarray, window_s: float = 2.0, band: tuple[float, float] = (25.0, 400.0),
                   n_fft: int = 512) -> np.ndarray:
    """Per non-overlapping window: [fraction of power outside `band`, spectral flatness inside it].

    Candidate segmentation-free quality scores — speech and rubbing put energy above the heart band,
    and a flat spectrum is noise-like. Whether they track the annotators' unmarked time is measured
    in the exploration notebook, not assumed.
    """
    size = int(window_s * SAMPLE_RATE)
    rows = []
    for start in range(0, len(x) - size + 1, size):
        f, p = sps.welch(x[start:start + size], fs=SAMPLE_RATE, nperseg=n_fft)
        inside = (f >= band[0]) & (f <= band[1])
        flat = np.exp(np.mean(np.log(p[inside] + 1e-20))) / (np.mean(p[inside]) + 1e-20)
        rows.append([1 - p[inside].sum() / (p.sum() + 1e-20), flat])
    return np.array(rows).reshape(-1, 2)


def band_shares(freqs: np.ndarray, power: np.ndarray,
                bands: list[tuple[float, float]]) -> np.ndarray:
    """Median over recordings (rows of `power`) of the share of power in each [low, high) band."""
    total = power.sum(axis=1)
    return np.array([np.nanmedian(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / total)
                     for lo, hi in bands])


def state_contrast(x: np.ndarray, systoles: list[tuple[float, float]], diastoles: list[tuple[float, float]],
                   start: float, end: float, fmin: float, fmax: float, min_beats: int = 2) -> float | None:
    """How much louder (dB) a time-frequency box is during systoles than during diastoles.

    A sound locked to the heartbeat — a systolic murmur — is much louder in systole (Present systoles
    sit ~13 dB above their diastoles at 50–300 Hz, Absent ~2 dB: notebook 01, section 4); noise that
    keeps its own time is not. Needs the segmentation, and returns None when the box holds fewer than
    `min_beats` systoles or diastoles.
    """
    freqs, times, db = log_spectrogram(x, n_fft=256, hop=40, fmax=SAMPLE_RATE / 2)
    band = (freqs >= fmin) & (freqs <= max(fmax, fmin + 1))
    power = 10 ** (db[band] / 10).mean(axis=0) if band.any() else None
    if power is None:
        return None

    def mean_in(spans):
        inside = [(s, e) for s, e in spans if s >= start and e <= end]
        if len(inside) < min_beats:
            return None
        mask = np.zeros_like(times, dtype=bool)
        for s, e in inside:
            mask |= (times >= s) & (times < e)
        return power[mask].mean() if mask.any() else None

    sys_power, dia_power = mean_in(systoles), mean_in(diastoles)
    if sys_power is None or dia_power is None:
        return None
    return float(10 * np.log10(sys_power / dia_power))
