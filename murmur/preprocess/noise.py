"""Label-free noise curves over time, from the audio alone (so they also exist for the hidden set).

Each returns one value per frame at FPS frames per second, in 0–1. They are built from what separates
noise from the heart: a voice is pitched, breathing sits above the heart band, movement is abrupt at
moments that are not heart sounds, and nothing but the heart repeats with every beat.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d

from murmur.core.constants import SAMPLE_RATE
from murmur.preprocess import spectral

FPS = 100
HOP = SAMPLE_RATE // FPS


def _fit(curve: np.ndarray, n: int) -> np.ndarray:
    """Pad or cut to n frames, so every curve lines up with the 100-per-second spectrogram."""
    return np.pad(curve, (0, max(0, n - len(curve))))[:n]


def frames(x: np.ndarray) -> int:
    return int(np.ceil(len(x) / HOP))


def pitch_track(x: np.ndarray, fmin: float = 220.0, fmax: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """pYIN pitch track (f0 Hz, voiced bool) per frame, after a high-pass at `fmin`. Without the high-pass
    pYIN locked onto the heart's low rumble: on the 15 listening recordings it called 20–64 % of frames
    voiced in every one, nearly all at its 150 Hz floor, and its per-frame probability stayed near 0
    even on a clear infant cry (84900_MV, 2026-10-30). The cost: adult voices below 220 Hz."""
    sos = sps.butter(6, fmin, btype="highpass", fs=SAMPLE_RATE, output="sos")
    f0, voiced, _ = librosa.pyin(sps.sosfiltfilt(sos, x), fmin=fmin, fmax=fmax, sr=SAMPLE_RATE,
                                 frame_length=512, hop_length=HOP)
    return f0, voiced


def pitched(x: np.ndarray) -> np.ndarray:
    """Whether a pitched sound is present (pYIN's voiced decision, smoothed over 0.1 s): voices — a baby
    crying or babbling, an adult talking — have a fundamental; heart sounds, breathing, movement do not."""
    _, voiced = pitch_track(x)
    return _fit(np.convolve(voiced.astype(float), np.ones(10) / 10, mode="same"), frames(x))


def voice(x: np.ndarray, faint_db: float = 3.0, strong_db: float = 20.0,
          track: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Pitched sound weighted by how strongly its harmonics stand out: the mean, over f0 and its first two
    harmonics, of the peak's dB above the median of the ±100 Hz around it. `faint_db` reads 0 and
    `strong_db` 1. On/off alone put a faint background conversation at the top of the lane like a loud
    cry (Tiago, 2026-10-30); prominence separates them: median 4.1 dB (85182_MV, faint talk) vs
    11.2 dB (84900_MV, crying)."""
    f0, voiced = track if track is not None else pitch_track(x)   # pass it to compute pYIN once
    f, _, db = spectral.log_spectrogram(x, n_fft=512, hop=HOP, fmax=2000.0)
    n = min(db.shape[1], len(voiced))
    strength = np.zeros(n)
    for j in np.flatnonzero(voiced[:n]):
        peaks = []
        for h in (1, 2, 3):
            if h * f0[j] >= f[-1] - 50:
                break
            b = int(np.argmin(np.abs(f - h * f0[j])))
            peaks.append(db[max(0, b - 1):b + 2, j].max() - np.median(db[max(0, b - 12):b + 13, j]))
        strength[j] = np.clip((np.mean(peaks) - faint_db) / (strong_db - faint_db), 0, 1)
    return _fit(np.convolve(strength, np.ones(10) / 10, mode="same"), frames(x))


def above_heart_band(x: np.ndarray, low: float = 400.0, high: float = 2000.0, span_db: float = 30.0) -> np.ndarray:
    """Energy between `low` and `high` Hz, in dB above the recording's own quiet floor (10th percentile),
    scaled so `span_db` above the floor reads 1. Heart sounds sit mostly below ~200 Hz; breathing,
    voices and hiss lift this band."""
    f, _, db = spectral.log_spectrogram(x, n_fft=256, hop=HOP, fmax=high)
    band = 10 * np.log10(np.mean(10 ** (db[f >= low] / 10), axis=0) + 1e-12)
    return _fit(np.clip((band - np.percentile(band, 10)) / span_db, 0, 1), frames(x))


def sudden_off_beat(x: np.ndarray, heart_sounds: list[tuple[float, float]] | None = None,
                    guard_s: float = 0.05) -> np.ndarray:
    """How abruptly the sound changes (positive spectral flux, 25–2000 Hz), with the moments of S1 and S2
    (± `guard_s`) removed when the segmentation is known: abrupt at the wrong moment = movement, bumps,
    knocks. In units of the recording's median flux, scaled so 20× the median reads 1."""
    f, _, db = spectral.log_spectrogram(x, n_fft=256, hop=HOP, fmax=2000.0)
    flux = np.maximum(np.diff(db[f >= 25], axis=1), 0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    curve = np.clip(flux / (20 * np.median(flux) + 1e-12), 0, 1)
    for start, end in heart_sounds or []:
        curve[max(0, int((start - guard_s) * FPS)):int((end + guard_s) * FPS) + 1] = 0
    return _fit(curve, frames(x))


def not_beat_locked(x: np.ndarray, repeat_after_s: float = 0.3) -> np.ndarray:
    """Share of each frame's energy that does NOT repeat elsewhere in the recording (REPET-SIM: each
    frame is compared with its most similar frames at least `repeat_after_s` away, i.e. other beats;
    what they share is the repeating part). Heart sounds and murmurs repeat with every beat; voices,
    crying, breathing out of step and movement do not. Computed at 50 frames/s, then repeated to 100."""
    hop = 2 * HOP
    s = np.abs(librosa.stft(x.astype(np.float32), n_fft=256, hop_length=hop))
    s = s[librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=256) >= 25]
    width = max(1, int(repeat_after_s * SAMPLE_RATE / hop))
    repeating = np.minimum(s, librosa.decompose.nn_filter(s, aggregate=np.median, metric="cosine", width=width))
    energy = (s ** 2).sum(axis=0) + 1e-12
    share = ((s - repeating) ** 2).sum(axis=0) / energy
    return _fit(np.repeat(np.clip(share, 0, 1), 2), frames(x))


def _track_beats(low: np.ndarray, onsets: list[float], hop: int, duration: float,
                 search: float = 0.25, fixed: bool = False) -> list[float]:
    """Continue the experts' S1 onsets over the unsegmented rest of the recording, following the real
    beats. From each end of the segmented stretch, predict the next onset one median beat away, then
    move it within ±`search` of a beat to where the heart-band envelope best matches the mean envelope
    of the experts' beats. A fixed beat length drifted out of step over 20 s (49824_AV: 7 segmented
    beats in 25 s) and every heart sound leaked into the leftover (2026-10-31).
    `low` is the 25–200 Hz magnitude spectrogram at `hop`."""
    fps = SAMPLE_RATE / hop
    env = (low ** 2).sum(axis=0)
    env = np.log(env + 1e-12)
    period = float(np.median(np.diff(onsets)))
    n = max(2, int(round(period * fps)))
    starts = [int(round(o * fps)) for o in onsets]
    shapes = [env[k:k + n] for k in starts if k + n <= len(env)]
    if not shapes:
        return onsets
    template = np.mean(shapes, axis=0)
    template = (template - template.mean()) / (template.std() + 1e-12)
    reach = max(1, int(round(search * n)))

    def best(k: int) -> int:
        scores = []
        for shift in range(-reach, reach + 1):
            j = k + shift
            if j < 0 or j + n > len(env):
                scores.append(-np.inf)
                continue
            w = env[j:j + n]
            scores.append(float(((w - w.mean()) / (w.std() + 1e-12) * template).mean()))
        if fixed:
            return k
        return k + int(np.argmax(scores)) - reach if np.isfinite(max(scores)) else k

    after, k = [], starts[-1]
    while k + n + n // 2 <= len(env):
        k = best(k + n)
        after.append(k)
    before, k = [], starts[0]
    while k - n >= 0:
        k = best(k - n)
        before.append(k)
    out = [b / fps for b in reversed(before)] + onsets + [a / fps for a in after]
    return [min(max(0.0, o), duration) for o in out]


def beat_sync_split(x: np.ndarray, s1_starts: list[float], steps: int = 40, neighbours: int = 9,
                    slack: int = 1, track: bool = True, gain_bands: tuple = ()):
    """Beat-synchronous REPET: (frequencies, magnitude spectrogram, its repeating part, covered frames)
    at 50 frames/s.

    Each cycle (from one S1 onset to the next, from the experts' segmentation) is time-warped to `steps`
    points, so a faster or slower beat still lines up point for point. The repeating part of a cycle is
    the median of its `neighbours` nearest cycles at the same points, warped back. Breathing repeats too,
    but at its own period, so at a fixed point of the heart cycle it does not line up and stays in the
    leftover; a REPET that matched any similar frame let steady breathing pass as heart (Tiago,
    2026-10-30). Beyond the segmented stretch the cycles continue at the median beat length. Returns
    None when there are fewer than 3 cycles."""
    hop = 2 * HOP
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=256)
    s = np.abs(librosa.stft(x.astype(np.float32), n_fft=256, hop_length=hop))[freqs >= 25]
    freqs = freqs[freqs >= 25]
    onsets = sorted(set(s1_starts))           # duplicated onsets made a zero beat length (2026-10-31)
    if len(onsets) >= 2 and np.median(np.diff(onsets)) > 0:
        onsets = _track_beats(s[freqs <= 200], onsets, hop, len(x) / SAMPLE_RATE, fixed=not track)
    bounds = sorted({int(round(t * SAMPLE_RATE / hop)) for t in onsets})
    cycles = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 4 and b <= s.shape[1]]
    if len(cycles) < 3:
        return None
    grid = np.linspace(0, 1, steps)
    warped = np.stack([np.stack([np.interp(grid, np.linspace(0, 1, b - a), row) for row in s[:, a:b]])
                       for a, b in cycles])                     # cycles x freqs x steps
    repeating, covered = np.zeros_like(s), np.zeros(s.shape[1], bool)
    for k, (a, b) in enumerate(cycles):
        near = np.argsort(np.abs(np.arange(len(cycles)) - k))[1:neighbours + 1]
        template = np.median(warped[near], axis=0)
        back = np.stack([np.interp(np.linspace(0, 1, b - a), grid, row) for row in template])
        # ±`slack` frames of timing tolerance: a sharp S1/S2 arriving 20 ms off its neighbours' timing
        # otherwise left a residue on every heart sound (49824_AV: S1 still 0.25 in the leftover band)
        back = maximum_filter1d(back, size=2 * slack + 1, axis=1)
        # optional per-beat loudness (least-squares gain per band, 0.5–2×). Tried 2026-10-31 for a strong
        # heart that swells with breathing (49824_AV): on 4 listening recordings it raised the leftover's
        # median without removing its beat-by-beat pulsing, in every variant. Off by default; tracking the
        # beats fixed the pulsing instead (one-beat autocorrelation 0.27 -> 0.04).
        for lo, hi in gain_bands:
            band = (freqs >= lo) & (freqs < hi)
            gain = (s[band, a:b] * back[band]).sum() / ((back[band] ** 2).sum() + 1e-12)
            back[band] *= np.clip(gain, 0.5, 2.0)
        repeating[:, a:b] = np.minimum(s[:, a:b], back)
        covered[a:b] = True
    return freqs, s, repeating, covered


def not_beat_locked_sync(x: np.ndarray, s1_starts: list[float], **split_options) -> np.ndarray:
    """Share of each frame's energy that does not repeat at the same point of the heart cycle
    (`beat_sync_split`). NaN where unknown (under 3 cycles), not 0."""
    split = beat_sync_split(x, s1_starts, **split_options)
    if split is None:
        return _fit(np.full(2 * (frames(x) // 2 + 1), np.nan), frames(x))
    _, s, repeating, covered = split
    share = ((s - repeating) ** 2).sum(axis=0) / ((s ** 2).sum(axis=0) + 1e-12)
    share[~covered] = np.nan
    return _fit(np.repeat(np.clip(share, 0, 1), 2), frames(x))


def above_heart_band_leftover(x: np.ndarray, s1_starts: list[float], low: float = 400.0,
                              span_db: float = 30.0, **split_options) -> np.ndarray:
    """Energy above `low` Hz that does NOT repeat with the beat, in dB above the recording's quiet floor
    (10th percentile of the whole band), scaled so `span_db` reads 1. `above_heart_band` alone fired on
    loud S1/S2 and on murmurs (Present vs Absent AUC 0.640; Tiago heard it on a strong-heart murmur
    recording, 49824_AV, 2026-10-31): the beat-locked part is removed first, so breathing and hiss
    remain."""
    split = beat_sync_split(x, s1_starts, **split_options)
    if split is None:
        return _fit(np.full(2 * (frames(x) // 2 + 1), np.nan), frames(x))
    freqs, s, repeating, covered = split
    band = freqs >= low
    total = 10 * np.log10((s[band] ** 2).mean(axis=0) + 1e-12)
    leftover = 10 * np.log10(((s[band] - repeating[band]) ** 2).mean(axis=0) + 1e-12)
    curve = np.clip((leftover - np.percentile(total, 10)) / span_db, 0, 1)
    curve[~covered] = np.nan
    return _fit(np.repeat(curve, 2), frames(x))
