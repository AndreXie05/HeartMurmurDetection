"""A pretrained AudioSet model (PANNs Cnn14_DecisionLevelMax) applied to our 4 kHz recordings.

Two uses: its sound-event probabilities over time (the "tagger": speech, crying, heart sounds…) and its
internal features (fc1, 2048 per frame) as a frozen representation. The model expects 32 kHz audio up to
14 kHz; ours is 4 kHz, so everything above 2 kHz is missing. `speed` plays the recording faster before
the model hears it: at 1 it is upsampled to 32 kHz, at 4 it is upsampled to 8 kHz and declared 32 kHz,
at 8 the raw samples are declared 32 kHz — moving the heart band (25–800 Hz) from 13 of the model's 64
mel bins to 38 or 48.

Needs `panns_inference` (in requirements.txt); the checkpoint (~300 MB) is downloaded on first use.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly

from murmur.core.constants import SAMPLE_RATE

RATE = 32000
CHUNK = 20 * RATE        # the recording goes through in 20 s pieces, to fit a 6 GB GPU
FPS = 100                # the model's frame rate (hop 320 at 32 kHz)
# AudioSet classes kept per frame by the tagger: the ones that match noise we expect, plus the heart
KEEP = ["Speech", "Child speech, kid speaking", "Conversation", "Babbling", "Baby cry, infant cry",
        "Crying, sobbing", "Whimper", "Screaming", "Laughter", "Baby laughter", "Breathing", "Cough",
        "Sniff", "Rustle", "Scratch", "Rub", "Tap", "Knock", "Thump, thud", "Clicking", "Music",
        "Television", "Hum", "Mains hum", "Heart sounds, heartbeat", "Heart murmur", "Silence"]
# the kept classes folded into a few curves (max over each group, per frame)
CURVES = {
    "heart": ["Heart sounds, heartbeat"],
    "speech": ["Speech", "Child speech, kid speaking", "Conversation", "Babbling", "Television"],
    "crying": ["Baby cry, infant cry", "Crying, sobbing", "Whimper", "Screaming"],
    "breathing": ["Breathing", "Cough", "Sniff"],
    "handling": ["Rustle", "Scratch", "Rub", "Tap", "Knock", "Thump, thud", "Clicking"],
}


def load(device: str = "cuda", random_weights: bool = False):
    """(model, AudioSet class names). `random_weights` is the control: the same architecture untrained,
    keeping only the pretrained input normalisation (bn0, per-mel-bin statistics), so its first layer sees
    input on the same scale."""
    from panns_inference import SoundEventDetection
    from panns_inference.config import labels
    from panns_inference.models import Cnn14_DecisionLevelMax
    pretrained = SoundEventDetection(checkpoint_path=None, device=device).model
    pretrained = getattr(pretrained, "module", pretrained)          # unwrap DataParallel
    if not random_weights:
        return pretrained.eval(), labels
    torch.manual_seed(0)
    model = Cnn14_DecisionLevelMax(sample_rate=RATE, window_size=1024, hop_size=320, mel_bins=64,
                                   fmin=50, fmax=14000, classes_num=527)
    model.bn0.load_state_dict(pretrained.bn0.state_dict())
    return model.to(device).eval(), labels


def prepare(x: np.ndarray, speed: int = 1, target_rms: float | None = None) -> np.ndarray:
    """The 4 kHz signal as the model will hear it at 32 kHz, optionally RMS-normalised first."""
    if target_rms is not None:
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * target_rms
    up = RATE // (SAMPLE_RATE * speed)
    return (resample_poly(x, up, 1) if up > 1 else x).astype(np.float32)


def _pieces(x32: np.ndarray):
    for start in range(0, len(x32), CHUNK):
        piece = x32[start:start + CHUNK]
        yield start, np.pad(piece, (0, max(0, RATE // 2 - len(piece))))   # under 0.5 s: padded


@torch.no_grad()
def frame_probs(model, x32: np.ndarray) -> np.ndarray:
    """(frames, 527) probabilities at FPS frames per second of the prepared signal."""
    device = next(model.parameters()).device
    out = []
    for start, piece in _pieces(x32):
        probs = model(torch.from_numpy(piece[None]).to(device), None)["framewise_output"][0].cpu().numpy()
        out.append(probs[: int(np.ceil(min(CHUNK, len(x32) - start) / RATE * FPS))])
    return np.concatenate(out)


@torch.no_grad()
def embedding(model, x32: np.ndarray) -> np.ndarray:
    """Mean- and max-pooled fc1 activations (2 × 2048) over the prepared signal."""
    device = next(model.parameters()).device
    grabbed = []
    hook = model.fc1.register_forward_hook(lambda _m, _i, out: grabbed.append(torch.relu(out)[0].cpu()))
    for _, piece in _pieces(x32):
        model(torch.from_numpy(piece[None]).to(device), None)
    hook.remove()
    frames = torch.cat(grabbed).numpy()
    return np.concatenate([frames.mean(0), frames.max(0)])


def kept_curves(probs: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """The CURVES groups (max over each group's classes) from full 527-class frame probabilities."""
    return {name: probs[:, [labels.index(c) for c in group]].max(axis=1) for name, group in CURVES.items()}


def kept_to_curves(kept: np.ndarray) -> dict[str, np.ndarray]:
    """The CURVES groups from frame probabilities restricted to the KEEP classes (columns in KEEP order)."""
    return {name: kept[:, [KEEP.index(c) for c in group]].max(axis=1) for name, group in CURVES.items()}


def tag_all(recordings, audio, device: str = "cuda"):
    """Run the tagger over `recordings` (rows of circor.recordings(); `audio` = circor.audio).
    Returns (summary: max probability per kept class and recording, plus the mean of the heart class;
    kept-class frames per recording as float16). ~4 min for all 3163 recordings on a 6 GB GPU."""
    import pandas as pd
    model, labels = load(device)
    keep = [labels.index(c) for c in KEEP]
    rows, frames = [], {}
    for rid, rec in recordings.iterrows():
        probs = frame_probs(model, prepare(audio(rec)))[:, keep]
        frames[rid] = probs.astype(np.float16)
        rows.append({"recording_id": rid, **dict(zip(KEEP, probs.max(axis=0).astype(float))),
                     "heart_mean": float(probs[:, KEEP.index("Heart sounds, heartbeat")].mean())})
    return pd.DataFrame(rows).set_index("recording_id"), frames


def embed_all(recordings, audio, speed: int, random_weights: bool = False, device: str = "cuda") -> np.ndarray:
    """(recordings, 4096) frozen features, RMS-normalised input at `speed` (see the module docstring)."""
    model, _ = load(device, random_weights)
    return np.stack([embedding(model, prepare(audio(rec), speed, target_rms=0.05))
                     for _, rec in recordings.iterrows()]).astype(np.float32)
