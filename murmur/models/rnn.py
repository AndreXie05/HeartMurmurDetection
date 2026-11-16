"""A small bidirectional GRU over spectrogram windows: (batch, channels, frequency bins, frames) -> 3 class logits.

Same input as the CNN (models/cnn.py), read differently: the CNN looks at a window as an image, the GRU
reads it as a sequence of 300 time steps (100 per second), one spectrogram column per step. Each step is
the column's frequency bins of every channel, flattened into one vector.

    window (C, F, T) -> T steps of C·F numbers -> linear projection -> 2-layer bidirectional GRU
    -> mean and max over time -> dropout -> linear -> 3 logits

- Bidirectional: at each moment the network knows what came before and after (is this sound between
  S1 and S2, i.e. in systole, where murmurs are?).
- Mean over time: the window's overall character (noise, for Unknown). Max over time: its strongest
  moment (a murmur heard in only some beats).
- Gradient clipping: recurrent networks can have exploding gradients, so every gradient element is
  clamped to ±CLIP during backpropagation (done with hooks, so the shared training loop is reused as is).
"""
from __future__ import annotations

import torch
from torch import nn

from murmur.core.constants import CLASSES

CLIP = 1.0


class SmallGRU(nn.Module):
    """About 135k weights for the 3-channel input, close to the CNN's ~100k: ~750 training patients."""

    def __init__(self, n_bins: int, in_channels: int = 1, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3, clip: float | None = CLIP):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_channels * n_bins, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, len(CLASSES)))
        if clip is not None:
            for p in self.parameters():
                p.register_hook(lambda g: g.clamp(-clip, clip))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        steps = x.permute(0, 3, 1, 2).reshape(b, t, c * f)        # (batch, time, channels·bins)
        out, _ = self.gru(self.project(steps))                      # (batch, time, 2·hidden)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        return self.head(pooled)
