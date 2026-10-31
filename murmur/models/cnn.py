"""A small 2D CNN over spectrogram windows: (batch, channels, frequency bins, frames) -> 3 class logits."""
from __future__ import annotations

from torch import nn

from murmur.core.constants import CLASSES


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """Four conv blocks (16-32-64-128 channels), global average pooling, dropout, a linear layer.
    About 100k weights: the training set is ~750 patients, so the network is kept small."""

    def __init__(self, dropout: float = 0.3, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(_block(in_channels, 16, True), _block(16, 32, True), _block(32, 64, True),
                                      _block(64, 128, False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(128, len(CLASSES)))

    def forward(self, x):
        return self.head(self.features(x))
