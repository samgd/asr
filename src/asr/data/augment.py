from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import MISSING

from asr.data.dataset import AudioFeatures


class TimeMask(torch.nn.Module):
    """Zero out a single contiguous span of time frames (SpecAugment time masking).

    Draws a mask width uniformly from ``[0, max_n]`` where
    ``max_n = min(max_n_mask_frame, max_prop_mask_frame * T)``, then places it at a
    uniformly random start so it stays within the ``T`` frames, and sets those frames
    to ``0``.

    Masks in place.

    Args:
        max_n_mask_frame: Upper bound on the masked span in frames.
        max_prop_mask_frame: Upper bound on the masked span as a fraction of the
            utterance length, capping ``max_n_mask_frame`` on short utterances.
    """

    def __init__(self, max_n_mask_frame: int, max_prop_mask_frame: float = 1.0):
        super().__init__()
        self.max_n_mask_frame = max_n_mask_frame
        self.max_prop_mask_frame = max_prop_mask_frame

    def forward(self, x: AudioFeatures) -> AudioFeatures:
        T = x.shape[0]
        max_n = min(self.max_n_mask_frame, int(self.max_prop_mask_frame * T))
        mask_n = int(torch.randint(low=0, high=max_n + 1, size=()))
        start = int(torch.randint(low=0, high=T - mask_n + 1, size=()))
        x[start : start + mask_n] = 0.0
        return x


class FreqMask(torch.nn.Module):
    """Zero out a single contiguous span of frequency bins (SpecAugment freq masking).

    Draws a mask width uniformly from ``[0, max_n_mask_freq]`` then places it at a
    uniformly random start so it stays within the ``F`` frequencies, and sets those
    frequencies to ``0``.

    Masks in place.

    Args:
        max_n_mask_freq: Upper bound on the masked span in frequencies.
    """

    def __init__(self, max_n_mask_freq: int):
        super().__init__()
        self.max_n_mask_freq = max_n_mask_freq

    def forward(self, x: AudioFeatures) -> AudioFeatures:
        F = x.shape[1]
        if self.max_n_mask_freq > F:
            raise ValueError(f"max_n_mask_freq={self.max_n_mask_freq} exceeds the number of frequency bins F={F}")
        mask_n = int(torch.randint(low=0, high=self.max_n_mask_freq + 1, size=()))
        start = int(torch.randint(low=0, high=F - mask_n + 1, size=()))
        x[:, start : start + mask_n] = 0.0
        return x


class Compose(torch.nn.Module):
    """Apply feature transforms sequentially."""

    def __init__(self, transforms: list[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x: AudioFeatures) -> AudioFeatures:
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class TimeMaskConfig:
    _target_: str = "asr.data.augment.TimeMask"
    max_n_mask_frame: int = MISSING
    max_prop_mask_frame: float = 1.0


@dataclass
class FreqMaskConfig:
    _target_: str = "asr.data.augment.FreqMask"
    max_n_mask_freq: int = MISSING


@dataclass
class ComposeConfig:
    _target_: str = "asr.data.augment.Compose"
    transforms: list[Any] = MISSING  # list of augmentation configs, each with its own _target_
