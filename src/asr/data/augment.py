from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torchaudio
from omegaconf import MISSING

from asr.data.dataset import AudioFeatures, Waveform


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


class SpeedPerturbation(torch.nn.Module):
    """Randomly resample the waveform by one of ``factors``.

    Args:
        orig_freq: Sample rate of the input waveform in Hz.
        factors: Candidate speed factors, sampled uniformly per call.
    """

    def __init__(self, orig_freq: int = 16_000, factors: Sequence[float] = (0.9, 1.0, 1.1)):
        super().__init__()
        self.perturb = torchaudio.transforms.SpeedPerturbation(orig_freq, list(factors))

    def forward(self, wav: Waveform) -> Waveform:
        wav, _ = self.perturb(wav)
        return wav


class Compose(torch.nn.Module):
    """Apply transforms sequentially."""

    def __init__(self, transforms: list[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class Augment(torch.nn.Module):
    """Augmentation split into waveform-domain and feature-domain stages.

    ``audio`` transforms run on the raw waveform before mel extraction. ``feature``
    transforms run on the log-mel spectrogram after normalization. Either stage may be
    empty, in which case the corresponding ``*_forward`` is the identity.

    >>> aug = Augment()
    >>> x = torch.randn(7, 4)
    >>> bool((aug.feature_forward(x) == x).all())
    True

    Args:
        audio: Waveform-domain transforms, applied in order before mel extraction.
        feature: Feature-domain transforms, applied in order after normalization.
    """

    def __init__(
        self,
        audio: list[torch.nn.Module] | None = None,
        feature: list[torch.nn.Module] | None = None,
    ):
        super().__init__()
        self.audio = Compose(audio or [])
        self.feature = Compose(feature or [])

    def audio_forward(self, wav: Waveform) -> Waveform:
        return self.audio(wav)

    def feature_forward(self, feats: AudioFeatures) -> AudioFeatures:
        return self.feature(feats)


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
class SpeedPerturbationConfig:
    _target_: str = "asr.data.augment.SpeedPerturbation"
    orig_freq: int = 16_000
    factors: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])


@dataclass
class AugmentConfig:
    _target_: str = "asr.data.augment.Augment"
    # Each list holds augmentation configs, each with its own _target_.
    audio: list[Any] = field(default_factory=list)
    feature: list[Any] = field(default_factory=list)
