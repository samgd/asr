import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from omegaconf import II, MISSING
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import MelSpectrogram

from asr.data.dataset import FeatureTransform, Sample


class LibriSpeech(Dataset[Sample]):
    Subset = Literal[
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    def __init__(
        self,
        subset: Subset,
        tokenizer,
        sample_rate: int = 16000,
        n_mels: int = 128,
        win_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        f_min: float = 40.0,
        f_max: float | None = 8000.0,
        root: str | None = None,
        normalize: FeatureTransform | None = None,
        augment: FeatureTransform | None = None,
        download: bool = False,
        **kwargs,
    ):
        if root is None:
            root = os.environ["DATA_ROOT"]
        self.base = LIBRISPEECH(root=root, url=subset, download=download)
        self.normalize = normalize
        self.augment = augment
        self.tok = tokenizer
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Sample:
        wav, _, text, *_ = self.base[i]
        feats = self.mel(wav).squeeze(0).clamp(min=1e-10).log().T
        if self.normalize is not None:
            feats = self.normalize(feats)
        if self.augment is not None:
            feats = self.augment(feats)
        ids = torch.tensor(self.tok.encode(text), dtype=torch.long)
        return feats, ids


@dataclass
class LibriSpeechConfig:
    _target_: str = "asr.data.LibriSpeech"
    subset: str = MISSING
    sample_rate: int = 16_000
    n_mels: int = 128
    win_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    f_min: float = 40.0
    f_max: float | None = 8000.0
    root: str | None = None
    download: bool = False
    normalize: Any = None
    augment: Any = None
    feature_dim: int = II(".n_mels")
