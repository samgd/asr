import os
from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Float, Int64
from omegaconf import MISSING
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import MelSpectrogram

Sample = tuple[
    # Audio features
    Float[torch.Tensor, "n_frames n_feats"],
    # Transcription
    Int64[torch.Tensor, "seq_len"],
]

Batch = tuple[
    # Batch of audio features
    Float[torch.Tensor, "batch n_frames n_feats"],
    # Batch of transcripts
    Int64[torch.Tensor, "batch seq_len"],
    # Audio frame lengths
    Int64[torch.Tensor, "batch"],
    # Transcript lengths
    Int64[torch.Tensor, "batch"],
]


class CharTokenizer:
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '"

    def __init__(self, specials: tuple[str, ...] = ()):
        self.specials = specials
        self.i2s = [*specials, *self.ALPHABET]
        self.s2i = {c: i for i, c in enumerate(self.i2s)}

    def __len__(self) -> int:
        return len(self.i2s)

    def encode(self, text: str) -> list[int]:
        return [self.s2i[c] for c in text.upper()]

    def decode(self, ids: list[int], skip_specials: bool = True) -> str:
        special_ids = {self.s2i[s] for s in self.specials} if skip_specials else set()
        return "".join(self.i2s[i] for i in ids if i not in special_ids)


@dataclass
class CharTokenizerConfig:
    __target__: str = "asr.data.CharTokenizer"
    specials: list[str] = field(default_factory=list)


def collate_fn(samples: list[Sample]) -> Batch:
    """Pad a list of (feats, ids) samples into a batch with length tensors.

    >>> a_feats, b_feats = torch.zeros(3, 4), torch.ones(5, 4)
    >>> a_ids = torch.tensor([1, 2], dtype=torch.long)
    >>> b_ids = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    >>> pf, pt, lf, lt = collate_fn([(a_feats, a_ids), (b_feats, b_ids)])
    >>> pf.shape, pt.shape
    (torch.Size([2, 5, 4]), torch.Size([2, 4]))
    >>> lf.tolist(), lt.tolist()
    ([3, 5], [2, 4])
    """
    feats, trans = zip(*samples, strict=True)
    pad_feats = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True)
    len_feats = torch.tensor([len(f) for f in feats], dtype=torch.long)
    pad_trans = torch.nn.utils.rnn.pad_sequence(list(trans), batch_first=True)
    len_trans = torch.tensor([len(t) for t in trans], dtype=torch.long)
    return pad_feats, pad_trans, len_feats, len_trans


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
        **kwargs,
    ):
        if root is None:
            root = os.environ["DATA_ROOT"]
        self.base = LIBRISPEECH(root=root, url=subset, **kwargs)
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
        ids = torch.tensor(self.tok.encode(text), dtype=torch.long)
        return feats, ids


@dataclass
class LibriSpeechConfig:
    __target__: str = "asr.data.LibriSpeech"
    subset: str = MISSING
    sample_rate: int = 16_000
    n_mels: int = 128
    win_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    f_min: float = 40.0
    f_max: float | None = 8000.0
    root: str | None = None
