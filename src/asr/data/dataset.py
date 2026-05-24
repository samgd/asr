from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float, Int64
from omegaconf import MISSING

AudioFeatures = Float[torch.Tensor, "n_frames n_feats"]

FeatureTransform = Callable[[AudioFeatures], AudioFeatures]

Sample = tuple[
    AudioFeatures,
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


def identity(x: Batch) -> Batch:
    return x


@dataclass
class DataLoaderConfig:
    _target_: str = "torch.utils.data.DataLoader"
    dataset: Any = MISSING
    batch_size: int = MISSING
    num_workers: int = 4
    shuffle: bool = True
    collate_fn: Any = field(default_factory=lambda: {"_target_": "asr.data.collate_fn", "_partial_": True})


@dataclass
class BucketDataLoaderConfig:
    _target_: str = "torch.utils.data.DataLoader"
    dataset: Any = MISSING
    batch_size: int | None = None
    num_workers: int = 4
    collate_fn: Any = field(default_factory=lambda: {"_target_": "asr.data.identity", "_partial_": True})
