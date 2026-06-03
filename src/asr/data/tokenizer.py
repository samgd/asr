import os
from dataclasses import dataclass, field
from typing import Any

import hydra
import sentencepiece as spm
from omegaconf import MISSING

MODEL_NAME = "bpe"


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

    def decode(self, ids: list[int]) -> str:
        special_ids = {self.s2i[s] for s in self.specials}
        return "".join(self.i2s[i] for i in ids if i not in special_ids)


class BPETokenizer:
    def __init__(self, model_dir: str, specials: tuple[str, ...] = ()):
        self.specials = specials
        self._n_specials = len(self.specials)
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(hydra.utils.get_original_cwd(), model_dir)
        prefix = os.path.join(model_dir, MODEL_NAME)
        self.sp: Any = spm.SentencePieceProcessor()
        self.sp.load(f"{prefix}.model")

    def __len__(self) -> int:
        return self._n_specials + self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return [self._n_specials + i for i in self.sp.encode(text.upper(), out_type=int)]

    def decode(self, ids: list[int]) -> str:
        pieces = [i - self._n_specials for i in ids if i >= self._n_specials]
        return self.sp.decode_ids(pieces)


@dataclass
class CharTokenizerConfig:
    _target_: str = "asr.data.CharTokenizer"
    specials: list[str] = field(default_factory=list)


@dataclass
class BPETokenizerConfig:
    _target_: str = "asr.data.BPETokenizer"
    model_dir: str = MISSING
    specials: list[str] = field(default_factory=list)
