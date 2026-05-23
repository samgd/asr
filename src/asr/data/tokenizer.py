from dataclasses import dataclass, field


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
    _target_: str = "asr.data.CharTokenizer"
    specials: list[str] = field(default_factory=list)
