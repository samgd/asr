from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from asr.data import CharTokenizerConfig
from asr.loss.ctc import CTCLossConfig
from asr.model.transformer import TransformerConfig


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"model": MISSING},
            {"loss": MISSING},
            {"tokenizer": MISSING},
            {"data": MISSING},
        ]
    )
    model: Any = MISSING
    loss: Any = MISSING
    tokenizer: Any = MISSING
    data: Any = MISSING
    max_steps: int = 100_000


cs = ConfigStore.instance()
cs.store(group="model", name="transformer", node=TransformerConfig)
cs.store(group="loss", name="ctc", node=CTCLossConfig)
cs.store(group="tokenizer", name="char", node=CharTokenizerConfig)
cs.store(group="data", name="librispeech", node=CharTokenizerConfig)
cs.store(name="config", node=TrainConfig)
