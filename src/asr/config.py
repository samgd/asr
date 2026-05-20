from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from asr.data import CharTokenizerConfig, DataLoaderConfig, LibriSpeechConfig
from asr.optim import AdamWConfig
from asr.system.ctc import CTCSystemConfig


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"system": MISSING},
            {"tokenizer": MISSING},
            {"dataset": MISSING},
            {"dataloader": "default"},
            {"optim": "adamw"},
        ]
    )
    system: Any = MISSING
    tokenizer: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    optim: Any = MISSING
    total_steps: int = MISSING
    eval_every: int = MISSING
    device: str = MISSING


cs = ConfigStore.instance()

cs.store(group="system", name="ctc", node=CTCSystemConfig)

cs.store(group="tokenizer", name="char", node=CharTokenizerConfig)

cs.store(group="dataset", name="librispeech", node=LibriSpeechConfig)
cs.store(group="dataloader", name="default", node=DataLoaderConfig)

cs.store(group="optim", name="adamw", node=AdamWConfig)

cs.store(name="config", node=TrainConfig)
