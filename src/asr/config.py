from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from asr.data.augment import ComposeConfig, TimeMaskConfig
from asr.data.bucket_dataset import BucketDatasetConfig
from asr.data.dataset import BucketDataLoaderConfig, DataLoaderConfig
from asr.data.librispeech import LibriSpeechConfig
from asr.data.norm import DynamicPerSamplePerFeatureNormConfig, GlobalNormConfig
from asr.data.tokenizer import CharTokenizerConfig
from asr.logging import AimLoggerConfig, TqdmLoggerConfig
from asr.model.encoder import ConvFrontendConfig, EncoderConfig
from asr.model.lstm import LSTMConfig
from asr.model.transformer import TransformerConfig
from asr.optim import AdamWConfig
from asr.sched import LinearWarmupCosineDecayConfig
from asr.system.ctc import CTCSystemConfig
from asr.system.rnnt import RNNTSystemConfig


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"normalize": "dynamicpersample"},
            {"augment": None},
            {"system": MISSING},
            {"tokenizer": MISSING},
            {"dataset": MISSING},
            {"dataloader": "default"},
            {"eval_dataset": MISSING},
            {"eval_dataloader": "default"},
            {"optim": "adamw"},
            {"sched": "linear_warmup_cosine_decay"},
            {"logger": MISSING},
        ]
    )
    normalize: Any = MISSING
    augment: Any = None
    system: Any = MISSING
    tokenizer: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    eval_dataset: Any = MISSING
    eval_dataloader: Any = MISSING
    optim: Any = MISSING
    sched: Any = MISSING
    logger: Any = MISSING
    total_steps: int = MISSING
    eval_steps: int | None = None
    eval_every: int = MISSING
    device: str = MISSING
    max_grad_norm: float | None = 1.0


cs = ConfigStore.instance()

cs.store(group="normalize", name="global", node=GlobalNormConfig)
cs.store(group="normalize", name="dynamicpersample", node=DynamicPerSamplePerFeatureNormConfig)

cs.store(group="augment", name="timemask", node=TimeMaskConfig)
cs.store(group="augment", name="compose", node=ComposeConfig)

cs.store(group="system", name="ctc", node=CTCSystemConfig)
cs.store(group="system", name="rnnt", node=RNNTSystemConfig)
cs.store(group="system/encoder", name="default", node=EncoderConfig)
cs.store(group="system/encoder/frontend", name="conv", node=ConvFrontendConfig)
cs.store(group="system/encoder/stem", name="transformer", node=TransformerConfig)
cs.store(group="system/decoder", name="lstm", node=LSTMConfig)

cs.store(group="tokenizer", name="char", node=CharTokenizerConfig)

cs.store(group="dataset", name="librispeech", node=LibriSpeechConfig)
cs.store(group="dataset", name="bucket", node=BucketDatasetConfig)

cs.store(group="dataloader", name="default", node=DataLoaderConfig)
cs.store(group="dataloader", name="bucket", node=BucketDataLoaderConfig)

cs.store(group="eval_dataset", name="librispeech", node=LibriSpeechConfig)
cs.store(group="eval_dataloader", name="default", node=DataLoaderConfig)

cs.store(group="optim", name="adamw", node=AdamWConfig)

cs.store(group="sched", name="linear_warmup_cosine_decay", node=LinearWarmupCosineDecayConfig)

cs.store(group="logger", name="aim", node=AimLoggerConfig)
cs.store(group="logger", name="tqdm", node=TqdmLoggerConfig)

cs.store(name="config", node=TrainConfig)
