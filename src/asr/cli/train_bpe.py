import itertools
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import hydra
import sentencepiece as spm
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

from asr.data.librispeech import LibriSpeechTranscriptsConfig
from asr.data.tokenizer import MODEL_NAME


@dataclass
class SentencePieceTrainerConfig:
    vocab_size: int = MISSING


@dataclass
class TrainBpeConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"trainer": "sentencepiece"},
        ]
    )
    output_dir: str = MISSING
    transcript_datasets: list[Any] = field(
        default_factory=lambda: [
            LibriSpeechTranscriptsConfig(subset="train-clean-100"),
            LibriSpeechTranscriptsConfig(subset="train-clean-360"),
            LibriSpeechTranscriptsConfig(subset="train-other-500"),
        ]
    )
    trainer: Any = "sentencepiece"


cs = ConfigStore.instance()
cs.store(group="trainer", name="sentencepiece", node=SentencePieceTrainerConfig)
cs.store(name="train_bpe_config", node=TrainBpeConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="train_bpe_config")
def main(cfg: DictConfig) -> None:
    trainer = hydra.utils.instantiate(cfg.trainer)

    output_dir = cfg.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(hydra.utils.get_original_cwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, MODEL_NAME)

    sources = [hydra.utils.instantiate(d) for d in cfg.transcript_datasets]

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as f:
        for transcript in itertools.chain.from_iterable(sources):
            f.write(transcript + "\n")
        f.flush()

        spm.SentencePieceTrainer.train(  # type: ignore
            input=f.name,
            model_prefix=prefix,
            model_type="bpe",
            vocab_size=trainer.vocab_size,
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )


if __name__ == "__main__":
    main()
