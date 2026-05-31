from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from jaxtyping import Float
from omegaconf import MISSING

from asr.data.dataset import Batch
from asr.decode.ctc import beam_decode, greedy_decode
from asr.logging import Logger
from asr.loss.ctc import CTCLossConfig
from asr.system import to_device
from asr.system.metrics import wer_counts


class CTCSystem(torch.nn.Module):
    def __init__(self, encoder, loss, n_vocab: int, tokenizer, decode: Literal["greedy", "beam"]):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.RMSNorm(encoder.out_dim),
            torch.nn.Linear(encoder.out_dim, n_vocab),
        )
        self.loss = loss
        self.tokenizer = tokenizer
        self.decode = decode
        match self.decode:
            case "greedy":
                self.decode_fn = greedy_decode
            case "beam":
                self.decode_fn = beam_decode
            case _:
                raise NotImplementedError()

    def train_step(self, batch: Batch) -> Float[torch.Tensor, ""]:
        x, y, xl, yl = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, xl = self.encoder(x, xl)
            logits = self.head(enc)
        return self.loss(logits, y, xl, yl).mean()

    def evaluate(self, loader: Iterable[Batch], logger: Logger, step: int, device: str | torch.device) -> None:
        was_training = self.training
        self.eval()
        losses: list[float] = []
        edits, ref_words = 0, 0
        for batch in loader:
            batch = to_device(batch, device)
            x, y, xl, yl = batch
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc, xl = self.encoder(x, xl)
                logits = self.head(enc)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            losses.extend(self.loss(logits, y, xl, yl).tolist())
            hyps = self.decode_fn(log_probs, xl)
            for i in range(len(hyps)):
                ref = self.tokenizer.decode(y[i, : yl[i]].tolist())
                hyp = self.tokenizer.decode(hyps[i])
                e, w = wer_counts(ref, hyp)
                edits += e
                ref_words += w
        self.train(was_training)
        if losses:
            logger.scalars(
                {"loss": sum(losses) / len(losses), "wer": 100 * edits / max(ref_words, 1)},
                step=step,
                stage="eval",
            )


@dataclass
class CTCSystemConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"encoder": "default"},
        ]
    )
    _target_: str = "asr.system.ctc.CTCSystem"
    encoder: Any = MISSING
    loss: CTCLossConfig = field(default_factory=CTCLossConfig)
    decode: str = "beam"
