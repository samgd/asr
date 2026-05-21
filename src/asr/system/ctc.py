from dataclasses import dataclass, field
from typing import Any

import jiwer
import torch
from jaxtyping import Float
from omegaconf import MISSING

from asr.data import Batch
from asr.decode.ctc import greedy_decode
from asr.loss.ctc import CTCLossConfig


class CTCSystem(torch.nn.Module):
    def __init__(self, encoder, loss, n_vocab: int, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.RMSNorm(encoder.out_dim),
            torch.nn.Linear(encoder.out_dim, n_vocab),
        )
        self.loss = loss
        self.tokenizer = tokenizer

    def train_step(self, batch: Batch) -> Float[torch.Tensor, ""]:
        x, y, xl, yl = batch
        with torch.autocast("cuda"):
            enc, xl = self.encoder(x, xl)
            logits = self.head(enc)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return self.loss(log_probs, y, xl, yl).mean()

    def eval_step(self, batch: Batch) -> list[dict]:
        x, y, xl, yl = batch
        with torch.autocast("cuda"):
            enc, xl = self.encoder(x, xl)
            logits = self.head(enc)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        loss_per = self.loss(log_probs, y, xl, yl)
        hyps = greedy_decode(log_probs, xl)
        out = []
        for i in range(loss_per.shape[0]):
            ref = self.tokenizer.decode(y[i, : yl[i]].tolist())
            hyp = self.tokenizer.decode(hyps[i])
            wo = jiwer.process_words(ref, hyp)
            edits = wo.substitutions + wo.deletions + wo.insertions
            ref_len = len(ref.split())
            out.append({"loss": loss_per[i].item(), "wer_edit": edits, "wer_ref_len": ref_len})
        return out


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
