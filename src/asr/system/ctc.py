from dataclasses import dataclass, field

import torch

from asr.loss.ctc import CTCLossConfig
from asr.model.encoder import EncoderConfig


class CTCSystem(torch.nn.Module):
    def __init__(self, encoder, loss, n_vocab: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.RMSNorm(encoder.out_dim),
            torch.nn.Linear(encoder.out_dim, n_vocab),
        )
        self.loss = loss

    def train_step(self, batch):
        x, y, xl, yl = batch
        with torch.amp.autocast("cuda"):
            enc, xl = self.encoder(x, xl)
            logits = self.head(enc)
        return self.loss(logits.float(), y, xl, yl)


@dataclass
class CTCSystemConfig:
    _target_: str = "asr.system.ctc.CTCSystem"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: CTCLossConfig = field(default_factory=CTCLossConfig)
