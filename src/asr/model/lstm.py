from dataclasses import dataclass

import torch
from jaxtyping import Float
from omegaconf import MISSING


class LSTM(torch.nn.Module):
    def __init__(self, depth: int, d_model: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=depth, batch_first=False)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.lstm(x)[0]


@dataclass
class LSTMConfig:
    _target_: str = "asr.model.lstm.LSTM"
    depth: int = MISSING
    d_model: int = MISSING
