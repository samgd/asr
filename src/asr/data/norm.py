from dataclasses import dataclass

import torch
from jaxtyping import Float
from omegaconf import MISSING


class DynamicPerSamplePerFeatureNorm(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "time features"]) -> Float[torch.Tensor, "time features"]:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt()


class GlobalNorm(torch.nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[None, :])
        self.register_buffer("std", torch.tensor(std)[None, :])

    def forward(self, x: Float[torch.Tensor, "time features"]) -> Float[torch.Tensor, "time features"]:
        return (x.float() - self.mean) / self.std  # type: ignore


@dataclass
class DynamicPerSamplePerFeatureNormConfig:
    _target_: str = "asr.data.norm.DynamicPerSamplePerFeatureNorm"


@dataclass
class GlobalNormConfig:
    _target_: str = "asr.data.norm.GlobalNorm"
    mean: list[float] = MISSING
    std: list[float] = MISSING
