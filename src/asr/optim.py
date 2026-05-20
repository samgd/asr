from dataclasses import dataclass


@dataclass
class AdamWConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
