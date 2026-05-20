from dataclasses import dataclass

import torch
from omegaconf import II


def linear_warmup_cosine_decay(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.01
):
    """Return a linear-warmup cosine-decay learning rate scheduler."""
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


@dataclass
class LinearWarmupCosineDecayConfig:
    _target_: str = "asr.sched.linear_warmup_cosine_decay"
    warmup_steps: int = 1000
    total_steps: int = II("total_steps")
    min_lr: float = 0.01
