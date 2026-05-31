from collections.abc import Iterable
from typing import Protocol, cast, runtime_checkable

import torch

from asr.data.dataset import Batch
from asr.logging import Logger


def to_device(batch: Batch, device: str | torch.device) -> Batch:
    return cast(Batch, tuple(d.to(device, non_blocking=True) for d in batch))


@runtime_checkable
class System(Protocol):
    training: bool

    def train_step(self, batch: Batch) -> torch.Tensor: ...
    def evaluate(self, loader: Iterable[Batch], logger: Logger, step: int, device: str | torch.device) -> None: ...
    def parameters(self) -> list[torch.Tensor]: ...
    def to(self, device): ...
    def train(self, mode: bool = True) -> "System": ...
    def eval(self) -> "System": ...
