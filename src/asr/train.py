from itertools import cycle, islice
from typing import cast

import torch

from asr.data import Batch
from asr.system import System


class Trainer:
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        system: System,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        device: str | torch.device,
        max_grad_norm: float | None = None,
    ):
        self.loader = loader
        self.eval_loader = eval_loader
        self.system = system
        self.optim = optim
        self.sched = sched
        self.device = device
        self.max_grad_norm = max_grad_norm

    def _to_device(self, batch: Batch) -> Batch:
        return cast(Batch, tuple(d.to(self.device, non_blocking=True) for d in batch))

    def train(self, total_steps: int, eval_steps: int | None, eval_every: int):
        for step, batch in enumerate(islice(cycle(self.loader), total_steps)):
            self.optim.zero_grad()
            loss = self.system.train_step(self._to_device(batch))
            loss.backward()
            if self.max_grad_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.max_grad_norm).cpu().item()
            else:
                norm = None
            self.optim.step()
            lr = self.sched.get_last_lr()
            self.sched.step()
            print(step, loss.item(), norm, lr)

            if (step + 1) % eval_every == 0:
                metrics = self.eval(eval_steps)
                print(metrics)

    def eval(self, eval_steps: int | None = None) -> dict:
        rows = []
        for batch in islice(self.eval_loader, eval_steps):
            with torch.no_grad():
                rows.extend(self.system.eval_step(self._to_device(batch)))
        return {
            "loss": sum(r["loss"] for r in rows) / len(rows),
            "wer": 100 * sum(r["wer_edit"] for r in rows) / sum(r["wer_ref_len"] for r in rows),
        }
