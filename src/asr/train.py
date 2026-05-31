from itertools import islice

import torch

from asr.logging import Logger
from asr.system import System, to_device


def _repeat(loader):
    while True:
        yield from loader


class Trainer:
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        system: System,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        logger: Logger,
        device: str | torch.device,
        max_grad_norm: float | None = None,
    ):
        self.loader = loader
        self.eval_loader = eval_loader
        self.system = system
        self.optim = optim
        self.sched = sched
        self.logger = logger
        self.device = device
        self.max_grad_norm = max_grad_norm

    def train(self, total_steps: int, eval_steps: int | None, eval_every: int):
        self.system.train()
        with self.logger as logger:
            for step, batch in enumerate(islice(_repeat(self.loader), total_steps), start=1):
                self.optim.zero_grad()
                loss = self.system.train_step(to_device(batch, self.device))
                loss.backward()
                if self.max_grad_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.max_grad_norm).cpu().item()
                else:
                    norm = None
                self.optim.step()
                lr = self.sched.get_last_lr()
                self.sched.step()

                metrics = {"loss": loss.item(), "lr": lr[0]}
                if norm is not None:
                    metrics["norm"] = norm
                logger.scalars(metrics, step=step, stage="train")

                if step % eval_every == 0:
                    # Each system owns its eval loop and logs whatever metrics it chooses.
                    self.system.evaluate(
                        islice(self.eval_loader, eval_steps),
                        logger,
                        step=step,
                        device=self.device,
                    )
