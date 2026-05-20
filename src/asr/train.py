from itertools import cycle, islice

import torch


class Trainer:
    def __init__(self, loader, eval_loader, system, optim, device):
        self.loader = loader
        self.eval_loader = eval_loader
        self.system = system
        self.optim = optim
        self.device = device

    def _to_device(self, batch):
        return (d.to(self.device, non_blocking=True) for d in batch)

    def train(self, total_steps: int, eval_steps: int | None, eval_every: int):
        for step, batch in enumerate(islice(cycle(self.loader), total_steps)):
            self.optim.zero_grad()
            loss = self.system.train_step(self._to_device(batch))
            loss.backward()
            self.optim.step()
            print(step, loss.item())

            if (step + 1) % eval_every == 0:
                metrics = self.eval(eval_steps)
                print(metrics)

    def eval(self, eval_steps: int | None = None) -> dict:
        rows = []
        for batch in islice(self.eval_loader, eval_steps):
            with torch.inference_mode():
                rows.extend(self.system.eval_step(self._to_device(batch)))
        return {
            "loss": sum(r["loss"] for r in rows) / len(rows),
            "wer": 100 * sum(r["wer_edit"] for r in rows) / sum(r["wer_ref_len"] for r in rows),
        }
