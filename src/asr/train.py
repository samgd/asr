from itertools import cycle, islice


class Trainer:
    def __init__(self, loader, system, optim, device):
        self.loader = loader
        self.system = system
        self.optim = optim
        self.device = device

    def train(self, total_steps: int, eval_every: int):
        for step, batch in enumerate(islice(cycle(self.loader), total_steps)):
            batch = (d.to(self.device, non_blocking=True) for d in batch)
            self.optim.zero_grad()
            loss = self.system.train_step(batch)
            loss.backward()
            self.optim.step()
            print(step, loss.item())
