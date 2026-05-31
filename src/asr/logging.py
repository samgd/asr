from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import aim
import torch
import tqdm
from omegaconf import II, MISSING


class Logger(Protocol):
    def set_config(self, config: dict) -> None: ...
    def scalar(self, name: str, value: float, *, step: int, stage: str) -> None: ...
    def scalars(self, values: dict[str, float], *, step: int, stage: str) -> None: ...
    def distribution(self, name: str, values: torch.Tensor | Sequence[float], *, step: int, stage: str) -> None: ...
    def text(self, name: str, value: str, *, step: int, stage: str) -> None: ...
    def __enter__(self) -> "Logger": ...
    def __exit__(self, *exc) -> None: ...


class TqdmLogger:
    def __init__(self, total_steps: int):
        self.pbar = None
        self.total_steps = total_steps

    def set_config(self, config: dict) -> None:
        pass

    def scalar(self, name: str, value: float, *, step: int, stage: str) -> None:
        self.scalars({name: value}, step=step, stage=stage)

    def scalars(self, values: dict[str, float], *, step: int, stage: str) -> None:
        assert self.pbar is not None
        values = {k: v for k, v in values.items() if v is not None}
        if stage == "train":
            self.pbar.n = step
            self.pbar.set_postfix(values)
            self.pbar.refresh()
        else:
            formatted = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
            self.pbar.write(f"{stage} @ {step}: {formatted}")

    def distribution(self, name: str, values: torch.Tensor | Sequence[float], *, step: int, stage: str) -> None:
        assert self.pbar is not None
        t = values if isinstance(values, torch.Tensor) else torch.as_tensor(list(values), dtype=torch.float32)
        if t.numel() == 0:
            self.pbar.write(f"{stage} @ {step}: {name}: (empty)")
            return
        t = t.float()
        self.pbar.write(f"{stage} @ {step}: {name}: mean={t.mean():.3f} std={t.std():.3f} n={t.numel()}")

    def text(self, name: str, value: str, *, step: int, stage: str) -> None:
        assert self.pbar is not None
        self.pbar.write(f"{stage} @ {step}: {name}: {value}")

    def __enter__(self) -> "TqdmLogger":
        self.pbar = tqdm.tqdm(total=self.total_steps, desc="train", unit="step", smoothing=0.1)
        return self

    def __exit__(self, *exc) -> None:
        if self.pbar is not None:
            self.pbar.close()


class AimLogger:
    def __init__(self, repo: str):
        self.repo = repo
        self.run = None
        self.config = None

    def set_config(self, config: dict) -> None:
        self.config = config

    def scalar(self, name: str, value: float, *, step: int, stage: str) -> None:
        assert self.run is not None
        self.run.track(value, name=name, step=step, context={"subset": stage})

    def scalars(self, values: dict[str, float], *, step: int, stage: str) -> None:
        assert self.run is not None
        clean = {k: v for k, v in values.items() if v is not None}
        if clean:
            self.run.track(clean, context={"subset": stage}, step=step)

    def distribution(self, name: str, values: torch.Tensor | Sequence[float], *, step: int, stage: str) -> None:
        assert self.run is not None
        arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else list(values)
        self.run.track(aim.Distribution(arr), name=name, step=step, context={"subset": stage})

    def text(self, name: str, value: str, *, step: int, stage: str) -> None:
        assert self.run is not None
        self.run.track(aim.Text(value), name=name, step=step, context={"subset": stage})

    def __enter__(self) -> "Logger":
        self.run = aim.Run(repo=self.repo)
        if self.config is not None:
            self.run["config"] = self.config
        return self

    def __exit__(self, *exc) -> None:
        if self.run is not None:
            self.run.close()


@dataclass
class TqdmLoggerConfig:
    _target_: str = "asr.logging.TqdmLogger"
    total_steps: int = II("total_steps")


@dataclass
class AimLoggerConfig:
    _target_: str = "asr.logging.AimLogger"
    repo: str = MISSING
