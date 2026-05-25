from dataclasses import dataclass
from typing import Any, Literal, Protocol

import aim
import tqdm
from omegaconf import II, MISSING


@dataclass
class Event:
    stage: Literal["train", "eval"]
    step: int
    values: dict[str, Any]


class Logger(Protocol):
    def set_config(self, config: dict) -> None: ...
    def append(self, event: Event) -> None: ...
    def __enter__(self) -> "Logger": ...
    def __exit__(self, *exc) -> None: ...


class TqdmLogger:
    def __init__(self, total_steps: int):
        self.pbar = None
        self.total_steps = total_steps

    def set_config(self, config: dict) -> None:
        pass

    def append(self, event: Event):
        assert self.pbar is not None, (
            "TqdmLogger.append() called outside its context manager - wrap usage in `with logger: ...`"
        )
        match event.stage:
            case "train":
                self.pbar.n = event.step
                self.pbar.set_postfix({k: v for k, v in event.values.items() if v is not None})
                self.pbar.refresh()
            case "eval":
                formatted = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in event.values.items()
                )
                self.pbar.write(f"eval @ {event.step}: {formatted}")
            case _:
                raise NotImplementedError(f"unknown {event.stage=}")

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

    def append(self, event: Event) -> None:
        assert self.run is not None, (
            "AimLogger.append() called outside its context manager - wrap usage in `with logger: ...`"
        )
        self.run.track(event.values, context={"subset": event.stage}, step=event.step)

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
