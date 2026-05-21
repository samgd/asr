from dataclasses import dataclass
from typing import Any, Literal, Protocol

import tqdm
from omegaconf import II


@dataclass
class Event:
    stage: Literal["train", "eval"]
    step: int
    values: dict[str, Any]


class Logger(Protocol):
    def append(self, event: Event) -> None: ...
    def __enter__(self) -> "Logger": ...
    def __exit__(self, *exc) -> None: ...


class TqdmLogger:
    def __init__(self, total_steps: int):
        self.pbar = None
        self.total_steps = total_steps

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


@dataclass
class TqdmLoggerConfig:
    _target_: str = "asr.logging.TqdmLogger"
    total_steps: int = II("total_steps")
