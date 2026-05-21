from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import II, MISSING


def _make_norm(kind: str, channels: int) -> torch.nn.Module:
    if kind == "bn":
        return torch.nn.BatchNorm1d(channels)
    if kind == "none":
        return torch.nn.Identity()
    raise ValueError(f"unknown norm: {kind!r}")


def _make_activation(kind: str) -> torch.nn.Module:
    if kind == "silu":
        return torch.nn.SiLU()
    if kind == "relu":
        return torch.nn.ReLU()
    if kind == "gelu":
        return torch.nn.GELU()
    if kind == "none":
        return torch.nn.Identity()
    raise ValueError(f"unknown activation: {kind!r}")


class ConvFrontend(torch.nn.Module):
    def __init__(self, in_dim: int, layers: list[dict]):
        super().__init__()
        blocks: list[torch.nn.Module] = []
        kernels: list[int] = []
        strides: list[int] = []
        paddings: list[int] = []
        in_ch = in_dim
        for spec in layers:
            out_ch = int(spec["out_channels"])
            kernel = int(spec["kernel_size"])
            stride = int(spec["stride"])
            padding = (kernel - 1) // 2
            blocks.append(torch.nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding))
            blocks.append(_make_norm(spec["norm"], out_ch))
            blocks.append(_make_activation(spec["activation"]))
            kernels.append(kernel)
            strides.append(stride)
            paddings.append(padding)
            in_ch = out_ch
        self.net = torch.nn.Sequential(*blocks)
        self.out_dim = in_ch
        self._kernels = kernels
        self._strides = strides
        self._paddings = paddings

    def forward(self, x, lengths=None):
        x = self.net(x.transpose(1, 2)).transpose(1, 2)
        if lengths is None:
            return x, None
        for k, s, p in zip(self._kernels, self._strides, self._paddings, strict=True):
            lengths = (lengths + 2 * p - (k - 1) - 1) // s + 1
        return x, lengths


class Encoder(torch.nn.Module):
    def __init__(self, frontend, stem):
        super().__init__()
        self.frontend = frontend
        self.stem = stem

    def forward(self, x, lengths=None):
        x, lengths = self.frontend(x, lengths)
        out = self.stem(x)
        if lengths is None:
            return out
        return out, lengths

    @property
    def out_dim(self) -> int:
        return self.stem.d_model


@dataclass
class ConvBlockConfig:
    out_channels: int = MISSING
    kernel_size: int = 3
    stride: int = 1
    norm: str = "bn"
    activation: str = "silu"


@dataclass
class ConvFrontendConfig:
    _target_: str = "asr.model.encoder.ConvFrontend"
    in_dim: int = II("dataset.feature_dim")
    layers: list[ConvBlockConfig] = field(default_factory=list)


@dataclass
class EncoderConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"frontend": "conv"},
            {"stem": "transformer"},
        ]
    )
    _target_: str = "asr.model.encoder.Encoder"
    frontend: Any = MISSING
    stem: Any = MISSING
