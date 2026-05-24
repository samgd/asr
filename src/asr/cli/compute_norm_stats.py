"""Compute global per-feature mean/std for ``GlobalNorm`` from the training data.

Samples ``n_samples`` audio samples from the inner datasets under
``cfg.dataset.datasets`` (drawing each by ``cfg.dataset.weights``) and accumulates
per-feature mean and standard deviation over every frame, in float64.

Prints a paste-ready ``normalize`` config block. Drop it into a config (or
save it as ``conf/normalize/<name>.yaml`` and select with
``normalize=<name>``) to normalize with ``GlobalNorm``.

Stats are computed from raw log-mel features, so leave ``normalize`` unset on
the inner datasets here (it defaults to ``None``) or the stats will be wrong.

Run with the same ``dataset=bucket dataset.datasets=[...] dataset.weights=[...]``
overrides used for training, and the same mel settings. Recompute if those mel
settings ever change. Example::

    uv run compute-norm-stats \\
        dataset=bucket \\
        'dataset.datasets=[{_target_: asr.data.LibriSpeech, subset: train-clean-100}]' \\
        'dataset.weights=[1.0]' \\
        tokenizer=char 'tokenizer.specials=["<blank>"]' \\
        n_samples=20000
"""

from dataclasses import dataclass, field
from typing import Any

import hydra
import numpy as np
import torch
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

import asr.config  # noqa: F401  # registers tokenizer/dataset configs


@dataclass
class NormStatsConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"tokenizer": MISSING},
            {"dataset": MISSING},
        ]
    )
    tokenizer: Any = MISSING
    dataset: Any = MISSING
    n_samples: int = 20_000
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="norm_stats_config", node=NormStatsConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="norm_stats_config")
def main(cfg: DictConfig) -> None:
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    datasets = [hydra.utils.instantiate(d, tokenizer=tokenizer) for d in cfg.dataset.datasets]
    weights = np.asarray(cfg.dataset.weights, dtype=float)
    norm_weights = weights / weights.sum()

    rng = np.random.default_rng(cfg.seed)
    fsum: torch.Tensor | None = None
    fsqsum: torch.Tensor | None = None
    frames = 0
    for _ in tqdm.tqdm(range(cfg.n_samples), desc="accumulating stats"):
        ds_idx = int(rng.choice(len(datasets), p=norm_weights))
        ds = datasets[ds_idx]
        sample_idx = int(rng.integers(0, len(ds)))
        x, _ = ds[sample_idx]  # (n_frames, n_feats)
        xd = x.double()
        if fsum is None:
            fsum = torch.zeros(xd.shape[1], dtype=torch.float64)
            fsqsum = torch.zeros(xd.shape[1], dtype=torch.float64)
        fsum += xd.sum(dim=0)
        fsqsum += (xd * xd).sum(dim=0)
        frames += xd.shape[0]

    assert fsum is not None and fsqsum is not None, "no samples accumulated"
    mean = fsum / frames
    var = (fsqsum / frames - mean * mean).clamp_min(1e-10)
    std = var.sqrt()

    subsets = [d.get("subset", "?") for d in cfg.dataset.datasets]
    mean_str = ", ".join(f"{v:.6f}" for v in mean.tolist())
    std_str = ", ".join(f"{v:.6f}" for v in std.tolist())

    print(f"\naccumulated {frames} frames from {cfg.n_samples} samples over {len(datasets)} dataset(s)")
    print(f"  feature_dim={mean.numel()}  subsets={subsets}  weights={cfg.dataset.weights}")
    print("\npaste into the normalize group (recompute if mel settings change):\n")
    print(f"# computed from {frames} frames; subsets={subsets} weights={list(cfg.dataset.weights)}")
    print("_target_: asr.data.norm.GlobalNorm")
    print(f"mean: [{mean_str}]")
    print(f"std: [{std_str}]")


if __name__ == "__main__":
    main()
