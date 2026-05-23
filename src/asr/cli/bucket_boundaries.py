"""Suggest bucket boundaries from the configured dataset distribution.

Samples ``n_samples`` audio samples from the inner datasets under
``cfg.dataset.datasets`` (drawing each by ``cfg.dataset.weights``), records the
post-mel feature-frame length, and prints quantile-derived boundaries that
yield ``n_buckets`` equal-flow buckets.

Run with the same ``dataset=bucket dataset.datasets=[...] dataset.weights=[...]``
overrides used for training. Example::

    uv run bucket-boundaries \\
        dataset=bucket \\
        'dataset.datasets=[{_target_: asr.data.LibriSpeech, subset: train-clean-100}]' \\
        'dataset.weights=[1.0]' \\
        tokenizer=char 'tokenizer.specials=["<blank>"]' \\
        n_samples=2000 n_buckets=5
"""

from dataclasses import dataclass, field
from typing import Any

import hydra
import numpy as np
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

import asr.config  # noqa: F401  # registers tokenizer/dataset configs


@dataclass
class BoundariesConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"tokenizer": MISSING},
            {"dataset": MISSING},
        ]
    )
    tokenizer: Any = MISSING
    dataset: Any = MISSING
    n_samples: int = 10_000
    n_buckets: int = 8
    trim: float = 0.001  # fraction trimmed from each tail before quantile split
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="boundaries_config", node=BoundariesConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="boundaries_config")
def main(cfg: DictConfig) -> None:
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    datasets = [hydra.utils.instantiate(d, tokenizer=tokenizer) for d in cfg.dataset.datasets]
    weights = np.asarray(cfg.dataset.weights, dtype=float)
    norm_weights = weights / weights.sum()

    rng = np.random.default_rng(cfg.seed)
    lengths: list[int] = []
    for _ in tqdm.tqdm(range(cfg.n_samples), desc="sampling lengths"):
        ds_idx = int(rng.choice(len(datasets), p=norm_weights))
        ds = datasets[ds_idx]
        sample_idx = int(rng.integers(0, len(ds)))
        x, _ = ds[sample_idx]
        lengths.append(x.shape[0])

    arr = np.asarray(lengths)
    quantiles = np.linspace(cfg.trim, 1 - cfg.trim, cfg.n_buckets + 1)
    boundaries = np.quantile(arr, quantiles).round().astype(int).tolist()

    dropped = int(((arr < boundaries[0]) | (arr >= boundaries[-1])).sum())

    print(f"\nsampled {cfg.n_samples} lengths from {len(datasets)} dataset(s)")
    print(f"  min={int(arr.min())}  max={int(arr.max())}  mean={arr.mean():.0f}  median={int(np.median(arr))}")
    print(f"\nsuggested {cfg.n_buckets}-bucket boundaries (trim={cfg.trim:.2%} per tail):")
    print(f"  {boundaries}")
    print(f"  → {dropped}/{cfg.n_samples} ({dropped / cfg.n_samples:.2%}) samples fall outside this range")
    print("\nper-bucket sample counts (from this sample):")
    for k in range(cfg.n_buckets):
        lo, hi = boundaries[k], boundaries[k + 1]
        count = int(((arr >= lo) & (arr < hi)).sum())
        print(f"  [{lo:>5}, {hi:>5}): {count:>5}")


if __name__ == "__main__":
    main()
