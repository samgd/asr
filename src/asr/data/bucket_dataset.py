from collections import deque
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from omegaconf import MISSING

from asr.data.dataset import Batch, Sample, collate_fn


class BucketDataset(torch.utils.data.IterableDataset):
    """Stream length-bucketed batches from a weighted mixture of source datasets.

    Samples are drawn from sources by weight, routed by length into K bucket buffers,
    and emitted as variable-size batches drawn one bucket at a time (uniform random).

    A batch grows until adding another sample would exceed ``batch_frame_budget``. The
    buffer is capped at ``max_bucket_count`` samples. Overflow evicts the oldest sample
    from the largest non-target bucket, preserving rare-bucket samples since rare
    buckets are never the largest.

    Args:
        datasets: Source map-style datasets to draw from.
        weights: Per-dataset sampling probabilities. Do not need to sum to 1.
        boundaries: K+1 length cutoffs in feature frames defining K half-open buckets
            ``[b_k, b_{k+1})``. Samples outside ``[boundaries[0], boundaries[-1])`` are
            dropped.
        batch_frame_budget: Maximum cumulative feature frames per batch. Must be at
            least ``boundaries[-1]`` so any in-range sample can fit alone.
        max_bucket_count: Cap on the total number of samples held across all bucket
            buffers.
        seed: Base seed. Combined with worker id for per-worker stream independence.
    """

    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        weights: list[float],
        boundaries: list[int],
        batch_frame_budget: int,
        max_bucket_count: int,
        seed: int = 0,
        **kwargs,
    ):
        assert len(datasets) == len(weights)
        assert len(boundaries) >= 2, "Must set [lower, upper) boundary for at least 1 bucket."
        assert all(left < right for left, right in zip(boundaries, boundaries[1:], strict=False)), (
            "Bucket boundaries must not overlap nor produce buckets with zero size."
        )
        assert batch_frame_budget >= boundaries[-1], (
            "Batch must be able to contain at least one sample from largest bucket."
        )
        self.datasets = datasets
        self.weights = weights
        self.norm_weights = np.asarray(weights) / sum(weights)
        self.boundaries = boundaries
        self.batch_frame_budget = batch_frame_budget
        self.max_bucket_count = max_bucket_count
        self.seed = seed

        self.n_datasets = len(datasets)
        self.n_buckets = len(self.boundaries) - 1
        self.buckets = [deque() for _ in range(self.n_buckets)]
        self.bucket_sizes = [0 for _ in range(self.n_buckets)]
        self.total = 0

    def __iter__(self) -> Iterator[Batch]:
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + worker_id)
        streams = [self._stream(d, worker_id, num_workers, rng) for d in self.datasets]
        while True:
            bucket_idx = rng.integers(low=0, high=self.n_buckets)

            batch = []
            batch_frames = 0

            while True:
                while self.bucket_sizes[bucket_idx] == 0:
                    # selected bucket empty, fill buckets until selected contains a sample
                    stream_idx = rng.choice(len(streams), p=self.norm_weights)
                    sample = next(streams[stream_idx])
                    fill_idx = np.searchsorted(self.boundaries, sample[0].shape[0], side="right") - 1
                    self.buckets[fill_idx].append(sample)
                    self.bucket_sizes[fill_idx] += 1
                    self.total += 1
                    if self.total >= self.max_bucket_count:
                        # total bucket size exceeded, drop a sample from most full bucket
                        drop_idx = np.argmax(self.bucket_sizes)
                        if bucket_idx == drop_idx:
                            print(
                                f"ASSERT FAILURE: bucket_idx={bucket_idx}, drop_idx={drop_idx}, "
                                f"sizes={self.bucket_sizes}, total={self.total}, max={self.max_bucket_count}",
                                flush=True,
                            )
                        assert bucket_idx != drop_idx, "Should never drop bucket being filled."
                        self.buckets[drop_idx].popleft()
                        self.bucket_sizes[drop_idx] -= 1
                        self.total -= 1

                # peak at next sample
                sample = self.buckets[bucket_idx][0]
                sample_frames = sample[0].shape[0]

                if batch_frames + sample_frames > self.batch_frame_budget:
                    # batch full; stop
                    break

                # batch not full; pop sample, update sizes, and add to batch
                self.buckets[bucket_idx].popleft()
                self.bucket_sizes[bucket_idx] -= 1
                self.total -= 1
                batch_frames += sample_frames
                batch.append(sample)

            yield collate_fn(batch)

    def _stream(
        self, dataset: torch.utils.data.Dataset[Sample], worker_id: int, num_workers: int, rng: np.random.Generator
    ) -> Iterator[Sample]:
        total_samples = len(cast(Sized, dataset))  # type checker thinks dataset lacks len?
        samples_per_shard = total_samples // num_workers
        assert samples_per_shard > 0, "Must be at least one sample per worker."
        start = samples_per_shard * worker_id
        end = total_samples if worker_id + 1 == num_workers else samples_per_shard * (worker_id + 1)

        while True:
            idx = rng.integers(low=start, high=end)
            x, y = dataset[idx]
            if len(x) < self.boundaries[0] or len(x) >= self.boundaries[-1]:
                continue
            yield x, y


@dataclass
class BucketDatasetConfig:
    _target_: str = "asr.data.BucketDataset"
    datasets: list[Any] = MISSING  # list of inner dataset configs
    weights: list[float] = MISSING
    boundaries: list[int] = MISSING
    batch_frame_budget: int = MISSING
    max_bucket_count: int = MISSING
    seed: int = 0
    feature_dim: int = MISSING
