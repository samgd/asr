"""Benchmark the fused CUDA RNN-T loss against torchaudio and the pure-torch reference.

Run: ``uv run python -m tests.loss.bench_rnnt``

Compares forward+backward wall-time and peak memory for three end-to-end paths that
all start from encoder/decoder/joint_W leaves and produce their gradients:

  - cuda:       our fused kernel (rnnt_loss); never materializes (B,T,U,V).
  - cuda-tf32:  same kernel with tf32=True (tensor-core GEMMs; ~1e-2 loss error).
  - torchaudio: equivalent torch joint network (tanh + matmul) feeding
                torchaudio.functional.rnnt_loss; materializes (B,T,U,V).
  - reference:  the pure-torch lattice (rnnt_loss_ref); materializes (B,T,U,V).
"""

import torch

from asr.loss.rnnt import rnnt_loss
from tests.loss.rnnt_reference import rnnt_loss as rnnt_loss_ref

SHAPES = [
    dict(B=16, T=200, U=40, V=256, d=256),
    dict(B=8, T=400, U=60, V=512, d=256),
]


def _inputs(B, T, U, V, d, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    enc = torch.randn(B, T, d, generator=g, device=device)
    dec = torch.randn(B, U, d, generator=g, device=device)
    W = torch.randn(V, d, generator=g, device=device) / (d**0.5)
    targets = torch.randint(1, V, (B, U - 1), generator=g, device=device, dtype=torch.int32)
    in_lens = torch.full((B,), T, device=device, dtype=torch.int32)
    tgt_lens = torch.full((B,), U - 1, device=device, dtype=torch.int32)
    return enc, dec, W, targets, in_lens, tgt_lens


def _cuda_step(enc, dec, W, targets, in_lens, tgt_lens):
    rnnt_loss(enc, dec, W, targets, in_lens, tgt_lens, 0).sum().backward()


def _cuda_tf32_step(enc, dec, W, targets, in_lens, tgt_lens):
    rnnt_loss(enc, dec, W, targets, in_lens, tgt_lens, 0, tf32=True).sum().backward()


def _reference_step(enc, dec, W, targets, in_lens, tgt_lens):
    rnnt_loss_ref(enc, dec, W, targets, in_lens, tgt_lens, 0).sum().backward()


def _torchaudio_step(enc, dec, W, targets, in_lens, tgt_lens):
    from torchaudio.functional import rnnt_loss as ta_rnnt_loss

    logits = torch.tanh(enc[:, :, None, :] + dec[:, None, :, :]) @ W.T
    ta_rnnt_loss(logits, targets, in_lens, tgt_lens, blank=0, reduction="sum").backward()


def _bench(step, enc, dec, W, targets, in_lens, tgt_lens, warmup=3, iters=10):
    leaves = [t.detach().requires_grad_() for t in (enc, dec, W)]

    for _ in range(warmup):
        for leaf in leaves:
            leaf.grad = None
        step(*leaves, targets, in_lens, tgt_lens)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        for leaf in leaves:
            leaf.grad = None
        step(*leaves, targets, in_lens, tgt_lens)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak_mb


def main():
    assert torch.cuda.is_available(), "benchmark requires CUDA"
    methods = {
        "cuda": _cuda_step,
        "cuda-tf32": _cuda_tf32_step,
        "torchaudio": _torchaudio_step,
        "reference": _reference_step,
    }

    for shape in SHAPES:
        print(f"\nshape: {shape}")
        print(f"  {'method':<12}{'fwd+bwd (ms)':>16}{'peak mem (MB)':>16}")
        inp = _inputs(**shape)
        for name, step in methods.items():
            try:
                ms, mem = _bench(step, *inp)
                print(f"  {name:<12}{ms:>16.3f}{mem:>16.1f}")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                torch.cuda.empty_cache()
                print(f"  {name:<12}{'FAILED: ' + str(e)[:48]:>32}")


if __name__ == "__main__":
    main()
