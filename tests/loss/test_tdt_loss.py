# Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
import pytest
import torch

from asr.loss.tdt import TDTLoss, tdt_loss
from tests.loss.tdt_reference import tdt_loss as tdt_loss_ref

ATOL = 1e-4
RTOL = 1e-4


def _durations(max_duration):
    return list(range(max_duration + 1))


def _grad(t: torch.Tensor) -> torch.Tensor:
    """Tensor.grad narrowed from ``Tensor | None`` after a backward pass."""
    assert t.grad is not None
    return t.grad


def _make_inputs(B, T, U, V, d_model, n_dur, in_lens=None, tgt_lens=None, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device=device).manual_seed(seed)
    encoder = torch.randn(B, T, d_model, generator=g, device=device, dtype=dtype)
    decoder = torch.randn(B, U, d_model, generator=g, device=device, dtype=dtype)
    joint_W = torch.randn(V, d_model, generator=g, device=device, dtype=dtype)
    joint_W_dur = torch.randn(n_dur, d_model, generator=g, device=device, dtype=dtype)
    # labels live in [1, V); index 0 is reserved for blank. targets has U-1 columns.
    targets = torch.randint(1, V, (B, max(U - 1, 0)), generator=g, device=device, dtype=torch.int32)
    if in_lens is None:
        in_lens = torch.full((B,), T, device=device, dtype=torch.int32)
    else:
        in_lens = torch.as_tensor(in_lens, device=device, dtype=torch.int32)
    if tgt_lens is None:
        tgt_lens = torch.full((B,), U - 1, device=device, dtype=torch.int32)
    else:
        tgt_lens = torch.as_tensor(tgt_lens, device=device, dtype=torch.int32)
    return encoder, decoder, joint_W, joint_W_dur, targets, in_lens, tgt_lens


# ------------------------------------------------------------------ analytic backward (gradcheck)


@pytest.mark.parametrize(
    "B,T,U,V,d_model,max_duration,in_lens,tgt_lens,sigma",
    [
        pytest.param(1, 4, 3, 5, 6, 2, None, None, 0.0, id="single"),
        pytest.param(3, 5, 4, 6, 4, 3, None, None, 0.0, id="batch_fixed"),
        pytest.param(3, 6, 4, 5, 4, 2, [6, 5, 4], [3, 2, 1], 0.0, id="variable_lens"),
        pytest.param(2, 5, 3, 4, 4, 2, [5, 4], [2, 0], 0.0, id="empty_target_in_batch"),
        pytest.param(3, 10, 4, 5, 4, 4, [10, 8, 6], [3, 2, 1], 0.0, id="max_duration_4"),
        pytest.param(3, 5, 4, 6, 4, 3, None, None, 0.1, id="sigma_0p1"),
        pytest.param(3, 6, 4, 5, 4, 2, [6, 5, 4], [3, 2, 1], 0.3, id="sigma_0p3_varlen"),
    ],
)
def test_gradcheck(B, T, U, V, d_model, max_duration, in_lens, tgt_lens, sigma):
    """Analytic backward matches numerical gradient for encoder, decoder and both joint heads."""
    durations = _durations(max_duration)
    enc, dec, W, W_dur, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, len(durations), in_lens=in_lens, tgt_lens=tgt_lens
    )
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)
    W_dur.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda e, d, w, wd: tdt_loss_ref(e, d, w, wd, targets, in_lens_, tgt_lens_, durations, 0, sigma),
        (enc, dec, W, W_dur),
        atol=1e-5,
        rtol=1e-4,
        nondet_tol=0.0,
    )


# ------------------------------------------------------------------ fused CUDA loss


_CUDA_SHAPES = [
    pytest.param(1, 4, 3, 5, 6, 4, None, None, id="single"),
    pytest.param(4, 12, 6, 8, 8, 4, None, None, id="batch_fixed"),
    pytest.param(3, 15, 6, 10, 8, 4, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    pytest.param(2, 5, 3, 4, 4, 2, [5, 4], [2, 0], id="empty_target_in_batch"),
    pytest.param(4, 12, 6, 8, 8, 2, None, None, id="max_duration_2"),
    pytest.param(3, 12, 5, 6, 6, 6, [12, 10, 8], [4, 3, 2], id="max_duration_6"),
]


def _ref_double(enc, dec, W, W_dur, targets, in_lens, tgt_lens, durations, sigma=0.0):
    """Reference loss in float64 — the exact oracle for the float32 kernel."""
    return tdt_loss_ref(
        enc.double(), dec.double(), W.double(), W_dur.double(), targets, in_lens, tgt_lens, durations, 0, sigma
    )


@pytest.mark.cuda
@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.2], ids=["sigma_0", "sigma_paper", "sigma_0p2"])
@pytest.mark.parametrize("B,T,U,V,d_model,max_duration,in_lens,tgt_lens", _CUDA_SHAPES)
def test_cuda_forward_matches_reference(B, T, U, V, d_model, max_duration, in_lens, tgt_lens, sigma):
    durations = _durations(max_duration)
    enc, dec, W, W_dur, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, len(durations), in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )
    ref = _ref_double(enc, dec, W, W_dur, targets, in_lens_, tgt_lens_, durations, sigma=sigma)
    out = tdt_loss(enc, dec, W, W_dur, targets, in_lens_, tgt_lens_, durations, 0, sigma=sigma)
    torch.testing.assert_close(out, ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.2], ids=["sigma_0", "sigma_paper", "sigma_0p2"])
@pytest.mark.parametrize("B,T,U,V,d_model,max_duration,in_lens,tgt_lens", _CUDA_SHAPES)
def test_cuda_backward_matches_reference(B, T, U, V, d_model, max_duration, in_lens, tgt_lens, sigma):
    """All four gradients match the float64 reference path."""
    durations = _durations(max_duration)
    enc, dec, W, W_dur, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, len(durations), in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )

    e_r, d_r, w_r, wd_r = (t.detach().double().requires_grad_() for t in (enc, dec, W, W_dur))
    tdt_loss_ref(e_r, d_r, w_r, wd_r, targets, in_lens_, tgt_lens_, durations, 0, sigma).sum().backward()

    e_c, d_c, w_c, wd_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W, W_dur))
    tdt_loss(e_c, d_c, w_c, wd_c, targets, in_lens_, tgt_lens_, durations, 0, sigma=sigma).sum().backward()

    for c, r in zip((e_c, d_c, w_c, wd_c), (e_r, d_r, w_r, wd_r), strict=True):
        torch.testing.assert_close(_grad(c), _grad(r).float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_cuda_weighted_upstream():
    """Per-utterance upstream weights (grad_loss != 1) match the reference."""
    B, T, U, V, d_model = 3, 8, 5, 6, 6
    durations = _durations(3)
    enc, dec, W, W_dur, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, len(durations), seed=7, device="cuda", dtype=torch.float32
    )
    weights = torch.tensor([0.5, 2.0, 1.5], device="cuda", dtype=torch.float32)

    e_r, d_r, w_r, wd_r = (t.detach().double().requires_grad_() for t in (enc, dec, W, W_dur))
    (tdt_loss_ref(e_r, d_r, w_r, wd_r, targets, in_lens, tgt_lens, durations, 0) * weights.double()).sum().backward()

    e_c, d_c, w_c, wd_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W, W_dur))
    (tdt_loss(e_c, d_c, w_c, wd_c, targets, in_lens, tgt_lens, durations, 0) * weights).sum().backward()

    for c, r in zip((e_c, d_c, w_c, wd_c), (e_r, d_r, w_r, wd_r), strict=True):
        torch.testing.assert_close(_grad(c), _grad(r).float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_cuda_tf32_approximately_matches_reference():
    """The tf32 opt-in uses tensor cores: looser precision, same answer to ~1%."""
    B, T, U, V, d_model = 4, 20, 8, 32, 64
    durations = _durations(4)
    enc, dec, W, W_dur, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, len(durations), seed=5, device="cuda", dtype=torch.float32
    )
    ref = _ref_double(enc, dec, W, W_dur, targets, in_lens, tgt_lens, durations)

    e_r, d_r, w_r, wd_r = (t.detach().double().requires_grad_() for t in (enc, dec, W, W_dur))
    tdt_loss_ref(e_r, d_r, w_r, wd_r, targets, in_lens, tgt_lens, durations, 0).sum().backward()

    e_c, d_c, w_c, wd_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W, W_dur))
    out = tdt_loss(e_c, d_c, w_c, wd_c, targets, in_lens, tgt_lens, durations, 0, tf32=True)
    out.sum().backward()

    torch.testing.assert_close(out, ref.float(), atol=5e-2, rtol=5e-2)
    for got, want in zip((e_c, d_c, w_c, wd_c), (e_r, d_r, w_r, wd_r), strict=True):
        rel = (_grad(got).double() - _grad(want)).norm() / _grad(want).norm()
        assert torch.isfinite(_grad(got)).all() and rel < 0.05, f"tf32 grad rel-err {rel:.3e}"


@pytest.mark.cuda
def test_cuda_padding_frames_and_labels_get_zero_grad():
    """Gradients vanish on padded encoder frames and padded decoder label slots."""
    B, T, U, V, d_model = 2, 6, 4, 5, 4
    durations = _durations(2)
    enc, dec, W, W_dur, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, len(durations), in_lens=[6, 4], tgt_lens=[3, 1], seed=3, device="cuda", dtype=torch.float32
    )
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)
    W_dur.requires_grad_(True)
    tdt_loss(enc, dec, W, W_dur, targets, in_lens, tgt_lens, durations, 0).sum().backward()

    # utterance 1 uses 4 frames and 1 label -> the rest are padding.
    assert torch.count_nonzero(_grad(enc)[1, 4:]) == 0
    assert torch.count_nonzero(_grad(dec)[1, 2:]) == 0


@pytest.mark.cuda
def test_tdt_module_normalizes_by_target_length():
    """TDTLoss divides the per-utterance loss by tgt_lens (clamped at 1)."""
    B, T, U, V, d_model, max_duration = 3, 15, 6, 10, 8, 4
    durations = _durations(max_duration)
    enc, dec, W, W_dur, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, len(durations), in_lens=[15, 12, 9], tgt_lens=[5, 3, 1], device="cuda", dtype=torch.float32
    )
    raw = tdt_loss(enc, dec, W, W_dur, targets, in_lens, tgt_lens, durations, 0)
    out = TDTLoss(max_duration)(enc, dec, W, W_dur, targets, in_lens, tgt_lens)
    torch.testing.assert_close(out, raw / tgt_lens.clamp_min(1), atol=ATOL, rtol=RTOL)
