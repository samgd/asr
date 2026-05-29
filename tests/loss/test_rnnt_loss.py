import pytest
import torch

from asr.loss.rnnt import RNNTLoss, rnnt_loss
from tests.loss.rnnt_reference import RNNTLossFn as RNNTLossRefFn
from tests.loss.rnnt_reference import rnnt_loss as rnnt_loss_ref

ATOL = 1e-4
RTOL = 1e-4


def _make_inputs(B, T, U, V, d_model, in_lens=None, tgt_lens=None, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device=device).manual_seed(seed)
    encoder = torch.randn(B, T, d_model, generator=g, device=device, dtype=dtype)
    decoder = torch.randn(B, U, d_model, generator=g, device=device, dtype=dtype)
    joint_W = torch.randn(V, d_model, generator=g, device=device, dtype=dtype)
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
    return encoder, decoder, joint_W, targets, in_lens, tgt_lens


def _logits(encoder, decoder, joint_W):
    """The joint network exactly as the loss computes it: logits = tanh(enc + dec) @ W.T."""
    return torch.tanh(encoder[:, :, None, :] + decoder[:, None, :, :]) @ joint_W.T


# ------------------------------------------------------------------ analytic backward (gradcheck)


@pytest.mark.parametrize(
    "B,T,U,V,d_model,in_lens,tgt_lens",
    [
        pytest.param(1, 4, 3, 5, 6, None, None, id="single"),
        pytest.param(3, 5, 4, 6, 4, None, None, id="batch_fixed"),
        pytest.param(3, 6, 4, 5, 4, [6, 5, 4], [3, 2, 1], id="variable_lens"),
        pytest.param(2, 5, 3, 4, 4, [5, 4], [2, 0], id="empty_target_in_batch"),
        pytest.param(2, 4, 1, 4, 3, None, None, id="all_empty_targets"),
    ],
)
def test_gradcheck(B, T, U, V, d_model, in_lens, tgt_lens):
    """Analytic backward matches numerical gradient for encoder, decoder and joint_W."""
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens)
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda e, d, w: RNNTLossRefFn.apply(e, d, w, in_lens_, tgt_lens_, targets, 0),
        (enc, dec, W),
        atol=1e-5,
        rtol=1e-4,
        nondet_tol=0.0,
    )


def test_gradcheck_weighted_upstream():
    """Per-utterance upstream weights (grad_loss != 1) propagate correctly."""
    B, T, U, V, d_model = 3, 5, 4, 5, 4
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(B, T, U, V, d_model, seed=7)
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)
    weights = torch.tensor([0.5, 2.0, 1.5], dtype=torch.float64)

    assert torch.autograd.gradcheck(
        lambda e, d, w: RNNTLossRefFn.apply(e, d, w, in_lens, tgt_lens, targets, 0) * weights,
        (enc, dec, W),
        atol=1e-5,
        rtol=1e-4,
    )


def test_padding_frames_and_labels_get_zero_grad():
    """Gradients vanish on padded encoder frames and padded decoder label slots."""
    B, T, U, V, d_model = 2, 6, 4, 5, 4
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, in_lens=[6, 4], tgt_lens=[3, 1], seed=3
    )
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)
    rnnt_loss_ref(enc, dec, W, targets, in_lens, tgt_lens, 0).sum().backward()

    # utterance 1 uses 4 frames and 1 label (U index range 0..1) -> the rest are padding.
    assert torch.count_nonzero(enc.grad[1, 4:]) == 0
    assert torch.count_nonzero(dec.grad[1, 2:]) == 0


def test_empty_target_is_all_blank_path():
    """With tgt_lens[b]==0 the only alignment is blanks across every frame at u=0.

    Checked in closed form rather than against torchaudio, whose rnnt_loss returns a
    (negative, invalid) value for empty targets.
    """
    B, T, U, V, d_model = 2, 8, 3, 5, 4
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, in_lens=[8, 6], tgt_lens=[2, 0], seed=11
    )
    logp = _logits(enc, dec, W).log_softmax(dim=-1)
    expected = -logp[1, : int(in_lens[1]), 0, 0].sum()  # blank at index 0, u=0

    out = rnnt_loss_ref(enc, dec, W, targets, in_lens, tgt_lens, 0)
    torch.testing.assert_close(out[1], expected, atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------ cross-check vs torchaudio


def _torchaudio_loss(logits, targets, in_lens, tgt_lens, blank=0):
    from torchaudio.functional import rnnt_loss as ta_rnnt_loss

    return ta_rnnt_loss(
        logits.float(),
        targets.int(),
        in_lens.int(),
        tgt_lens.int(),
        blank=blank,
        reduction="none",
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "B,T,U,V,d_model,in_lens,tgt_lens",
    [
        pytest.param(1, 4, 3, 5, 6, None, None, id="single"),
        pytest.param(4, 12, 6, 8, 8, None, None, id="batch_fixed"),
        pytest.param(3, 15, 6, 10, 8, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    ],
)
def test_forward_matches_torchaudio(B, T, U, V, d_model, in_lens, tgt_lens):
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )
    ref = _torchaudio_loss(_logits(enc, dec, W), targets, in_lens_, tgt_lens_)
    out = rnnt_loss_ref(enc, dec, W, targets, in_lens_, tgt_lens_, 0)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "B,T,U,V,d_model,in_lens,tgt_lens",
    [
        pytest.param(4, 12, 6, 8, 8, None, None, id="batch_fixed"),
        pytest.param(3, 15, 6, 10, 8, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    ],
)
def test_backward_matches_torchaudio(B, T, U, V, d_model, in_lens, tgt_lens):
    """All three gradients match autograd through the same joint + torchaudio's lattice."""
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )

    e_ref, d_ref, w_ref = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    _torchaudio_loss(_logits(e_ref, d_ref, w_ref), targets, in_lens_, tgt_lens_).sum().backward()

    e_cust, d_cust, w_cust = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    rnnt_loss_ref(e_cust, d_cust, w_cust, targets, in_lens_, tgt_lens_, 0).sum().backward()

    torch.testing.assert_close(e_cust.grad, e_ref.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(d_cust.grad, d_ref.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w_cust.grad, w_ref.grad, atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------ fused CUDA loss


_CUDA_SHAPES = [
    pytest.param(1, 4, 3, 5, 6, None, None, id="single"),
    pytest.param(4, 12, 6, 8, 8, None, None, id="batch_fixed"),
    pytest.param(3, 15, 6, 10, 8, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    pytest.param(2, 5, 3, 4, 4, [5, 4], [2, 0], id="empty_target_in_batch"),
]


def _ref_double(enc, dec, W, targets, in_lens, tgt_lens):
    """Reference loss in float64 — the exact oracle for the float32 kernel."""
    return rnnt_loss_ref(enc.double(), dec.double(), W.double(), targets, in_lens, tgt_lens, 0)


@pytest.mark.cuda
@pytest.mark.parametrize("B,T,U,V,d_model,in_lens,tgt_lens", _CUDA_SHAPES)
def test_cuda_forward_matches_reference(B, T, U, V, d_model, in_lens, tgt_lens):
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )
    ref = _ref_double(enc, dec, W, targets, in_lens_, tgt_lens_)
    out = rnnt_loss(enc, dec, W, targets, in_lens_, tgt_lens_, 0)
    torch.testing.assert_close(out, ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
@pytest.mark.parametrize("B,T,U,V,d_model,in_lens,tgt_lens", _CUDA_SHAPES)
def test_cuda_backward_matches_reference(B, T, U, V, d_model, in_lens, tgt_lens):
    """All three gradients match the float64 reference path."""
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )

    e_ref, d_ref, w_ref = (t.detach().double().requires_grad_() for t in (enc, dec, W))
    rnnt_loss_ref(e_ref, d_ref, w_ref, targets, in_lens_, tgt_lens_, 0).sum().backward()

    e_c, d_c, w_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    rnnt_loss(e_c, d_c, w_c, targets, in_lens_, tgt_lens_, 0).sum().backward()

    torch.testing.assert_close(e_c.grad, e_ref.grad.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(d_c.grad, d_ref.grad.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w_c.grad, w_ref.grad.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "B,T,U,V,d_model,in_lens,tgt_lens",
    [
        pytest.param(1, 4, 3, 5, 6, None, None, id="single"),
        pytest.param(4, 12, 6, 8, 8, None, None, id="batch_fixed"),
        pytest.param(3, 15, 6, 10, 8, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    ],
)
def test_cuda_forward_matches_torchaudio(B, T, U, V, d_model, in_lens, tgt_lens):
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )
    ref = _torchaudio_loss(_logits(enc, dec, W), targets, in_lens_, tgt_lens_)
    out = rnnt_loss(enc, dec, W, targets, in_lens_, tgt_lens_, 0)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "B,T,U,V,d_model,in_lens,tgt_lens",
    [
        pytest.param(4, 12, 6, 8, 8, None, None, id="batch_fixed"),
        pytest.param(3, 15, 6, 10, 8, [15, 12, 9], [5, 3, 1], id="variable_lens"),
    ],
)
def test_cuda_backward_matches_torchaudio(B, T, U, V, d_model, in_lens, tgt_lens):
    enc, dec, W, targets, in_lens_, tgt_lens_ = _make_inputs(
        B, T, U, V, d_model, in_lens=in_lens, tgt_lens=tgt_lens, device="cuda", dtype=torch.float32
    )

    e_ref, d_ref, w_ref = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    _torchaudio_loss(_logits(e_ref, d_ref, w_ref), targets, in_lens_, tgt_lens_).sum().backward()

    e_c, d_c, w_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    rnnt_loss(e_c, d_c, w_c, targets, in_lens_, tgt_lens_, 0).sum().backward()

    torch.testing.assert_close(e_c.grad, e_ref.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(d_c.grad, d_ref.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w_c.grad, w_ref.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_cuda_weighted_upstream():
    """Per-utterance upstream weights (grad_loss != 1) match the reference."""
    B, T, U, V, d_model = 3, 8, 5, 6, 6
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, seed=7, device="cuda", dtype=torch.float32
    )
    weights = torch.tensor([0.5, 2.0, 1.5], device="cuda", dtype=torch.float32)

    e_ref, d_ref, w_ref = (t.detach().double().requires_grad_() for t in (enc, dec, W))
    (rnnt_loss_ref(e_ref, d_ref, w_ref, targets, in_lens, tgt_lens, 0) * weights.double()).sum().backward()

    e_c, d_c, w_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    (rnnt_loss(e_c, d_c, w_c, targets, in_lens, tgt_lens, 0) * weights).sum().backward()

    torch.testing.assert_close(e_c.grad, e_ref.grad.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(d_c.grad, d_ref.grad.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(w_c.grad, w_ref.grad.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_cuda_empty_target_is_all_blank_path():
    """tgt_lens[b]==0: forward equals the closed-form all-blank loss (torchaudio is wrong here)."""
    B, T, U, V, d_model = 2, 8, 3, 5, 4
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, in_lens=[8, 6], tgt_lens=[2, 0], seed=11, device="cuda", dtype=torch.float32
    )
    logp = _logits(enc, dec, W).log_softmax(dim=-1)
    expected = -logp[1, : int(in_lens[1]), 0, 0].sum()

    out = rnnt_loss(enc, dec, W, targets, in_lens, tgt_lens, 0)
    torch.testing.assert_close(out[1], expected, atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_cuda_tf32_approximately_matches_reference():
    """The tf32 opt-in uses tensor cores: looser precision, same answer to ~1%."""
    B, T, U, V, d_model = 4, 20, 8, 32, 64
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, seed=5, device="cuda", dtype=torch.float32
    )
    ref = _ref_double(enc, dec, W, targets, in_lens, tgt_lens)

    e_ref, d_ref, w_ref = (t.detach().double().requires_grad_() for t in (enc, dec, W))
    rnnt_loss_ref(e_ref, d_ref, w_ref, targets, in_lens, tgt_lens, 0).sum().backward()

    e_c, d_c, w_c = (t.detach().clone().requires_grad_() for t in (enc, dec, W))
    out = rnnt_loss(e_c, d_c, w_c, targets, in_lens, tgt_lens, 0, tf32=True)
    out.sum().backward()

    # Loss is well-conditioned; grads checked on relative norm to stay magnitude-robust.
    torch.testing.assert_close(out, ref.float(), atol=5e-2, rtol=5e-2)
    for got, want in ((e_c.grad, e_ref.grad), (d_c.grad, d_ref.grad), (w_c.grad, w_ref.grad)):
        rel = (got.double() - want).norm() / want.norm()
        assert torch.isfinite(got).all() and rel < 0.05, f"tf32 grad rel-err {rel:.3e}"


@pytest.mark.cuda
def test_cuda_padding_frames_and_labels_get_zero_grad():
    """Gradients vanish on padded encoder frames and padded decoder label slots."""
    B, T, U, V, d_model = 2, 6, 4, 5, 4
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, in_lens=[6, 4], tgt_lens=[3, 1], seed=3, device="cuda", dtype=torch.float32
    )
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    W.requires_grad_(True)
    rnnt_loss(enc, dec, W, targets, in_lens, tgt_lens, 0).sum().backward()

    assert torch.count_nonzero(enc.grad[1, 4:]) == 0
    assert torch.count_nonzero(dec.grad[1, 2:]) == 0


@pytest.mark.cuda
def test_rnnt_module_normalizes_by_target_length():
    """RNNTLoss divides the per-utterance loss by tgt_lens (clamped at 1)."""
    B, T, U, V, d_model = 3, 15, 6, 10, 8
    enc, dec, W, targets, in_lens, tgt_lens = _make_inputs(
        B, T, U, V, d_model, in_lens=[15, 12, 9], tgt_lens=[5, 3, 1], device="cuda", dtype=torch.float32
    )
    raw = rnnt_loss(enc, dec, W, targets, in_lens, tgt_lens, 0)
    out = RNNTLoss()(enc, dec, W, targets, in_lens, tgt_lens)
    torch.testing.assert_close(out, raw / tgt_lens.clamp_min(1), atol=ATOL, rtol=RTOL)
