import pytest
import torch
import torch.nn.functional as F

from asr.loss.ctc_loss import CTCLoss, CTCLossFn

ATOL = 1e-4
RTOL = 1e-4


pytestmark = pytest.mark.cuda


def _make_inputs(B, T, V, S_max, in_lens=None, tgt_lens=None, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g, device=device, dtype=torch.float32)
    log_probs = logits.log_softmax(dim=-1)
    # targets in [1, V) as 0 reserved for blank
    targets = torch.randint(1, V, (B, S_max), generator=g, device=device, dtype=torch.int32)
    if in_lens is None:
        in_lens = torch.full((B,), T, device=device, dtype=torch.int32)
    else:
        in_lens = torch.as_tensor(in_lens, device=device, dtype=torch.int32)
    if tgt_lens is None:
        tgt_lens = torch.full((B,), S_max, device=device, dtype=torch.int32)
    else:
        tgt_lens = torch.as_tensor(tgt_lens, device=device, dtype=torch.int32)
    return log_probs, targets, in_lens, tgt_lens


def _torch_ref_loss(log_probs, targets, in_lens, tgt_lens, reduction="none"):
    return F.ctc_loss(
        log_probs.transpose(0, 1).contiguous(),  # (T, B, V)
        targets.long(),
        in_lens.long(),
        tgt_lens.long(),
        blank=0,
        reduction=reduction,
        zero_infinity=False,
    )


# forward


@pytest.mark.parametrize(
    "B,T,V,S_max,in_lens,tgt_lens",
    [
        pytest.param(1, 4, 3, 1, None, None, id="trivial"),
        pytest.param(1, 5, 4, 2, None, None, id="tight_boundary_T_eq_2S+1"),
        pytest.param(4, 20, 8, 6, None, None, id="small_batch_fixed"),
        pytest.param(3, 30, 10, 8, [30, 25, 20], [8, 5, 3], id="variable_lens_padded"),
        pytest.param(8, 50, 29, 20, None, None, id="larger_fixed"),
    ],
)
def test_forward_matches_pytorch(B, T, V, S_max, in_lens, tgt_lens):
    log_probs, targets, in_lens_, tgt_lens_ = _make_inputs(B, T, V, S_max, in_lens=in_lens, tgt_lens=tgt_lens)
    ref = _torch_ref_loss(log_probs, targets, in_lens_, tgt_lens_)
    out = CTCLossFn.apply(log_probs, targets, in_lens_, tgt_lens_)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_forward_repeated_targets():
    """Adjacent repeated labels must be separated by blank."""
    device = "cuda"
    log_probs = torch.randn(1, 6, 5, device=device).log_softmax(-1)
    targets = torch.tensor([[1, 1, 2]], device=device, dtype=torch.int32)
    in_lens = torch.tensor([6], device=device, dtype=torch.int32)
    tgt_lens = torch.tensor([3], device=device, dtype=torch.int32)

    ref = _torch_ref_loss(log_probs, targets, in_lens, tgt_lens)
    out = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_forward_random_fuzz_varlen(seed):
    B, T, V, S_max = 4, 40, 12, 10
    device = "cuda"
    log_probs, targets, _, _ = _make_inputs(B, T, V, S_max, seed=seed)
    g = torch.Generator(device=device).manual_seed(seed + 100)
    tgt_lens = torch.randint(1, S_max + 1, (B,), generator=g, device=device, dtype=torch.int32)
    min_T = int((2 * tgt_lens + 1).max().item())
    in_lens = torch.randint(min_T, T + 1, (B,), generator=g, device=device, dtype=torch.int32)

    ref = _torch_ref_loss(log_probs, targets, in_lens, tgt_lens)
    out = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


# backward


@pytest.mark.parametrize(
    "B,T,V,S_max",
    [
        pytest.param(1, 4, 3, 1, id="trivial"),
        pytest.param(4, 20, 8, 6, id="small"),
        pytest.param(8, 50, 29, 20, id="larger"),
    ],
)
def test_backward_matches_pytorch(B, T, V, S_max):
    """Compare d loss / d log_probs between the kernel and PyTorch's autograd."""
    log_probs_base, targets, in_lens, tgt_lens = _make_inputs(B, T, V, S_max)

    lp_ref = log_probs_base.detach().clone().requires_grad_()
    F.ctc_loss(
        lp_ref.transpose(0, 1).contiguous(),
        targets.long(),
        in_lens.long(),
        tgt_lens.long(),
        blank=0,
        reduction="none",
        zero_infinity=False,
    ).sum().backward()

    lp_cust = log_probs_base.detach().clone().requires_grad_()
    CTCLossFn.apply(lp_cust, targets, in_lens, tgt_lens).sum().backward()

    torch.testing.assert_close(lp_cust.grad, lp_ref.grad, atol=ATOL, rtol=RTOL)


def test_backward_weighted_grad_loss():
    """Upstream grad scaling should pass through correctly (grad_loss != 1)."""
    B, T, V, S_max = 3, 15, 6, 4
    log_probs_base, targets, in_lens, tgt_lens = _make_inputs(B, T, V, S_max, seed=42)
    weights = torch.tensor([0.5, 2.0, 1.5], device="cuda", dtype=torch.float32)

    lp_ref = log_probs_base.detach().clone().requires_grad_()
    (_torch_ref_loss(lp_ref, targets, in_lens, tgt_lens) * weights).sum().backward()

    lp_cust = log_probs_base.detach().clone().requires_grad_()
    (CTCLossFn.apply(lp_cust, targets, in_lens, tgt_lens) * weights).sum().backward()

    torch.testing.assert_close(lp_cust.grad, lp_ref.grad, atol=ATOL, rtol=RTOL)


# module wrapper


def test_module_reduction_none_matches_fn():
    log_probs, targets, in_lens, tgt_lens = _make_inputs(4, 20, 8, 6)
    module = CTCLoss(reduction="none")
    out = module(log_probs, targets, in_lens, tgt_lens)
    ref = _torch_ref_loss(log_probs, targets, in_lens, tgt_lens, reduction="none")
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


def test_module_reduction_mean_normalises_by_tgt_lens():
    """CTCLoss(reduction='mean') divides per-sample loss by tgt_lens before averaging."""
    log_probs, targets, in_lens, tgt_lens = _make_inputs(4, 20, 8, 6)
    module = CTCLoss(reduction="mean")
    out = module(log_probs, targets, in_lens, tgt_lens)

    per_sample = _torch_ref_loss(log_probs, targets, in_lens, tgt_lens, reduction="none")
    expected = (per_sample / tgt_lens).mean()
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


# edge cases


def test_empty_target_loss_is_all_blank_path():
    """For tgt_lens[b]==0 only valid path is all-blank so test loss = -sum_{t < in_lens[b]} log_probs[b, t, 0]."""
    device = "cuda"
    B, T, V, S_max = 2, 10, 5, 3
    log_probs = torch.randn(B, T, V, device=device).log_softmax(-1)
    targets = torch.randint(1, V, (B, S_max), device=device, dtype=torch.int32)
    in_lens = torch.tensor([T, 7], device=device, dtype=torch.int32)
    tgt_lens = torch.tensor([S_max, 0], device=device, dtype=torch.int32)

    expected_empty = -log_probs[1, : int(in_lens[1]), 0].sum()
    ref = _torch_ref_loss(log_probs, targets, in_lens, tgt_lens)
    torch.testing.assert_close(ref[1], expected_empty, atol=ATOL, rtol=RTOL)

    out = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
    torch.testing.assert_close(out[1], expected_empty, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
