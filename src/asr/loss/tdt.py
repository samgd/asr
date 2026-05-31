from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Integer

_so = next(Path(__file__).parent.glob("_C*.so"))
torch.ops.load_library(str(_so))


class TDTLossFn(torch.autograd.Function):
    """Token-and-Duration (TDT) loss."""

    @staticmethod
    def forward(
        ctx,
        encoder,
        decoder,
        joint_W,
        joint_W_dur,
        in_lens,
        tgt_lens,
        targets,
        durations,
        max_duration,
        has_zero,
        blank_idx,
        sigma,
        tf32,
    ):
        targets = targets.int()
        in_lens = in_lens.int()
        tgt_lens = tgt_lens.int()
        durations = durations.int()
        loss, alpha, log_P_blank, log_P_y, log_P_dur, log_prob = torch.ops.asr.tdt_forward(
            encoder.contiguous(),
            decoder.contiguous(),
            joint_W.contiguous(),
            joint_W_dur.contiguous(),
            targets,
            in_lens,
            tgt_lens,
            durations,
            max_duration,
            has_zero,
            blank_idx,
            sigma,
            tf32,
        )
        ctx.save_for_backward(
            encoder,
            decoder,
            joint_W,
            joint_W_dur,
            targets,
            in_lens,
            tgt_lens,
            durations,
            alpha,
            log_P_blank,
            log_P_y,
            log_P_dur,
            log_prob,
        )
        ctx.max_duration = max_duration
        ctx.has_zero = has_zero
        ctx.blank_idx = blank_idx
        ctx.tf32 = tf32
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs
        (
            enc,
            dec,
            W,
            W_dur,
            targets,
            in_lens,
            tgt_lens,
            durations,
            alpha,
            log_P_blank,
            log_P_y,
            log_P_dur,
            log_prob,
        ) = ctx.saved_tensors
        grad_encoder, grad_decoder, grad_joint_W, grad_joint_W_dur = torch.ops.asr.tdt_backward(
            enc.contiguous(),
            dec.contiguous(),
            W.contiguous(),
            W_dur.contiguous(),
            targets,
            in_lens,
            tgt_lens,
            durations,
            ctx.max_duration,
            ctx.has_zero,
            ctx.blank_idx,
            alpha,
            log_P_blank,
            log_P_y,
            log_P_dur,
            log_prob,
            grad_loss.contiguous(),
            ctx.tf32,
        )
        nones = (None,) * 9
        return grad_encoder, grad_decoder, grad_joint_W, grad_joint_W_dur, *nones


def tdt_loss(
    encoder: Float[torch.Tensor, "batch time d_model"],
    decoder: Float[torch.Tensor, "batch u d_model"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    joint_W_dur: Float[torch.Tensor, "n_dur d_model"],
    targets: Integer[torch.Tensor, "batch seq"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    durations: Sequence[int],
    blank_idx: int = 0,
    sigma: float = 0.0,
    tf32: bool = False,
) -> Float[torch.Tensor, "batch"]:
    """Token-and-Duration (TDT) loss."""
    durations = list(durations)
    if len(durations) != joint_W_dur.shape[0]:
        raise ValueError(f"len(durations)={len(durations)} must equal joint_W_dur rows={joint_W_dur.shape[0]}")
    if min(durations) < 0:
        raise ValueError(f"durations must be non-negative, got {durations}")
    if 1 not in durations:
        raise ValueError(f"durations must contain 1 so every frame count is reachable, got {durations}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    durations_t = torch.as_tensor(durations, dtype=torch.int32, device=encoder.device)
    max_duration = int(max(durations))
    has_zero = int(0 in durations)
    return TDTLossFn.apply(
        encoder,
        decoder,
        joint_W,
        joint_W_dur,
        in_lens,
        tgt_lens,
        targets,
        durations_t,
        max_duration,
        has_zero,
        blank_idx,
        float(sigma),
        int(tf32),
    )


class TDTLoss(torch.nn.Module):
    """Token-and-Duration (TDT) loss, normalized per utterance by the target length.

    Args:
        max_duration: Largest frame skip ``D``.
        blank_idx: Vocabulary index of the blank symbol.
        sigma: Logits-under-normalization offset (Xu et al. 2023 §3.3).
            Subtracted from log P_token at every node during the
            forward-backward to bias training toward fewer emissions.
        tf32: Use TF32 tensor cores for the joint GEMMs.

    References:
        Xu et al. (2023). Efficient Sequence Transduction by Jointly Predicting Tokens and Durations.
        https://arxiv.org/abs/2304.06795
    """

    def __init__(self, max_duration: int = 4, blank_idx: int = 0, sigma: float = 0.0, tf32: bool = False):
        super().__init__()
        if max_duration < 1:
            raise ValueError(f"max_duration must be >= 1, got {max_duration}")
        self.max_duration = max_duration
        self.durations = list(range(max_duration + 1))
        self.blank_idx = blank_idx
        self.sigma = sigma
        self.tf32 = tf32

    def forward(
        self,
        encoder: Float[torch.Tensor, "batch time d_model"],
        decoder: Float[torch.Tensor, "batch u d_model"],
        joint_W: Float[torch.Tensor, "vocab d_model"],
        joint_W_dur: Float[torch.Tensor, "n_dur d_model"],
        targets: Integer[torch.Tensor, "batch seq"],
        in_lens: Integer[torch.Tensor, "batch"],
        tgt_lens: Integer[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch"]:
        loss = tdt_loss(
            encoder,
            decoder,
            joint_W,
            joint_W_dur,
            targets,
            in_lens,
            tgt_lens,
            self.durations,
            self.blank_idx,
            self.sigma,
            self.tf32,
        )
        return loss / tgt_lens.clamp_min(1)


@dataclass
class TDTLossConfig:
    _target_: str = "asr.loss.tdt.TDTLoss"
    max_duration: int = 4
    blank_idx: int = 0
    sigma: float = 0.0
    tf32: bool = False
