from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Integer

_so = next(Path(__file__).parent.glob("_C*.so"))
torch.ops.load_library(str(_so))


class RNNTLossFn(torch.autograd.Function):
    """Fused CUDA RNN-T loss with an exact analytic backward.

    The joint network (tanh + ``joint_W`` matmul + log-softmax) runs inside the
    kernel per lattice node, so the dense ``(B, T, U, V)`` tensor is never
    materialized. forward returns the per-utterance loss ``-log P(y|x)``; backward
    returns the gradient w.r.t. ``encoder``, ``decoder`` and ``joint_W``.
    """

    @staticmethod
    def forward(ctx, encoder, decoder, joint_W, in_lens, tgt_lens, targets, blank_idx, tf32):
        targets = targets.int()
        in_lens = in_lens.int()
        tgt_lens = tgt_lens.int()
        loss, alpha, log_P_blank, log_P_y, log_prob = torch.ops.asr.rnnt_forward(
            encoder.contiguous(),
            decoder.contiguous(),
            joint_W.contiguous(),
            targets,
            in_lens,
            tgt_lens,
            blank_idx,
            tf32,
        )
        ctx.save_for_backward(
            encoder, decoder, joint_W, targets, in_lens, tgt_lens, alpha, log_P_blank, log_P_y, log_prob
        )
        ctx.blank_idx = blank_idx
        ctx.tf32 = tf32
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs
        enc, dec, W, targets, in_lens, tgt_lens, alpha, log_P_blank, log_P_y, log_prob = ctx.saved_tensors
        grad_encoder, grad_decoder, grad_joint_W = torch.ops.asr.rnnt_backward(
            enc.contiguous(),
            dec.contiguous(),
            W.contiguous(),
            targets,
            in_lens,
            tgt_lens,
            ctx.blank_idx,
            alpha,
            log_P_blank,
            log_P_y,
            log_prob,
            grad_loss.contiguous(),
            ctx.tf32,
        )
        return grad_encoder, grad_decoder, grad_joint_W, None, None, None, None, None


def rnnt_loss(
    encoder: Float[torch.Tensor, "batch time d_model"],
    decoder: Float[torch.Tensor, "batch u d_model"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    targets: Integer[torch.Tensor, "batch seq"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int = 0,
    tf32: bool = False,
) -> Float[torch.Tensor, "batch"]:
    """Fused CUDA RNN-T loss (float32 CUDA tensors), differentiable w.r.t. encoder, decoder, joint_W.

    With ``tf32=True`` the joint GEMMs use TF32 tensor cores (~2x faster on Ampere+,
    at ~1e-2 absolute loss error); the default runs full float32.
    """
    return RNNTLossFn.apply(encoder, decoder, joint_W, in_lens, tgt_lens, targets, blank_idx, int(tf32))


class RNNTLoss(torch.nn.Module):
    """RNN-T (transducer) loss, normalized per utterance by the target length.

    The joint network runs fused inside a CUDA kernel, so the dense
    ``(B, T, U, V)`` tensor is never materialized (float32 CUDA tensors only).

    Args:
        blank_idx: vocabulary index of the blank symbol.
        tf32: use TF32 tensor cores for the joint GEMMs (~2x faster on Ampere+,
            ~1e-2 absolute loss error). Default runs full float32.

    Shape:
        encoder:  (B, T, d_model)
        decoder:  (B, U, d_model)   with U = max target length + 1
        joint_W:  (V, d_model)      vocabulary projection, blank at ``blank_idx``
        targets:  (B, S)            label ids, S >= U-1
        in_lens:  (B,)              valid frames per utterance, in [1, T]
        tgt_lens: (B,)              valid labels per utterance, in [0, U-1]

    References:
        Graves (2012). Sequence Transduction with Recurrent Neural Networks.
        https://arxiv.org/abs/1211.3711
    """

    def __init__(self, blank_idx: int = 0, tf32: bool = False):
        super().__init__()
        self.blank_idx = blank_idx
        self.tf32 = tf32

    def forward(
        self,
        encoder: Float[torch.Tensor, "batch time d_model"],
        decoder: Float[torch.Tensor, "batch u d_model"],
        joint_W: Float[torch.Tensor, "vocab d_model"],
        targets: Integer[torch.Tensor, "batch seq"],
        in_lens: Integer[torch.Tensor, "batch"],
        tgt_lens: Integer[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch"]:
        loss = rnnt_loss(encoder, decoder, joint_W, targets, in_lens, tgt_lens, self.blank_idx, self.tf32)
        return loss / tgt_lens.clamp_min(1)


@dataclass
class RNNTLossConfig:
    _target_: str = "asr.loss.rnnt.RNNTLoss"
    blank_idx: int = 0
    tf32: bool = False
