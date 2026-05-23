from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Integer

_so = next(Path(__file__).parent.glob("_C*.so"))
torch.ops.load_library(str(_so))


class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, in_lens, tgt_lens, zero_infinity):
        alpha, log_Z = torch.ops.asr.ctc_alpha(log_probs, targets, in_lens, tgt_lens)
        ctx.save_for_backward(log_probs, alpha, log_Z, targets, in_lens, tgt_lens)
        ctx.zero_infinity = zero_infinity
        loss = -log_Z
        if zero_infinity:
            loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs  # keep typing happy as backward expects varargs
        log_probs, alpha, log_Z, targets, in_lens, tgt_lens = ctx.saved_tensors
        grad_logits = torch.ops.asr.ctc_grad(
            alpha, log_Z, log_probs, targets, in_lens, tgt_lens, grad_loss.contiguous(), ctx.zero_infinity
        )
        return grad_logits, None, None, None, None


class CTCLoss(torch.nn.Module):
    """Connectionist Temporal Classification (CTC) Loss.

    Args:
        zero_infinity: If True, replace ``-inf`` losses with 0 before reduction so they don't propagate gradients.

    Shape:
        x: (B, T, V) log-probabilities over the vocabularly, including the blank symbol at index 0.
        targets: (B, S_max) target label sequences.
        in_lens: (B,) valid input lengths in [1, T].
        tgt_lens: (B,) valid target lengths in [0, S_max].

    References:
        Graves et al. (2006). Connectionist Temporal Classification: Labelling
        unsegmented sequence data with recurrent neural networks.
        https://www.cs.toronto.edu/~graves/icml_2006.pdf
    """

    def __init__(self, zero_infinity: bool = False):
        super().__init__()
        self.zero_infinity = zero_infinity

    def forward(
        self,
        x: Float[torch.Tensor, "batch time vocab"],
        targets: Integer[torch.Tensor, "batch seq"],
        in_lens: Integer[torch.Tensor, "batch"],
        tgt_lens: Integer[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch"]:
        loss = CTCLossFn.apply(x, targets.int(), in_lens.int(), tgt_lens.int(), self.zero_infinity)
        return loss / tgt_lens


@dataclass
class CTCLossConfig:
    _target_: str = "asr.loss.ctc.CTCLoss"
    zero_infinity: bool = False
