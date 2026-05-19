from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, Int

_so = next(Path(__file__).parent.glob("_C*.so"))
torch.ops.load_library(str(_so))


class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, in_lens, tgt_lens):
        alpha, log_Z = torch.ops.asr.ctc_alpha(log_probs, targets, in_lens, tgt_lens)
        ctx.save_for_backward(log_probs, alpha, log_Z, targets, in_lens, tgt_lens)
        return -log_Z

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs  # keep typing happy as backward expects varargs
        log_probs, alpha, log_Z, targets, in_lens, tgt_lens = ctx.saved_tensors
        grad_logits = torch.ops.asr.ctc_grad(
            alpha, log_Z, log_probs, targets, in_lens, tgt_lens, grad_loss.contiguous()
        )
        return grad_logits, None, None, None


class CTCLoss(torch.nn.Module):
    """Connectionist Temporal Classification (CTC) Loss.

    Args:
        reduction: How to reduce the per-sample loss to a scalar.
            "mean" averages the length-normalized losses across the batch and "none" returns the per-sample tensor.
        zero_infinity: If True, replace ``-inf`` losses with 0 before reduction so they don't propagate gradients.

    Shape:
        log_probs: (B, T, V) log-probabilities over the vocabularly, including the blank symbol at index 0.
        targets: (B, S_max) target label sequences.
        in_lens: (B,) valid input lengths in [1, T].
        tgt_lens: (B,) valid target lengths in [0, S_max].

    References:
        Graves et al. (2006). Connectionist Temporal Classification: Labelling
        unsegmented sequence data with recurrent neural networks.
        https://www.cs.toronto.edu/~graves/icml_2006.pdf
    """

    def __init__(self, reduction: Literal["mean", "none"] = "mean", zero_infinity: bool = False):
        super().__init__()
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: Float[torch.Tensor, "batch time vocab"],
        targets: Int[torch.Tensor, "batch seq"],
        in_lens: Int[torch.Tensor, "batch"],
        tgt_lens: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch"]:
        loss = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
        if self.zero_infinity:
            loss[loss.isinf()] = 0.0
        match self.reduction:
            case "mean":
                return (loss / tgt_lens).mean()
            case "none":
                return loss
            case _:
                raise NotImplementedError(f"unknown reduction {self.reduction}")
