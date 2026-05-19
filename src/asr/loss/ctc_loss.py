from pathlib import Path

import torch

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
    def __init__(self, reduction="mean", zero_infinity=False):
        super().__init__()
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, in_lens, tgt_lens):
        loss = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
        if self.zero_infinity:
            loss[loss.isinf()] = 0.0
        if self.reduction == "mean":
            return (loss / tgt_lens).mean()
        elif self.reduction == "none":
            return loss
        raise NotImplementedError(f"unknown reduction {self.reduction}")
