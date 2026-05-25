from pathlib import Path

import torch
from jaxtyping import Float, Integer

_so = next(Path(__file__).parent.glob("_C*.so"))
torch.ops.load_library(str(_so))


def greedy_decode(
    log_probs: Float[torch.Tensor, "batch time vocab"], in_lens: Integer[torch.Tensor, "batch"]
) -> list[list[int]]:
    """Greedy decode each batch entry to a list of token IDs."""
    blank_id = 0

    out = []
    for b in range(log_probs.shape[0]):
        in_len = in_lens[b]
        tokens = log_probs[b].argmax(dim=-1)[:in_len]

        # keep first token and those that differ from previous
        keep = torch.ones_like(tokens, dtype=torch.bool)
        keep[1:] = tokens[1:] != tokens[:-1]
        collapsed = tokens[keep]

        # remove blanks
        collapsed = collapsed[collapsed != blank_id]

        out.append(collapsed.tolist())
    return out


def beam_decode(
    log_probs: Float[torch.Tensor, "batch time vocab"],
    in_lens: Integer[torch.Tensor, "batch"],
    beam_width: int = 32,
) -> list[list[int]]:
    log_probs = log_probs.cpu().contiguous()
    in_lens = in_lens.cpu().contiguous()
    return [o.tolist() for o in torch.ops.asr.beam_decode(log_probs, in_lens, beam_width)]
