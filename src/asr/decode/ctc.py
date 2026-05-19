import torch
from jaxtyping import Float, Integer


def greedy_decode(
    logits: Float[torch.Tensor, "batch time vocab"], in_lens: Integer[torch.Tensor, "batch"]
) -> list[list[int]]:
    """Greedy decode each batch entry to a list of token IDs."""
    blank_id = 0

    out = []
    for b in range(logits.shape[0]):
        in_len = in_lens[b]
        tokens = logits[b].argmax(dim=-1)[:in_len]

        # keep first token and those that differ from previous
        keep = torch.ones_like(tokens, dtype=torch.bool)
        keep[1:] = tokens[1:] != tokens[:-1]
        collapsed = tokens[keep]

        # remove blanks
        collapsed = collapsed[collapsed != blank_id]

        out.append(collapsed.tolist())
    return out
