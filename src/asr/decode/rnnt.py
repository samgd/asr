from collections.abc import Callable

import torch
from jaxtyping import Float, Integer

PredictFn = Callable[[Integer[torch.Tensor, "batch seq_len"]], Float[torch.Tensor, "batch seq_len d_model"]]

JointFn = Callable[[Float[torch.Tensor, "d_model"], Float[torch.Tensor, "d_model"]], Float[torch.Tensor, "vocab"]]


@torch.no_grad()
def greedy_decode(
    encoder_out: Float[torch.Tensor, "batch n_frames d_model"],
    in_lens: Integer[torch.Tensor, "batch"],
    predict: PredictFn,
    joint: JointFn,
    blank_id: int = 0,
    sos_id: int = 1,
    max_symbols_per_frame: int = 10,
) -> list[list[int]]:
    """Greedy RNN-T decode."""
    device = encoder_out.device
    hyps: list[list[int]] = []
    for b in range(encoder_out.shape[0]):
        prefix = [sos_id]
        pred = predict(torch.tensor(prefix, device=device).unsqueeze(0))[0, -1]
        hyp: list[int] = []
        for t in range(int(in_lens[b])):
            for _ in range(max_symbols_per_frame):
                token = int(joint(encoder_out[b, t], pred).argmax())
                if token == blank_id:
                    break
                hyp.append(token)
                prefix.append(token)
                pred = predict(torch.tensor(prefix, device=device).unsqueeze(0))[0, -1]
        hyps.append(hyp)
    return hyps
