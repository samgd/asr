from collections.abc import Callable

import torch
from jaxtyping import Float, Integer

PredictFn = Callable[[Integer[torch.Tensor, "batch seq_len"]], Float[torch.Tensor, "batch seq_len d_model"]]

JointFn = Callable[[Float[torch.Tensor, "d_model"], Float[torch.Tensor, "d_model"]], Float[torch.Tensor, "vocab"]]

DurationFn = Callable[[Float[torch.Tensor, "d_model"], Float[torch.Tensor, "d_model"]], Float[torch.Tensor, "n_dur"]]


@torch.no_grad()
def greedy_decode(
    encoder_out: Float[torch.Tensor, "batch n_frames d_model"],
    in_lens: Integer[torch.Tensor, "batch"],
    predict: PredictFn,
    joint: JointFn,
    joint_dur: DurationFn,
    max_duration: int,
    blank_id: int = 0,
    sos_id: int = 1,
    max_symbols_per_frame: int = 10,
) -> tuple[list[list[int]], list[list[int]]]:
    """Greedy TDT decode.

    Returns:
        List of hypotheses and a list of durations for each token, including blanks.
    """
    durations = list(range(max_duration + 1))
    device = encoder_out.device
    hyps: list[list[int]] = []
    chosen: list[list[int]] = []
    for b in range(encoder_out.shape[0]):
        prefix = [sos_id]
        pred = predict(torch.tensor(prefix, device=device).unsqueeze(0))[0, -1]
        hyp: list[int] = []
        durs: list[int] = []
        t = 0
        T_b = int(in_lens[b])
        while t < T_b:
            for _ in range(max_symbols_per_frame):
                token = int(joint(encoder_out[b, t], pred).argmax())
                d = durations[int(joint_dur(encoder_out[b, t], pred).argmax())]
                durs.append(d)
                if token != blank_id:
                    hyp.append(token)
                    prefix.append(token)
                    pred = predict(torch.tensor(prefix, device=device).unsqueeze(0))[0, -1]
                if d > 0:
                    t += d
                    break
                # Blank must progress.
                if token == blank_id:
                    t += 1
                    break
            else:
                # Hit max_symbols_per_frame, force progress.
                t += 1
        hyps.append(hyp)
        chosen.append(durs)
    return hyps, chosen
