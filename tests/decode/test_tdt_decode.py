# Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
import torch

from asr.decode.tdt import greedy_decode

# Synthetic joints read the encoder vector directly so the test can dictate the argmax
# token and duration at every frame: encoder_out[b, t, :V] are the token logits and
# encoder_out[b, t, V:] are the duration logits. predict/pred are ignored.
BLANK = 0


def _zero_predict(d_model):
    def predict(tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], d_model)

    return predict


def _split_joints(V):
    def joint(enc_t, pred_u):
        return enc_t[:V]

    def joint_dur(enc_t, pred_u):
        return enc_t[V:]

    return joint, joint_dur


def _frames(plan, T, V, durations, default=(BLANK, 1)):
    """Build a (1, T, V+n_dur) encoder tensor whose per-frame argmax is (token, duration)."""
    n_dur = len(durations)
    enc = torch.zeros(1, T, V + n_dur)
    for t in range(T):
        token, dur_value = plan.get(t, default)
        enc[0, t, token] = 10.0
        enc[0, t, V + durations.index(dur_value)] = 10.0
    return enc


def _decode(enc, T_b, V, durations, **kw):
    joint, joint_dur = _split_joints(V)
    in_lens = torch.tensor([T_b])
    # durations are always range(D+1), so the duration head width is max(durations)+1.
    return greedy_decode(
        enc, in_lens, _zero_predict(enc.shape[-1]), joint, joint_dur, max(durations), blank_id=BLANK, **kw
    )


def test_duration_one_emits_one_token_per_frame():
    V, durations, T = 6, [0, 1, 2], 5
    enc = _frames({t: (2, 1) for t in range(T)}, T, V, durations)
    assert _decode(enc, T, V, durations)[0] == [2, 2, 2, 2, 2]


def test_blank_with_duration_skips_frames_and_terminates():
    V, durations, T = 6, [0, 1, 2], 5
    enc = _frames({t: (BLANK, 2) for t in range(T)}, T, V, durations)
    # t: 0 -> 2 -> 4 -> 6 (>=5, stop). No tokens emitted; must terminate, not hang.
    assert _decode(enc, T, V, durations)[0] == []


def test_duration_zero_stacks_up_to_cap():
    V, durations, T = 6, [0, 1, 2], 2
    enc = _frames({t: (3, 0) for t in range(T)}, T, V, durations)
    # Each frame stacks token 3 up to the cap (3), then forced advance: 2 frames * 3 = 6.
    assert _decode(enc, T, V, durations, max_symbols_per_frame=3)[0] == [3, 3, 3, 3, 3, 3]


def test_blank_with_duration_zero_forces_progress():
    V, durations, T = 6, [0, 1, 2], 4
    enc = _frames({t: (BLANK, 0) for t in range(T)}, T, V, durations)
    # blank+0 is illegal in the lattice; the decoder must still force one frame of progress.
    assert _decode(enc, T, V, durations)[0] == []


def test_duration_causes_frame_skip():
    """A duration > 1 must skip intervening frames -- the skipped frame's token is never emitted."""
    V, durations, T = 10, [0, 1, 2, 3], 10
    plan = {
        0: (2, 1),  # emit 2, -> t=1
        1: (BLANK, 2),  # skip, -> t=3 (frame 2 jumped over)
        2: (9, 1),  # frame 2 would emit 9, but it is skipped
        3: (4, 1),  # emit 4, -> t=4
        4: (BLANK, 1),  # -> t=5
        5: (5, 3),  # emit 5, -> t=8
    }
    enc = _frames(plan, T, V, durations)  # frames 6-9 default to (blank, 1)
    hyp = _decode(enc, T, V, durations)[0]
    assert hyp == [2, 4, 5]
    assert 9 not in hyp  # the skipped frame's token never appears


def test_batch_with_different_in_lens():
    V, durations, T = 6, [0, 1, 2], 6
    joint, joint_dur = _split_joints(V)
    enc = torch.zeros(2, T, V + len(durations))
    for t in range(T):  # both utterances: emit token 2 every frame at duration 1
        enc[:, t, 2] = 10.0
        enc[:, t, V + durations.index(1)] = 10.0
    in_lens = torch.tensor([6, 3])
    hyps = greedy_decode(enc, in_lens, _zero_predict(enc.shape[-1]), joint, joint_dur, max(durations), blank_id=BLANK)
    assert hyps[0] == [2, 2, 2, 2, 2, 2]
    assert hyps[1] == [2, 2, 2]  # second utterance only 3 valid frames
