# Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
"""Pure-PyTorch TDT (Token-and-Duration Transducer) loss reference.

A correctness-only oracle with an exact analytic backward, in the same spirit as
``rnnt_reference.py``: it materializes the dense ``(B, T, U, V)`` and
``(B, T, U, D)`` joint tensors, so it is only ever used to validate a fused
kernel, never in training. The hand-written ``backward`` mirrors what a fused CUDA
kernel must compute, so it doubles as the derivation blueprint for that kernel.

TDT (https://arxiv.org/abs/2304.06795) emits a *(token, duration)* pair at every
lattice node. The joint network has two independently-normalized heads, and the
two factors are conditionally independent given the (encoder, decoder) state::

    P(v, d | t, u) = P_token(v | t, u) * P_dur(d | t, u)

A symbol emitted at frame ``t`` with duration ``d`` advances time to ``t + d``;
a blank keeps the label index ``u`` while a real token advances it to ``u + 1``.
The paper's forward recursion (0-indexed here; ``D`` is the set of durations) is::

    alpha(t, u) = sum_{d in D\\{0}} alpha(t-d, u  ) * P(blank, d | t-d, u  )
                + sum_{d in D}      alpha(t-d, u-1) * P(y_u,   d | t-d, u-1)

with two rules that distinguish TDT from RNN-T:

* **Blank may not have duration 0** ("we disallow blank emission with duration 0")
  -- a 0-frame blank is a no-op self-loop, so the blank sum runs over ``D\\{0}``.
  A 0-duration *token* edge is allowed and lets multiple tokens share one frame.
* **Exact landing.** The objective is ``P(y|x) = alpha(T+1, U)`` in the paper's
  1-indexed frames, i.e. the alignment must consume *exactly* ``T`` frames; an
  edge that would overshoot the boundary is simply not on any accepted path. In
  the 0-indexed lattice below the terminal node is ``(in_lens, tgt_lens)``: time
  index ``in_lens`` is "one past the last real frame" and column ``tgt_lens``
  (``= U_b - 1``) is the all-tokens-emitted column.

Note that exact landing makes reachability depend on ``D``: if the remaining
frame count cannot be written as a sum of durations in ``D`` the loss is ``+inf``.
Keeping ``1 in D`` guarantees every length is reachable.
"""

from collections.abc import Sequence

import torch
from jaxtyping import Float, Integer

# Finite "log zero": large enough that exp(NEG) underflows to 0, small enough that
# NEG + NEG stays well inside float range (no -inf -> NaN through logsumexp).
NEG = -1e30


def _joint(
    encoder: Float[torch.Tensor, "batch time d_model"],
    decoder: Float[torch.Tensor, "batch u d_model"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    joint_W_dur: Float[torch.Tensor, "n_dur d_model"],
):
    """Dense two-head joint network over the full (T, U) lattice.

    Returns the post-activation ``hidden`` (needed to backprop through ``tanh`` and
    the two weight matrices) plus the log-softmax token and duration log-probs. The
    heads share ``hidden`` and are each normalized over their own axis.
    """
    pre = encoder[:, :, None, :] + decoder[:, None, :, :]  # (B, T, U, d_model)
    hidden = torch.tanh(pre)
    logp_tok = torch.log_softmax(hidden @ joint_W.T, dim=-1)  # (B, T, U, V)
    logp_dur = torch.log_softmax(hidden @ joint_W_dur.T, dim=-1)  # (B, T, U, n_dur)
    return hidden, logp_tok, logp_dur


def _edge_logprobs(
    logp_tok: Float[torch.Tensor, "batch time u vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int,
    sigma: float = 0.0,
):
    """Token log-probs labelling the two edge *families* leaving each node.

    ``log_P_blank[b, t, u]`` labels blank edges and ``log_P_y[b, t, u]`` labels the
    edges emitting ``targets[b, u]`` (advancing the label axis). The duration factor
    is *not* folded in here -- it is shared across both families at a node and is
    applied per-edge. Identical to the RNN-T reference; only the time advance
    attached to each edge differs.

    ``sigma`` is the logits-under-normalization offset from Xu et al. 2023 §3.3:
    ``log P'(v | t, u) = log_softmax(h_v) - sigma`` on the token head only. Shifting
    the two emitted log-probs by ``-sigma`` here is sufficient -- alpha, beta, logZ
    and the edge posteriors all inherit the under-normalization without further
    changes. NEG sentinels (invalid label edges) are preserved.
    """
    B, T, U, _ = logp_tok.shape
    device = logp_tok.device

    log_P_blank = logp_tok[..., blank_idx] - sigma  # (B, T, U)

    u_ar = torch.arange(U, device=device)
    S = targets.shape[1]
    if S == 0:  # every utterance has an empty target -> no label edges at all
        return log_P_blank, log_P_blank.new_full((B, T, U), NEG)

    y_at = targets.long()[:, u_ar.clamp(max=S - 1)]  # (B, U)
    y_full = y_at[:, None, :].expand(B, T, U)  # (B, T, U)
    log_P_y = torch.gather(logp_tok, 3, y_full.unsqueeze(-1)).squeeze(-1) - sigma  # (B, T, U)

    valid_y = u_ar[None, :] < tgt_lens.long()[:, None]  # (B, U)
    log_P_y = torch.where(valid_y[:, None, :], log_P_y, log_P_y.new_full((), NEG))
    return log_P_blank, log_P_y


def _alpha(
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
    log_P_dur: Float[torch.Tensor, "batch time u n_dur"],
    durations: Sequence[int],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
):
    """Forward variable ``alpha[b, t, u]`` = log prob of reaching node (t, u).

    The time axis is extended to ``T + 1``: index ``in_lens[b]`` is the terminal
    "one past the last frame" row that the alignment must land on exactly.

    Edges with skip ``d >= 1`` arrive from a strictly earlier frame, so they read
    already-finished rows. The skip-0 *token* edge stays on the current frame and
    advances the label, creating an intra-frame chain along ``u``; it is folded in
    a second sweep (over ``u`` ascending) that reads the current column.
    """
    B, T, U = log_P_blank.shape
    device, dtype = log_P_blank.device, log_P_blank.dtype
    negc = torch.full((B,), NEG, device=device, dtype=dtype)

    T_b = in_lens.long()  # frames per utterance              (B,)
    U_b = tgt_lens.long() + 1  # label-axis size per utterance    (B,)

    has_zero = any(int(d) == 0 for d in durations)
    zero_k = next((k for k, d in enumerate(durations) if int(d) == 0), -1)

    alpha: list[list[torch.Tensor]] = []  # alpha[t][u] -> (B,)
    for t in range(T + 1):
        cur: list[torch.Tensor] = []
        for u in range(U):
            terms: list[torch.Tensor] = []

            # ---- edges arriving from an earlier frame: skip d >= 1 ----
            if t > 0:
                for k, d in enumerate(durations):
                    d = int(d)
                    if d < 1:
                        continue  # skip-0 handled in the intra-frame sweep below
                    s = t - d
                    if s < 0:
                        continue
                    src_ok = s <= T_b - 1  # (B,) source must be a real frame

                    blank_ok = src_ok & (u < U_b)  # blank: (s, u) -> (t, u)
                    val = alpha[s][u] + log_P_blank[:, s, u] + log_P_dur[:, s, u, k]
                    terms.append(torch.where(blank_ok, val, negc))

                    if u >= 1:  # label: (s, u-1) -> (t, u), emitting targets[u-1]
                        label_ok = src_ok & (u <= U_b - 1)
                        val = alpha[s][u - 1] + log_P_y[:, s, u - 1] + log_P_dur[:, s, u - 1, k]
                        terms.append(torch.where(label_ok, val, negc))

            if t == 0 and u == 0:  # start node
                terms.append(torch.zeros(B, device=device, dtype=dtype))

            base = torch.logsumexp(torch.stack(terms), dim=0) if terms else negc.clone()

            # ---- intra-frame token edge: skip d == 0, (t, u-1) -> (t, u) ----
            if has_zero and u >= 1 and t <= T - 1:
                z_ok = (t <= T_b - 1) & (u <= U_b - 1)
                val = cur[u - 1] + log_P_y[:, t, u - 1] + log_P_dur[:, t, u - 1, zero_k]
                base = torch.logaddexp(base, torch.where(z_ok, val, negc))

            cur.append(base)
        alpha.append(cur)

    return torch.stack([torch.stack(col, dim=1) for col in alpha], dim=1)  # (B, T+1, U)


def _beta(
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
    log_P_dur: Float[torch.Tensor, "batch time u n_dur"],
    durations: Sequence[int],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
):
    """Backward variable ``beta[b, t, u]`` = log prob of completing from (t, u) to
    the terminal node ``(in_lens, tgt_lens)``.

    Mirror of ``_alpha``: skip ``d >= 1`` edges land on a strictly later frame
    (already finished); the skip-0 token edge lands on the same frame at ``u + 1``
    and is read from the current column (swept ``u`` descending). Each utterance's
    terminal cell is seeded to ``0`` via a per-batch mask, since ``(T_b, U_b-1)``
    differs across the batch.
    """
    B, T, U = log_P_blank.shape
    device, dtype = log_P_blank.device, log_P_blank.dtype
    negc = torch.full((B,), NEG, device=device, dtype=dtype)
    zeros = torch.zeros(B, device=device, dtype=dtype)

    T_b = in_lens.long()  # (B,)
    U_b = tgt_lens.long() + 1  # (B,)

    has_zero = any(int(d) == 0 for d in durations)
    zero_k = next((k for k, d in enumerate(durations) if int(d) == 0), -1)

    # beta_rows[t][u] -> (B,). Pre-filled with NEG placeholders; each cell is read
    # only after it has been written (later frames, then larger u within a frame).
    beta_rows: list[list[torch.Tensor]] = [[negc.clone() for _ in range(U)] for _ in range(T + 1)]
    for t in range(T, -1, -1):
        cur = beta_rows[t]
        for u in range(U - 1, -1, -1):
            terms: list[torch.Tensor] = []
            is_term = (t == T_b) & (u == U_b - 1)  # (B,)

            # ---- edges leaving to a later frame: skip d >= 1 ----
            if t < T:
                for k, d in enumerate(durations):
                    d = int(d)
                    if d < 1:
                        continue  # skip-0 handled in the intra-frame sweep below
                    nt = t + d
                    if nt > T:
                        continue
                    src_ok = t < T_b  # (B,) emit only from a real frame
                    land_ok = nt <= T_b

                    blank_ok = src_ok & land_ok & (u < U_b)  # blank: (t, u) -> (nt, u)
                    val = log_P_blank[:, t, u] + log_P_dur[:, t, u, k] + beta_rows[nt][u]
                    terms.append(torch.where(blank_ok, val, negc))

                    if u < U - 1:  # token: (t, u) -> (nt, u+1), emitting targets[u]
                        token_ok = src_ok & land_ok & (u < U_b - 1)
                        val = log_P_y[:, t, u] + log_P_dur[:, t, u, k] + beta_rows[nt][u + 1]
                        terms.append(torch.where(token_ok, val, negc))

            base = torch.logsumexp(torch.stack(terms), dim=0) if terms else negc.clone()

            # ---- intra-frame token edge: skip d == 0, (t, u) -> (t, u+1) ----
            if has_zero and u < U - 1 and t <= T - 1:
                token0_ok = (t < T_b) & (u < U_b - 1)  # nt == t <= T_b is implied
                val = log_P_y[:, t, u] + log_P_dur[:, t, u, zero_k] + cur[u + 1]
                base = torch.logaddexp(base, torch.where(token0_ok, val, negc))

            cur[u] = torch.where(is_term, zeros, base)  # terminal node has beta = 0

    return torch.stack([torch.stack(col, dim=1) for col in beta_rows], dim=1)  # (B, T+1, U)


def _grads(
    hidden: Float[torch.Tensor, "batch time u d_model"],
    logp_tok: Float[torch.Tensor, "batch time u vocab"],
    logp_dur: Float[torch.Tensor, "batch time u n_dur"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    joint_W_dur: Float[torch.Tensor, "n_dur d_model"],
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
    alpha: Float[torch.Tensor, "batch time1 u"],
    beta: Float[torch.Tensor, "batch time1 u"],
    durations: Sequence[int],
    targets: Integer[torch.Tensor, "batch seq"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int,
    grad_loss: Float[torch.Tensor, "batch"],
):
    """Exact gradient of ``L_b = -log P(y|x)`` w.r.t. encoder, decoder and both heads.

    Every lattice edge ``e`` from node ``from`` to node ``to`` carries log-weight
    ``w_e = log_P_token + log_P_dur``. Its posterior (expected count) is::

        gamma_e = exp(alpha[from] + w_e + beta[to] - logZ),   logZ = log P(y|x)

    and ``dL/dw_e = -gamma_e``. Because ``w_e`` is a *sum* of a token log-prob and a
    duration log-prob, that same ``-gamma_e`` flows to both the token-head entry and
    the duration-head entry composing the edge. Aggregated per node (t, u):

        d L / d logp_tok[blank]    = - sum_d  gamma_blank(t, u, d)
        d L / d logp_tok[y_u]      = - sum_d  gamma_token(t, u, d)
        d L / d logp_dur[d]        = -(gamma_blank(t, u, d) + gamma_token(t, u, d))

    These soft counts are pushed through each head's log-softmax Jacobian, then
    through ``logits = hidden @ W.T`` and ``hidden = tanh(enc + dec)``. This is
    exactly the scatter a fused kernel performs while sweeping the lattice.
    """
    B, T, U, _ = logp_tok.shape
    Dn = logp_dur.shape[-1]
    device, dtype = logp_tok.device, logp_tok.dtype

    b_ar = torch.arange(B, device=device)
    logZ = alpha[b_ar, in_lens.long(), tgt_lens.long()].view(B, 1, 1)  # (B,1,1)

    alpha_s = alpha[:, :T, :]  # alpha at source nodes (frames 0..T-1)  (B,T,U)

    T_b = in_lens.long().view(B, 1, 1)
    U_b = (tgt_lens.long() + 1).view(B, 1, 1)
    t_idx = torch.arange(T, device=device).view(1, T, 1)
    u_idx = torch.arange(U, device=device).view(1, 1, U)
    zero = torch.zeros((), device=device, dtype=dtype)

    # Per-(node, duration) edge posteriors on the source-frame grid (B, T, U, Dn).
    gamma_blank = torch.zeros(B, T, U, Dn, device=device, dtype=dtype)
    gamma_token = torch.zeros(B, T, U, Dn, device=device, dtype=dtype)
    for k, d in enumerate(durations):
        d = int(d)
        nt = t_idx + d  # landing frame (1, T, 1)

        # beta at the landing nodes, gathered onto the source grid (shift by d).
        bb = beta.new_full((B, T, U), NEG)  # beta[t+d, u]   (blank lands)
        bt = beta.new_full((B, T, U), NEG)  # beta[t+d, u+1] (token lands)
        rows = min(T, T + 1 - d)  # source frames whose landing t+d <= T
        if rows > 0:
            bb[:, :rows, :] = beta[:, d : d + rows, :]
            bt[:, :rows, : U - 1] = beta[:, d : d + rows, 1:]

        land_ok = nt <= T_b
        if d >= 1:  # blank disallows duration 0
            valid_blank = (t_idx < T_b) & (u_idx < U_b) & land_ok
            g = alpha_s + log_P_blank + logp_dur[..., k] + bb - logZ
            gamma_blank[..., k] = torch.where(valid_blank, torch.exp(g), zero)

        valid_token = (t_idx < T_b) & (u_idx < U_b - 1) & land_ok
        g = alpha_s + log_P_y + logp_dur[..., k] + bt - logZ
        gamma_token[..., k] = torch.where(valid_token, torch.exp(g), zero)

    # d L / d logp_tok: blank column and the emitted-target columns.
    grad_logp_tok = torch.zeros_like(logp_tok)
    grad_logp_tok[..., blank_idx] = -gamma_blank.sum(-1)
    S = targets.shape[1]
    if S > 0:
        u_ar = torch.arange(U, device=device)
        y_full = targets.long()[:, u_ar.clamp(max=S - 1)][:, None, :].expand(B, T, U)
        grad_logp_tok.scatter_add_(3, y_full.unsqueeze(-1), (-gamma_token.sum(-1)).unsqueeze(-1))

    # d L / d logp_dur: both edge families share the node's duration distribution.
    grad_logp_dur = -(gamma_blank + gamma_token)  # (B, T, U, Dn)

    # Through each head's log-softmax: d L / d logits = g - softmax * sum(g).
    grad_logits_tok = grad_logp_tok - logp_tok.exp() * grad_logp_tok.sum(-1, keepdim=True)
    grad_logits_dur = grad_logp_dur - logp_dur.exp() * grad_logp_dur.sum(-1, keepdim=True)
    grad_logits_tok = grad_logits_tok * grad_loss.view(B, 1, 1, 1)  # chain upstream grad
    grad_logits_dur = grad_logits_dur * grad_loss.view(B, 1, 1, 1)

    # Through logits = hidden @ W.T  and  hidden = tanh(enc + dec).
    grad_joint_W = torch.einsum("btuv,btuh->vh", grad_logits_tok, hidden)
    grad_joint_W_dur = torch.einsum("btud,btuh->dh", grad_logits_dur, hidden)
    grad_hidden = torch.einsum("btuv,vh->btuh", grad_logits_tok, joint_W) + torch.einsum(
        "btud,dh->btuh", grad_logits_dur, joint_W_dur
    )
    grad_pre = grad_hidden * (1.0 - hidden * hidden)

    grad_encoder = grad_pre.sum(dim=2)  # sum over the label axis -> (B, T, d_model)
    grad_decoder = grad_pre.sum(dim=1)  # sum over the time axis  -> (B, U, d_model)

    return grad_encoder, grad_decoder, grad_joint_W, grad_joint_W_dur


class TDTLossFn(torch.autograd.Function):
    """Reference TDT loss with an exact analytic backward.

    forward returns the per-utterance loss ``-log P(y|x)``; backward returns the
    gradient w.r.t. ``encoder``, ``decoder``, ``joint_W`` and ``joint_W_dur``.

    Shapes:
        encoder:     (B, T, d_model)
        decoder:     (B, U, d_model)   with U = max target length + 1
        joint_W:     (V, d_model)      token head
        joint_W_dur: (n_dur, d_model)  duration head, n_dur == len(durations)
        in_lens:     (B,)              valid frames per utterance, in [1, T]
        tgt_lens:    (B,)              valid labels per utterance, in [0, U-1]
        targets:     (B, S)            label ids, S >= U-1
    """

    @staticmethod
    def forward(ctx, encoder, decoder, joint_W, joint_W_dur, in_lens, tgt_lens, targets, durations, blank_idx, sigma):
        hidden, logp_tok, logp_dur = _joint(encoder, decoder, joint_W, joint_W_dur)
        log_P_blank, log_P_y = _edge_logprobs(logp_tok, targets, tgt_lens, blank_idx, sigma)
        alpha = _alpha(log_P_blank, log_P_y, logp_dur, durations, in_lens, tgt_lens)
        beta = _beta(log_P_blank, log_P_y, logp_dur, durations, in_lens, tgt_lens)

        ctx.save_for_backward(
            hidden,
            logp_tok,
            logp_dur,
            joint_W,
            joint_W_dur,
            log_P_blank,
            log_P_y,
            alpha,
            beta,
            targets,
            in_lens,
            tgt_lens,
        )
        ctx.durations = durations
        ctx.blank_idx = blank_idx

        b_ar = torch.arange(encoder.shape[0], device=encoder.device)
        return -alpha[b_ar, in_lens.long(), tgt_lens.long()]  # (B,) exact-landing terminal

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs
        (
            hidden,
            logp_tok,
            logp_dur,
            joint_W,
            joint_W_dur,
            log_P_blank,
            log_P_y,
            alpha,
            beta,
            targets,
            in_lens,
            tgt_lens,
        ) = ctx.saved_tensors
        grad_encoder, grad_decoder, grad_joint_W, grad_joint_W_dur = _grads(
            hidden,
            logp_tok,
            logp_dur,
            joint_W,
            joint_W_dur,
            log_P_blank,
            log_P_y,
            alpha,
            beta,
            ctx.durations,
            targets,
            in_lens,
            tgt_lens,
            ctx.blank_idx,
            grad_loss,
        )
        # grads line up with forward args: encoder, decoder, joint_W, joint_W_dur, then non-tensors.
        return grad_encoder, grad_decoder, grad_joint_W, grad_joint_W_dur, None, None, None, None, None, None


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
) -> Float[torch.Tensor, "batch"]:
    """Per-utterance TDT loss, differentiable w.r.t. encoder, decoder and both heads.

    ``durations`` is the sequence of frame skips, e.g. ``[0, 1, 2, 3, 4]``; keep
    ``1`` in it so every frame count is reachable under exact landing. ``sigma`` is
    the token-head logits-under-normalization offset from Xu et al. 2023 §3.3.
    """
    return TDTLossFn.apply(
        encoder, decoder, joint_W, joint_W_dur, in_lens, tgt_lens, targets, durations, blank_idx, sigma
    )
