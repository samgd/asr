import torch
from jaxtyping import Float, Integer

# Finite "log zero": large enough that exp(NEG) underflows to 0, small enough that
# NEG + NEG stays well inside float range (no -inf -> NaN through autograd / logaddexp).
NEG = -1e30


def antidiag_indices(T: int, U: int, device=None):
    D = U + T - 1
    d = torch.arange(D, device=device)

    # number of entries in each antidiagonal
    n = torch.minimum(
        torch.minimum(d + 1, torch.full_like(d, U)),
        torch.minimum(torch.full_like(d, T), torch.full_like(d, D) - d),
    )

    # diagonal id for every flattened entry
    diag = torch.repeat_interleave(d, n)

    # offset within each diagonal: 0, 1, 2, ...
    diag_start = torch.repeat_interleave(torch.cumsum(n, dim=0) - n, n)
    i = torch.arange(U * T, device=device) - diag_start

    start_t = torch.clamp(d - U + 1, min=0)
    start_u = torch.clamp(d, max=U - 1)

    t = start_t[diag] + i
    u = start_u[diag] - i

    offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), torch.cumsum(n, dim=0)])

    return t, u, offsets


def _joint(
    encoder: Float[torch.Tensor, "batch time d_model"],
    decoder: Float[torch.Tensor, "batch u d_model"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
):
    """Dense joint network over the full (T, U) lattice.

    Returns the post-activation ``hidden`` (needed to backprop through ``tanh`` and
    ``joint_W``) and the log-softmax ``logp`` over the vocabulary.
    """
    pre = encoder[:, :, None, :] + decoder[:, None, :, :]  # (B, T, U, d_model)
    hidden = torch.tanh(pre)
    logits = hidden @ joint_W.T  # (B, T, U, V)
    logp = torch.log_softmax(logits, dim=-1)
    return hidden, logp


def _edge_logprobs(
    logp: Float[torch.Tensor, "batch time u vocab"],
    targets: Integer[torch.Tensor, "batch seq"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int,
):
    """Pull out the two log-probs that label every lattice edge.

    ``log_P_blank[b, t, u]`` is the horizontal (blank, advance time) edge and
    ``log_P_y[b, t, u]`` is the vertical (emit ``targets[b, u]``, advance label) edge.
    Vertical edges only exist while ``u < tgt_lens[b]`` — the rest are masked to NEG.
    """
    B, T, U, _ = logp.shape
    device = logp.device

    log_P_blank = logp[..., blank_idx]  # (B, T, U)

    u_ar = torch.arange(U, device=device)
    S = targets.shape[1]
    if S == 0:  # every utterance has an empty target -> no vertical edges at all
        return log_P_blank, log_P_blank.new_full((B, T, U), NEG)

    # Edge leaving label position u emits targets[:, u]; clamp keeps the gather in
    # bounds for u >= tgt_lens (those columns are masked out below).
    y_at = targets.long()[:, u_ar.clamp(max=S - 1)]  # (B, U)
    y_full = y_at[:, None, :].expand(B, T, U)  # (B, T, U)
    log_P_y = torch.gather(logp, 3, y_full.unsqueeze(-1)).squeeze(-1)  # (B, T, U)

    valid_y = u_ar[None, :] < tgt_lens.long()[:, None]  # (B, U)
    log_P_y = torch.where(valid_y[:, None, :], log_P_y, log_P_y.new_full((), NEG))
    return log_P_blank, log_P_y


def _alpha(
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
):
    """Forward variable: ``alpha[b, t, u]`` = log prob of reaching node (t, u).

    Swept along antidiagonals so every node on a diagonal is updated in one batched,
    vectorised step (the recursion only depends on the previous diagonal).
    """
    B, T, U = log_P_blank.shape
    device = log_P_blank.device

    t_flat, u_flat, offsets = antidiag_indices(T, U, device=device)

    alpha = log_P_blank.new_full((B, T, U), NEG)
    alpha[:, 0, 0] = 0.0

    for d in range(1, T + U - 1):
        s, e = offsets[d].item(), offsets[d + 1].item()
        t_d = t_flat[s:e]
        u_d = u_flat[s:e]

        has_h = (t_d > 0)[None]  # (1, N_d)
        has_v = (u_d > 0)[None]
        t_prev = (t_d - 1).clamp(min=0)
        u_prev = (u_d - 1).clamp(min=0)

        term_h = alpha[:, t_prev, u_d] + log_P_blank[:, t_prev, u_d]
        term_h = torch.where(has_h, term_h, term_h.new_full((), NEG))

        term_v = alpha[:, t_d, u_prev] + log_P_y[:, t_d, u_prev]
        term_v = torch.where(has_v, term_v, term_v.new_full((), NEG))

        alpha[:, t_d, u_d] = torch.logaddexp(term_h, term_v)

    return alpha


def _beta(
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
):
    """Backward variable: ``beta[b, t, u]`` = log prob of completing the alignment
    from node (t, u) to the end (emit the remaining labels, then a terminal blank).

    Each utterance terminates at its own ``(in_lens[b]-1, tgt_lens[b])`` node, so the
    per-batch boundary is applied with masks rather than a single shared terminal.
    Swept along antidiagonals in reverse (a node only depends on the next diagonal).
    """
    B, T, U = log_P_blank.shape
    device = log_P_blank.device

    t_flat, u_flat, offsets = antidiag_indices(T, U, device=device)

    beta = log_P_blank.new_full((B, T, U), NEG)
    T_b = in_lens.long().view(B, 1)  # frames per utterance
    U_b = (tgt_lens.long() + 1).view(B, 1)  # label-axis size per utterance

    for d in range(T + U - 2, -1, -1):
        s, e = offsets[d].item(), offsets[d + 1].item()
        t_d = t_flat[s:e]
        u_d = u_flat[s:e]
        t_row = t_d[None]  # (1, N_d)
        u_row = u_d[None]

        node_valid = (t_row < T_b) & (u_row < U_b)  # (B, N_d)
        is_term = (t_row == T_b - 1) & (u_row == U_b - 1)
        can_blank = (t_row < T_b - 1) & node_valid  # blank edge stays in lattice
        can_label = (u_row < U_b - 1) & node_valid  # vertical edge stays in lattice

        lp_b = log_P_blank[:, t_d, u_d]
        lp_y = log_P_y[:, t_d, u_d]
        beta_t1 = beta[:, (t_d + 1).clamp(max=T - 1), u_d]  # beta[t+1, u]
        beta_u1 = beta[:, t_d, (u_d + 1).clamp(max=U - 1)]  # beta[t, u+1]

        negc = lp_b.new_full((), NEG)
        # Terminal blank completes the alignment (nothing follows it -> beta_after = 0).
        blank = torch.where(is_term, lp_b, torch.where(can_blank, lp_b + beta_t1, negc))
        label = torch.where(can_label, lp_y + beta_u1, negc)

        beta[:, t_d, u_d] = torch.where(node_valid, torch.logaddexp(blank, label), negc)

    return beta


def _grads(
    hidden: Float[torch.Tensor, "batch time u d_model"],
    logp: Float[torch.Tensor, "batch time u vocab"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    log_P_blank: Float[torch.Tensor, "batch time u"],
    log_P_y: Float[torch.Tensor, "batch time u"],
    alpha: Float[torch.Tensor, "batch time u"],
    beta: Float[torch.Tensor, "batch time u"],
    targets: Integer[torch.Tensor, "batch seq"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int,
    grad_loss: Float[torch.Tensor, "batch"],
):
    """Exact gradient of the per-utterance loss w.r.t. encoder, decoder and joint_W.

    The loss is ``L_b = -log P(y|x)``. For every lattice edge the gradient w.r.t. its
    log-prob is minus the edge posterior (its share of total alignment probability):

        d L_b / d log_P_edge = -exp(alpha[from] + log_P_edge + beta[to] - log P(y|x))

    Those posteriors are scattered back into ``logp``, pushed through the log-softmax
    Jacobian to ``logits``, then through ``logits = tanh(enc+dec) @ joint_W.T``.
    ``grad_loss`` is the upstream gradient on the (B,) loss vector.
    """
    B, T, U, _ = logp.shape
    device = logp.device

    log_prob = beta[:, 0, 0].view(B, 1, 1)  # log P(y|x) = -loss, the path normaliser

    t_idx = torch.arange(T, device=device).view(1, T, 1)
    u_idx = torch.arange(U, device=device).view(1, 1, U)
    T_b = in_lens.long().view(B, 1, 1)
    U_b = (tgt_lens.long() + 1).view(B, 1, 1)

    node_valid = (t_idx < T_b) & (u_idx < U_b)
    is_term = (t_idx == T_b - 1) & (u_idx == U_b - 1)
    can_blank = (t_idx < T_b - 1) & node_valid
    can_label = (u_idx < U_b - 1) & node_valid

    zero = torch.zeros((), device=device, dtype=logp.dtype)

    # Blank-edge posterior. Interior edges land on beta[t+1, u]; the terminal blank
    # completes the path so its "beta after" is 0 (its posterior is exactly 1).
    beta_after_blank = log_P_blank.new_full((B, T, U), NEG)
    beta_after_blank[:, : T - 1, :] = beta[:, 1:, :]
    beta_after_blank = torch.where(is_term, torch.zeros_like(beta_after_blank), beta_after_blank)
    blank_logpost = alpha + log_P_blank + beta_after_blank - log_prob
    blank_post = torch.where(can_blank | is_term, torch.exp(blank_logpost), zero)

    # Label-edge posterior, landing on beta[t, u+1].
    beta_after_label = log_P_blank.new_full((B, T, U), NEG)
    beta_after_label[:, :, : U - 1] = beta[:, :, 1:]
    label_logpost = alpha + log_P_y + beta_after_label - log_prob
    label_post = torch.where(can_label, torch.exp(label_logpost), zero)

    # d L / d logp: -posterior at the blank and the emitted-target vocab indices.
    grad_logp = torch.zeros_like(logp)
    grad_logp[..., blank_idx] = -blank_post
    S = targets.shape[1]
    if S > 0:
        u_ar = torch.arange(U, device=device)
        y_full = targets.long()[:, u_ar.clamp(max=S - 1)][:, None, :].expand(B, T, U)
        grad_logp.scatter_add_(3, y_full.unsqueeze(-1), (-label_post).unsqueeze(-1))

    # Through log-softmax: d L / d logits = g - softmax * sum(g).
    softmax = logp.exp()
    grad_logits = grad_logp - softmax * grad_logp.sum(-1, keepdim=True)
    grad_logits = grad_logits * grad_loss.view(B, 1, 1, 1)  # chain in the upstream grad

    # Through logits = hidden @ joint_W.T  and  hidden = tanh(enc + dec).
    grad_joint_W = torch.einsum("btuv,btuk->vk", grad_logits, hidden)
    grad_hidden = torch.einsum("btuv,vk->btuk", grad_logits, joint_W)
    grad_pre = grad_hidden * (1.0 - hidden * hidden)

    grad_encoder = grad_pre.sum(dim=2)  # sum over the label axis -> (B, T, d_model)
    grad_decoder = grad_pre.sum(dim=1)  # sum over the time axis  -> (B, U, d_model)

    return grad_encoder, grad_decoder, grad_joint_W


class RNNTLossFn(torch.autograd.Function):
    """Reference RNN-T loss with an exact analytic backward.

    forward returns the per-utterance loss ``-log P(y|x)``; backward returns the
    gradient w.r.t. ``encoder``, ``decoder`` and ``joint_W``.

    Shapes:
        encoder:  (B, T, d_model)
        decoder:  (B, U, d_model)   with U = max target length + 1
        joint_W:  (V, d_model)
        in_lens:  (B,)              valid frames per utterance, in [1, T]
        tgt_lens: (B,)              valid labels per utterance, in [0, U-1]
        targets:  (B, S)            label ids, S >= U-1
    """

    @staticmethod
    def forward(ctx, encoder, decoder, joint_W, in_lens, tgt_lens, targets, blank_idx):
        hidden, logp = _joint(encoder, decoder, joint_W)
        log_P_blank, log_P_y = _edge_logprobs(logp, targets, tgt_lens, blank_idx)
        alpha = _alpha(log_P_blank, log_P_y)
        beta = _beta(log_P_blank, log_P_y, in_lens, tgt_lens)

        ctx.save_for_backward(hidden, logp, joint_W, log_P_blank, log_P_y, alpha, beta, targets, in_lens, tgt_lens)
        ctx.blank_idx = blank_idx
        return -beta[:, 0, 0]  # (B,) per-utterance loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_loss,) = grad_outputs
        hidden, logp, joint_W, log_P_blank, log_P_y, alpha, beta, targets, in_lens, tgt_lens = ctx.saved_tensors
        grad_encoder, grad_decoder, grad_joint_W = _grads(
            hidden,
            logp,
            joint_W,
            log_P_blank,
            log_P_y,
            alpha,
            beta,
            targets,
            in_lens,
            tgt_lens,
            ctx.blank_idx,
            grad_loss,
        )
        return grad_encoder, grad_decoder, grad_joint_W, None, None, None, None


def rnnt_loss(
    encoder: Float[torch.Tensor, "batch time d_model"],
    decoder: Float[torch.Tensor, "batch u d_model"],
    joint_W: Float[torch.Tensor, "vocab d_model"],
    targets: Integer[torch.Tensor, "batch seq"],
    in_lens: Integer[torch.Tensor, "batch"],
    tgt_lens: Integer[torch.Tensor, "batch"],
    blank_idx: int = 0,
) -> Float[torch.Tensor, "batch"]:
    """Per-utterance RNN-T loss, differentiable w.r.t. encoder, decoder and joint_W."""
    return RNNTLossFn.apply(encoder, decoder, joint_W, in_lens, tgt_lens, targets, blank_idx)


def transducer_alpha(encoder, in_lens, decoder, tgt_lens, joint_W, targets, blank_idx=0):
    """Forward-only loss (kept for compatibility); equivalent to ``rnnt_loss``."""
    return rnnt_loss(encoder, decoder, joint_W, targets, in_lens, tgt_lens, blank_idx)
