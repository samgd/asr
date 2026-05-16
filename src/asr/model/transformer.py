from typing import cast

import torch
from einx import rearrange  # pyright: ignore[reportPrivateImportUsage]
from jaxtyping import Float, Integer

from asr.model.rotary import RotaryEmbedding


class MHA(torch.nn.Module):
    """Multi-head attention block with rotary embeddings and optional causal masking.

    Args:
        d_model: Model hidden dimension.
        n_head: Number of attention heads.
        is_causal: If true apply casual mask to attention.
        device: Optional device to initialise on.
        dtype: Optional dtype to use.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        is_causal: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.is_causal = is_causal
        assert self.d_model % self.n_head == 0, f"{d_model=} must be divisible by {n_head=}"
        self.head_dim = self.d_model // self.n_head
        self.rope = RotaryEmbedding(self.head_dim, device=device)
        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.Wout = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], seqlens: Integer[torch.Tensor, "batch"] | None = None
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Apply causal multi-head self-attention to the input sequence."""
        B, L, _ = x.shape
        q, k, v = self.Wqkv(x).view(B, L, 3, self.n_head, self.head_dim).unbind(dim=2)

        q = cast(torch.Tensor, rearrange("B L nh hd -> B nh L hd", self.rope(q)))
        k = cast(torch.Tensor, rearrange("B L nh hd -> B nh L hd", self.rope(k)))
        v = cast(torch.Tensor, rearrange("B L nh hd -> B nh L hd", v))

        valid = None
        attn_mask = None
        is_causal = self.is_causal
        if seqlens is not None:
            positions = torch.arange(L, device=x.device)
            valid = positions[None, :] < seqlens.to(device=x.device)[:, None]  # B L
            attn_mask = valid[:, None, None, :]  # B 1 1 L
            if self.is_causal:
                attn_mask = attn_mask & (positions[None, :] <= positions[:, None])[None, None, :, :]
                is_causal = False

        h = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        h = cast(torch.Tensor, rearrange("B nh L hd -> B L (nh hd)", h))
        h = self.Wout(h)
        return h


class SwiGLU(torch.nn.Module):
    """SwiGLU feed-forward block.

    Args:
        d_model: Model hidden dimension.
        d_ff: Feed-forward hidden dimension.
        device: Optional device to initialise on.
        dtype: Optional dtype to use.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = torch.nn.Linear(d_model, 2 * d_ff, bias=False, device=device, dtype=dtype)
        self.fc2 = torch.nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Project up, apply SwiGLU gate, and project back down."""
        a, gate = self.fc1(x).split(self.d_ff, dim=2)
        h = a * torch.nn.functional.silu(gate)
        return self.fc2(h)


class TransformerBlock(torch.nn.Module):
    """Pre-norm transformer block with attention and SwiGLU.

    Args:
        d_model: Model hidden dimension.
        n_head: Number of attention heads.
        is_causal: If true apply casual mask to attention.
        d_ff: Feed-forward hidden dimension.
        device: Optional device to initialise on.
        dtype: Optional dtype to use.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        is_causal: bool,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.is_causal = is_causal
        self.d_ff = d_ff
        self.norm1 = torch.nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = torch.nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MHA(d_model, n_head, is_causal, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], seqlens: Integer[torch.Tensor, "batch"] | None = None
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Apply feedforward and attention with pre-norm and residual components for each."""
        h = x + self.mha(self.norm1(x), seqlens)
        o = h + self.ff(self.norm2(h))
        return o


class Transformer(torch.nn.Module):
    """Stacked pre-norm transformer model with projection head.

    Args:
        depth: Number of transformer blocks.
        d_model: Model hidden dimension.
        n_head: Number of attention heads.
        is_causal: If true apply casual mask to attention.
        d_ff: Feed-forward hidden dimension.
        device: Optional device to initialise on.
        dtype: Optional dtype to use.
    """

    def __init__(
        self,
        n_vocab: int,
        depth: int,
        d_model: int,
        n_head: int,
        is_causal: bool,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[TransformerBlock(d_model, n_head, is_causal, d_ff, device=device, dtype=dtype) for _ in range(depth)]
        )
        self.norm = torch.nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.proj = torch.nn.Linear(d_model, n_vocab, bias=False, device=device, dtype=dtype)

    def forward(
        self, data: Float[torch.Tensor, "batch seq_len d_model"], seqlens: Integer[torch.Tensor, "batch"] | None = None
    ) -> Float[torch.Tensor, "batch seq_len n_vocab"]:
        """Run transformer layers, normalize, and project to logits."""
        h = data
        for layer in self.layers:
            h = layer(h, seqlens)
        return self.proj(self.norm(h))
