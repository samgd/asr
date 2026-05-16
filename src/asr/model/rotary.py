import torch
from jaxtyping import Float


def angular_freqs(
    head_dim: int, base: float = 10_000.0, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> Float[torch.Tensor, " half_head_dim"]:
    """Compute frequencies for rotary embeddings.

    Args:
        head_dim: Attention head dimension (must be even).
        base: Exponential base controlling frequency spacing.
        device: Target device for the frequency tensor.
        dtype: Desired dtype of the returned tensor.

    Returns:
        Vector of angular frequencies of length ``head_dim // 2``.
    """
    j = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    return (base ** ((-2 / head_dim) * j)).to(dtype)


def rotate_half(x: Float[torch.Tensor, "... 2 half_head_dim"]) -> Float[torch.Tensor, "... 2 half_head_dim"]:
    """Rotate (x, y) points to (-y, x)."""
    x1 = x[..., 0, :]
    y1 = x[..., 1, :]
    return torch.stack([-y1, x1], dim=-2)


def rotary_embedding(
    x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    cos: Float[torch.Tensor, "seq_len half_head_dim"],
    sin: Float[torch.Tensor, "seq_len half_head_dim"],
) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
    x = x.unflatten(dim=-1, sizes=(2, -1))
    # The vector x stores the indices of 2D points in a non-interleaved layout. e.g. The x1 components for all
    # points are stored first followed by the x2 components for all points. So `x = [x1, x2]` where x1 is a vector.
    #
    # The single-2D-point rotation matrix is:
    #
    #     [[cos(theta) -sin(theta)],
    #      [sin(theta)  cos(theta)]]
    #
    # Applying this to all points gives the `x' = [x1', x2']` vector where:
    #
    #     x1' = x1*cos(theta) - x2*sin(theta)
    #     x2' = x1*sin(theta) + x2*cos(theta)
    #
    # The vector x is split into the x1 and x2 components. Each component performs two multplies and an
    # add/subtract before being concatenated back together.
    #
    # An equivalent, but computationally neater, method is to form `x_r = [-x2, x1]` and do
    # `x' = x*cos(theta) + x_r*sin(theta)`:
    out = x * cos + rotate_half(x) * sin
    return out.flatten(start_dim=-2)


class RotaryEmbedding(torch.nn.Module):
    """Applies rotary positional embeddings.

    Args:
        head_dim: Attention head dimension (must be even).
        base: Exponential base controlling frequency spacing.
        device: Target device for the frequency tensor.
        dtype: Desired dtype of the returned tensor.
        max_seq_len: Optional initial cache length.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Model head dimension must be divisible by 2"
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.freqs = angular_freqs(self.head_dim, self.base, device, dtype=torch.float32)
        self._cos = None
        self._sin = None

    def _get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Float[torch.Tensor, "seq_len 1 1 half_head_dim"], Float[torch.Tensor, "seq_len 1 1 half_head_dim"]]:
        if self._cos is not None and self._cos.dtype == dtype and self._cos.shape[0] >= seq_len:
            if device is not None:
                self._cos = self._cos.to(device)
                self._sin = self._sin.to(device)  # type: ignore
            return self._cos[:seq_len], self._sin[:seq_len]
        self.max_seq_len = seq_len if self.max_seq_len is None else max(seq_len, self.max_seq_len)
        if device is not None:
            self.freqs = self.freqs.to(device)
        pos = torch.arange(self.max_seq_len, dtype=self.freqs.dtype, device=self.freqs.device)
        self._cos = torch.cos(self.freqs[None, :] * pos[:, None]).to(dtype)[:, None, None, :]
        self._sin = torch.sin(self.freqs[None, :] * pos[:, None]).to(dtype)[:, None, None, :]
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    ) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
        """Rotate a vector."""
        cos, sin = self._get_cos_sin(x.shape[-3], device=x.device, dtype=x.dtype)
        return rotary_embedding(x, cos, sin)
