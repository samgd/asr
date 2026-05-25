import torch

from asr.model.transformer import Transformer


def test_transformer_masks_padded_frames_through_stack():
    torch.manual_seed(0)
    model = Transformer(depth=2, d_model=8, n_head=2, is_causal=False, d_ff=16)
    model.eval()

    x = torch.randn(1, 6, 8)
    x_with_different_padding = x.clone()
    x_with_different_padding[:, 3:] = torch.randn(1, 3, 8) * 100.0
    seqlens = torch.tensor([3])

    with torch.no_grad():
        out = model(x, seqlens)
        out_with_different_padding = model(x_with_different_padding, seqlens)

    torch.testing.assert_close(out[:, :3], out_with_different_padding[:, :3], atol=1e-5, rtol=1e-5)
