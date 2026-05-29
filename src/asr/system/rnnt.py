from dataclasses import dataclass, field
from typing import Any

import jiwer
import torch
from jaxtyping import Float, Integer
from omegaconf import MISSING

from asr.data.dataset import Batch
from asr.decode.rnnt import greedy_decode
from asr.loss.rnnt import RNNTLossConfig


class RNNTSystem(torch.nn.Module):
    def __init__(self, encoder, decoder, loss, n_vocab: int, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = torch.nn.Embedding(n_vocab, encoder.out_dim, padding_idx=0)
        self.joint = torch.nn.Linear(encoder.out_dim, n_vocab, bias=False)
        self.loss = loss
        self.tokenizer = tokenizer

    def _predict(self, tokens: Integer[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.decoder(self.embed(tokens))

    def _joint(
        self,
        enc_t: Float[torch.Tensor, "d_model"],
        pred_u: Float[torch.Tensor, "d_model"],
    ) -> Float[torch.Tensor, "vocab"]:
        return self.joint(torch.tanh(enc_t + pred_u))

    def _step(
        self, batch: Batch
    ) -> tuple[
        Float[torch.Tensor, "batch n_frames d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
        Integer[torch.Tensor, "batch"],
        Float[torch.Tensor, "batch"],
    ]:
        x, y, xl, yl = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, xl = self.encoder(x, xl)
            start = y.new_full((y.shape[0], 1), fill_value=1)
            hy = self.embed(torch.cat([start, y], dim=1))
            dec = self.decoder(hy)
        return enc, dec, xl, self.loss(enc.float(), dec.float(), self.joint.weight.float(), y, xl, yl)

    def train_step(self, batch: Batch) -> Float[torch.Tensor, ""]:
        _, _, _, loss_per = self._step(batch)
        return loss_per.mean()

    def eval_step(self, batch: Batch) -> list[dict]:
        _, y, _, yl = batch
        enc, _, xl, loss_per = self._step(batch)

        hyps = greedy_decode(enc.float(), xl, self._predict, self._joint, blank_id=0)

        out = []
        for i in range(loss_per.shape[0]):
            ref = self.tokenizer.decode(y[i, : yl[i]].tolist())
            hyp = self.tokenizer.decode(hyps[i])
            wo = jiwer.process_words(ref, hyp)
            edits = wo.substitutions + wo.deletions + wo.insertions
            ref_len = len(ref.split())
            out.append({"loss": loss_per[i].item(), "wer_edit": edits, "wer_ref_len": ref_len})
        return out


@dataclass
class RNNTSystemConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"encoder": "default"},
            {"decoder": "default"},
        ]
    )
    _target_: str = "asr.system.rnnt.RNNTSystem"
    encoder: Any = MISSING
    decoder: Any = MISSING
    loss: RNNTLossConfig = field(default_factory=RNNTLossConfig)
