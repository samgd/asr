# ASR

An experimental repo to play around with automatic speech recognition (ASR) papers and ideas.

## Usage

Training is driven by [Hydra](https://hydra.cc/docs/intro/).

Eventually there will be YAML files containing standard configurations for each part -- data, model, loss, etc -- that can be composed for different experiments but for now training is driven by the command line:

```bash
uv run train \
\
    dataset=librispeech \
    dataset.subset=train-clean-100 \
    dataloader.batch_size=64 \
\
    eval_dataset=librispeech \
    eval_dataset.subset=dev-clean \
    eval_dataloader.batch_size=64 \
    eval_dataloader.shuffle=false \
\
    system=ctc \
\
    system/encoder/frontend=conv \
    'system.encoder.frontend.layers=[{out_channels: 256, kernel_size: 3, stride: 2, norm: bn, activation: silu}, {out_channels: 512, kernel_size: 3, stride: 2, norm: bn, activation: silu}]' \
\
    system/encoder/stem=transformer \
    system.encoder.stem.d_model=512 \
    system.encoder.stem.depth=12 \
    system.encoder.stem.n_head=8 \
    system.encoder.stem.is_causal=false \
    system.encoder.stem.d_ff=2048 \
\
    tokenizer=char \
    'tokenizer.specials=["<blank>"]' \
\
    total_steps=10_000 \
    eval_every=500 \
    optim.lr=0.0003 \
    max_grad_norm=2.0 \
    device="cuda:1" \
    logger=tqdm
```