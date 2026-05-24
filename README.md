# ASR

An experimental repo to play around with automatic speech recognition (ASR) papers and ideas.

## Usage

Training is driven by [Hydra](https://hydra.cc/docs/intro/).

Eventually there will be YAML files containing standard configurations for each part -- data, model, loss, etc -- that can be composed for different experiments but for now training is driven by the command line:

```
uv run train \
    dataset=bucket \
    dataset.feature_dim=128 \
    'dataset.datasets=[{_target_: asr.data.LibriSpeech, subset: train-clean-100}, {_target_: asr.data.LibriSpeech, subset: train-clean-360}, {_target_: asr.data.LibriSpeech, subset: train-other-500}]' \
    'dataset.weights=[0.195, 0.369, 0.436]' \
    'dataset.boundaries=[191, 928, 1308, 1440, 1532, 1709]' \
    dataset.batch_frame_budget=80000 \
    dataset.max_bucket_count=20000 \
\
    dataloader=bucket \
    dataloader.num_workers=4 \
\
    eval_dataset=librispeech \
    eval_dataset.subset=dev-clean \
\
    eval_dataloader.batch_size=64  \
    eval_dataloader.shuffle=false \
\
    system=ctc \
    system.decode=beam \
    system.loss.zero_infinity=true \
\
    system/encoder/normalize=dynamicpersampleperfeaturenorm \
\
    system/encoder/frontend=conv \
    'system.encoder.frontend.layers=[{out_channels: 256, kernel_size: 3, stride: 2, norm: ln, activation: silu}, {out_channels: 512, kernel_size: 3, stride: 2, norm: ln, activation: silu}]' \
\
    system/encoder/stem=transformer \
    system.encoder.stem.d_model=512  \
    system.encoder.stem.depth=12 \
    system.encoder.stem.n_head=8 \
    system.encoder.stem.is_causal=false \
    system.encoder.stem.d_ff=2048 \
\
    tokenizer=char \
    'tokenizer.specials=["<blank>"]' \
\
    total_steps=10_000 \
    eval_steps=10 \
    eval_every=500 \
\
    optim.lr=0.0001 \
    max_grad_norm=2.0 \
\
    device="cuda:1" \
    logger=tqdm
```