# ASR

An experimental repo to play around with automatic speech recognition (ASR) papers and ideas.

## Usage

Training is driven by [Hydra](https://hydra.cc/docs/intro/).

Reusable building blocks and system configs live under [`src/asr/conf/`](src/asr/conf). A training run requires selecting a system config (`system=ctc` / `system=rnnt`, which bundles the model components and loss), the data configs, and the remaining per-run settings via the command line. 

The command line permits config file settings to be overridden and, if necessary, a full training run to be configured (although it's verbose!).

### CTC

```
uv run train \
    dataset=ls960 augment=specaug dataloader=bucket \
    eval_dataset=dev_clean eval_dataloader=dev \
    system=ctc \
    tokenizer=char 'tokenizer.specials=["<blank>"]' \
    total_steps=10_000 eval_every=500 optim.lr=0.0003 max_grad_norm=2.0 \
    device="cuda:1" logger=tqdm
```

### RNN-T

```
uv run train \
    dataset=ls960 augment=specaug dataloader=bucket \
    eval_dataset=dev_clean eval_dataloader=dev \
    system=rnnt \
    tokenizer=char 'tokenizer.specials=["<blank>", "<sos>"]' \
    total_steps=10_000 eval_every=500 optim.lr=0.0003 max_grad_norm=2.0 \
    device="cuda:1" logger=tqdm
```

### TDT

```
uv run train \
    dataset=ls960 augment=specaug dataloader=bucket \
    eval_dataset=dev_clean eval_dataloader=dev \
    system=tdt \
    tokenizer=char 'tokenizer.specials=["<blank>", "<sos>"]' \
    total_steps=10_000 eval_every=500 optim.lr=0.0003 max_grad_norm=2.0 \
    device="cuda:1" logger=tqdm
```

## Features

### Loss

- Connectionist Temporal Classification (CTC). [Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf). Implementation in [CUDA](src/asr/loss/csrc/ctc.cu) with PyTorch stable ABI [bindings](src/asr/loss/csrc/ctc_bindings.cpp).
- Transducer / RNN-T. [Paper](https://arxiv.org/abs/1211.3711). Implementation in [CUDA](src/asr/loss/csrc/rnnt.cu) with PyTorch stable ABI [bindings](src/asr/loss/csrc/rnnt_bindings.cpp).
- Token-and-Duration Transducer (TDT). [Paper](https://arxiv.org/abs/2304.06795). Implementation in [CUDA][src/asr/loss/csrc/tdt.cu] with PyTorch stable ABI [bindings](src/asr/loss/csrc/tdt_bindings.cpp).

### Architecture

- Transformer (RoPE, SwiGLU, Prenorm). [Implementation](src/asr/model/transformer.py).
- LSTM stack. Wraps PyTorch, [implementation](src/asr/model/lstm.py).
- Conv subsampling frontend, [implementation](src/asr/model/encoder.py).

### Dataset

- LibriSpeech. [Website](https://www.openslr.org/12).

### Data Normalization

- "Global" normalization. Per-feature normalization using pre-computed per-feature mean and std.
- Per sample per feature normalization. Mean and std stats computed over the time dimension for each sample.

### Data Augmentation

- SpecAugment time and frequency masking. [Paper](https://arxiv.org/abs/1904.08779).

### Data Batching

- Dynamic length bucketing with a per-batch frame budget, inspired by [Lhotse](https://github.com/lhotse-speech/lhotse).

### Logging

- tqdm. [Website](https://github.com/tqdm/tqdm).
- Aim. [Website](https://github.com/aimhubio/aim).