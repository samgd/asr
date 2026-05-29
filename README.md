# ASR

An experimental repo to play around with automatic speech recognition (ASR) papers and ideas.

## Usage

Training is driven by [Hydra](https://hydra.cc/docs/intro/).

Reusable building blocks and system configs live under [`src/asr/conf/`](src/asr/conf). A training run requires selecting a system config (`system=ctc` / `system=rnnt`, which bundles the model components and loss), the data config, and the remaining per-run settings via the command line. 

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

### RNN-T Example

```
uv run train \
    dataset=ls960 augment=specaug dataloader=bucket \
    eval_dataset=dev_clean eval_dataloader=dev \
    system=rnnt \
    tokenizer=char 'tokenizer.specials=["<blank>", "<sos>"]' \
    total_steps=10_000 eval_every=500 optim.lr=0.0003 max_grad_norm=2.0 \
    device="cuda:1" logger=tqdm
```