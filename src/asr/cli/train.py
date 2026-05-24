import hydra
from omegaconf import DictConfig

# Trigger ConfigStore.store(...) so Hydra knows schemas
import asr.config  # noqa: F401
from asr.train import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    normalize = hydra.utils.instantiate(cfg.normalize)
    augment = hydra.utils.instantiate(cfg.augment)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    if cfg.dataset._target_.endswith("BucketDataset"):
        inner = [hydra.utils.instantiate(d, tokenizer=tokenizer) for d in cfg.dataset.datasets]
        dataset = hydra.utils.instantiate(
            cfg.dataset, datasets=inner, normalize=normalize, augment=augment, _recursive_=False
        )
    else:
        dataset = hydra.utils.instantiate(cfg.dataset, normalize=normalize, augment=augment, tokenizer=tokenizer)

    loader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    eval_dataset = hydra.utils.instantiate(cfg.eval_dataset, normalize=normalize, tokenizer=tokenizer)
    eval_loader = hydra.utils.instantiate(cfg.eval_dataloader, dataset=eval_dataset)
    system = hydra.utils.instantiate(cfg.system, n_vocab=len(tokenizer), tokenizer=tokenizer).to(cfg.device)
    optim = hydra.utils.instantiate(cfg.optim, params=system.parameters())
    sched = hydra.utils.instantiate(cfg.sched, optimizer=optim)
    logger = hydra.utils.instantiate(cfg.logger)

    trainer = Trainer(loader, eval_loader, system, optim, sched, logger, cfg.device, cfg.max_grad_norm)
    trainer.train(cfg.total_steps, cfg.eval_steps, cfg.eval_every)


if __name__ == "__main__":
    main()
