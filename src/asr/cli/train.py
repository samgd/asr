import hydra
from omegaconf import DictConfig

# Trigger ConfigStore.store(...) so Hydra knows schemas
import asr.config  # noqa: F401
from asr.train import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    dataset = hydra.utils.instantiate(cfg.dataset, tokenizer=tokenizer)
    loader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    eval_dataset = hydra.utils.instantiate(cfg.eval_dataset, tokenizer=tokenizer)
    eval_loader = hydra.utils.instantiate(cfg.eval_dataloader, dataset=eval_dataset)
    system = hydra.utils.instantiate(cfg.system, n_vocab=len(tokenizer), tokenizer=tokenizer).to(cfg.device)
    optim = hydra.utils.instantiate(cfg.optim, params=system.parameters())

    Trainer(loader, eval_loader, system, optim, cfg.device).train(cfg.total_steps, cfg.eval_steps, cfg.eval_every)


if __name__ == "__main__":
    main()
