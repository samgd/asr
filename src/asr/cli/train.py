import hydra
from omegaconf import DictConfig, OmegaConf

# Trigger ConfigStore.store(...) so Hydra knows schemas
import asr.config  # noqa: F401


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    loss = hydra.utils.instantiate(cfg.loss)
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    data = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer)
    model = hydra.utils.instantiate(cfg.model, n_vocab=len(tokenizer))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
