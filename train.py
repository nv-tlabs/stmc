import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

logger = logging.getLogger(__name__)


def save_stats(dataset, run_dir):
    is_training = dataset.is_training
    # don't drop anything
    dataset.is_training = False

    motion_stats_dir = os.path.join(run_dir, "motion_stats")
    os.makedirs(motion_stats_dir, exist_ok=True)

    text_stats_dir = os.path.join(run_dir, "text_stats")
    os.makedirs(text_stats_dir, exist_ok=True)

    from tqdm import tqdm
    import torch
    from src.normalizer import Normalizer

    logger.info("Compute motion embedding stats")
    motionfeats = torch.cat([x["x"] for x in tqdm(dataset)])
    mean_motionfeats = motionfeats.mean(0)
    std_motionfeats = motionfeats.std(0)

    motion_normalizer = Normalizer(base_dir=motion_stats_dir, disable=True)
    motion_normalizer.save(mean_motionfeats, std_motionfeats)

    logger.info("Compute text embedding stats")
    textfeats = torch.cat([x["tx"]["x"] for x in tqdm(dataset)])
    mean_textfeats = textfeats.mean(0)
    std_textfeats = textfeats.std(0)

    text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
    text_normalizer.save(mean_textfeats, std_textfeats)

    # re enable droping
    dataset.is_training = is_training


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        resume_dir = cfg.resume_dir
        max_epochs = cfg.trainer.max_epochs
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        cfg.trainer.max_epochs = max_epochs
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{resume_dir}")
    else:
        resume_dir = None
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")

    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")

    if resume_dir is not None:
        logger.info("Computing statistics")
        save_stats(train_dataset, cfg.run_dir)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
