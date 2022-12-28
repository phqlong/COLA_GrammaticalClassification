import torch
import hydra
import logging
import wandb
from omegaconf.omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime, timezone, timedelta

from datamodule import ColaDatamodule
from model import ColaModel
from callback import LogPredictionsCallback

# Monitoring by Tensorboard
# python -m tensorboard.main --logdir_spec=logs/

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    logger.info("Configs tree:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.model_checkpoint}")

    # Initialize DataModule & Model
    cola_data = ColaDatamodule(
        model_checkpoint=cfg.model.model_checkpoint, 
        data_dir=cfg.processing.data_dir,
        num_workers=cfg.processing.num_workers, 
        batch_size=cfg.processing.batch_size, 
        max_length=cfg.processing.max_length
    )
    cola_model = ColaModel(
        model_checkpoint=cfg.model.model_checkpoint,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr)

    # Initialize Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.model_checkpoint_dir, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    sample_logger_callback = LogPredictionsCallback()

    # Initialize Wandb loger
    wandb_logger = WandbLogger(
        project="CoLA Gramartical Classification - MLOps Basics",
        name=datetime.now(tz=timezone(+timedelta(hours=7))).strftime("%Y%m%d-%Hh%Mm%Ss"),
        log_model='all'  # log all new checkpoints during training
        ) 

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=(1 if torch.cuda.is_available() else 0),

        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        
        fast_dev_run=cfg.training.fast_dev_run,
        deterministic=cfg.training.deterministic,
        default_root_dir=cfg.training.log_dir,

        logger=[wandb_logger, pl.loggers.TensorBoardLogger("logs/", name="cola", version=1)],
        callbacks=[checkpoint_callback, 
                    early_stopping_callback, 
                    sample_logger_callback],
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()

if __name__ == "__main__":
    main()
