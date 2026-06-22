import subprocess
from pathlib import Path

import git
import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from news_classifier.data.data_module import NewsDataModule
from news_classifier.data.download_data import download_data
from news_classifier.data.ensure_data import ensure_data
from news_classifier.models.lstm import LSTMClassifier


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    ensure_data(cfg)

    dm = NewsDataModule(cfg)
    dm.setup()

    model = LSTMClassifier(
        vocab_size=dm.vocab_size,
        num_classes=dm.num_classes,
        class_weights=dm.class_weights,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        lr=cfg.train.lr,
    )

    model.loss_fn = torch.nn.CrossEntropyLoss(weight=dm.class_weights)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    mlf_logger.log_hyperparams(cfg)

    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha

    mlf_logger.experiment.log_param(mlf_logger.run_id, "git_commit", commit_id)

    early_stop = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=5,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=1.0,
        accelerator="auto",
        logger=mlf_logger,
        log_every_n_steps=10,
        callbacks=[early_stop, lr_monitor, checkpoint_callback],
    )

    trainer.fit(model, dm)

    metrics = trainer.callback_metrics

    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.item()

        mlf_logger.experiment.log_metric(
            mlf_logger.run_id,
            name,
            value,
        )

    print("Logged metrics:", trainer.logged_metrics)
    print("Callback metrics:", trainer.callback_metrics)
    print("MLflow run ID:", mlf_logger.run_id)


if __name__ == "__main__":
    main()
