import os

import git
import hydra
import lightning as pl
import lightning.pytorch.loggers as pl_loggers
import matplotlib.pyplot as plt
import mlflow
import torch

from models.lstm import LSTMClassifier
from scripts.modules.data_module import NewsDataModule
from scripts.modules.plot_callback import PlotCallback


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    dm = NewsDataModule(cfg)

    model = LSTMClassifier(
        vocab_size=len(dm.word2idx),
        num_classes=dm.num_classes,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
    )

    model.loss_fn = torch.nn.CrossEntropyLoss(weight=dm.class_weights)

    mlf_logger = pl_loggers.MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    mlf_logger.log_hyperparams(dict(cfg))

    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha

    mlf_logger.experiment.log_param(mlf_logger.run_id, "git_commit", commit_id)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="auto",
        logger=mlf_logger,
        log_every_n_steps=10,
        callbacks=[PlotCallback()],
    )

    trainer.fit(model, dm)

    os.makedirs("plots", exist_ok=True)

    metrics = trainer.callback_metrics


if __name__ == "__main__":
    main()
