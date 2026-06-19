import os

import git
import hydra
import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from models.lstm import LSTMClassifier
from scripts.modules.data_module import NewsDataModule
from scripts.modules.plot_callback import PlotCallback

mlflow.pytorch.autolog()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    dm = NewsDataModule(cfg)
    dm.setup()

    model = LSTMClassifier(
        vocab_size=dm.vocab_size,
        num_classes=dm.num_classes,
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

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="auto",
        logger=mlf_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)

    os.makedirs("plots", exist_ok=True)

    metrics = trainer.callback_metrics

    print("Logged metrics:", trainer.logged_metrics)
    print("Callback metrics:", trainer.callback_metrics)
    print("MLflow run ID:", mlf_logger.run_id)


if __name__ == "__main__":
    main()
